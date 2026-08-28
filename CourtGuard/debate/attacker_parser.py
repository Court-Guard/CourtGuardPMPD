"""
Attacker Output Parser

Extract the structured accusation from the PMPD attacker agent's output.

The attacker/selector may emit either the legacy accusation block or the new
selection-oriented block:
    CANDIDATE_CATEGORIES: S9, S6
    EVIDENCE_SPANS: "quote"; "quote"

The output label is now taxonomy-aware through OutputMapper, but the parser
still accepts legacy numeric PMPD labels 0/1 for backward compatibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evaluation.output_mapper import OutputMapper


@dataclass
class ParsedAccusation:
    """
    Structured output extracted from the attacker response.

    Attributes
    ----------
    accusation:
        Legacy accusation text or a compact selection summary.
    categories:
        Top-level PMPD category IDs selected by the attacker.
    preliminary_label:
        One configured output label from the active taxonomy.
    raw_output:
        The full raw model output.
    parse_success:
        True when the structured block was parsed cleanly.
    default_label:
        The configured compliant/default label for this run.
    evidence:
        Short grounding text extracted from the structured block when present.
    had_category_text:
        Whether the raw structured block included a non-empty CATEGORIES field.
    invalid_category_mismatch:
        True when category text was present but none of it matched live PMPD IDs.
    """

    accusation: str
    categories: list[str]
    preliminary_label: str
    raw_output: str
    parse_success: bool = True
    default_label: str = "SAFE"
    evidence: str = ""
    had_category_text: bool = False
    invalid_category_mismatch: bool = False

    @property
    def is_accusation(self) -> bool:
        """True when the attacker flagged a non-default label or categories."""
        return self.preliminary_label != self.default_label or bool(self.categories)

    @classmethod
    def safe_default(
        cls,
        raw_output: str,
        default_label: str = "SAFE",
    ) -> "ParsedAccusation":
        """Return a conservative default/compliant accusation on parse failure."""
        return cls(
            accusation="Parse failure - raw output preserved.",
            categories=[],
            preliminary_label=default_label,
            raw_output=raw_output,
            parse_success=False,
            default_label=default_label,
            evidence="",
            had_category_text=False,
            invalid_category_mismatch=False,
        )


class AttackerOutputParser:
    """
    Parse the PMPD attacker output into a ParsedAccusation.

    Strategy:
      Pass 1 - strict structured selector or accusation block
      Pass 2 - relaxed line-based parsing
      Pass 3 - free-text inference

    The parser never raises. Raw output is always preserved.
    """

    _ACCUSATION_PAT = re.compile(
        r"ACCUSATION[:\s]+(.+?)(?=EVIDENCE[:\s]|EVIDENCE_SPANS[:\s]|CATEGORIES[:\s]|CANDIDATE_CATEGORIES[:\s]|PRELIMINARY_LABEL[:\s]|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _EVIDENCE_PAT = re.compile(
        r"(?:EVIDENCE|EVIDENCE_SPANS)[:\s]+(.+?)(?=CATEGORIES[:\s]|CANDIDATE_CATEGORIES[:\s]|PRELIMINARY_LABEL[:\s]|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _CATEGORIES_PAT = re.compile(
        r"(?:CATEGORIES|CANDIDATE_CATEGORIES):[ \t]*([^\n]*)",
        re.IGNORECASE,
    )
    _LABEL_PAT = re.compile(r"PRELIMINARY_LABEL[:\s]+([^\n]+)", re.IGNORECASE)

    def __init__(
        self,
        valid_categories: list[str] | None = None,
        output_mapper: OutputMapper | None = None,
    ) -> None:
        self._valid = set(valid_categories or [])
        self._output_mapper = output_mapper or OutputMapper(OutputMapper.DEFAULT_LABELS)

    def parse(self, raw: str) -> ParsedAccusation:
        """Parse one attacker response into a structured accusation."""
        if not raw or not raw.strip():
            return ParsedAccusation.safe_default(
                raw or "",
                default_label=self._output_mapper.default_label,
            )

        result = self._parse_structured(raw)
        if result is not None:
            result.raw_output = raw
            return result

        result = self._parse_relaxed(raw)
        if result is not None:
            result.raw_output = raw
            return result

        result = self._parse_inferred(raw)
        result.raw_output = raw
        return result

    def _parse_structured(self, raw: str) -> ParsedAccusation | None:
        """Extract categories and label from the structured block."""
        label_match = self._LABEL_PAT.search(raw)
        cats_match = self._CATEGORIES_PAT.search(raw)
        evidence_match = self._EVIDENCE_PAT.search(raw)

        if not label_match and not cats_match and not evidence_match:
            return None

        raw_label = label_match.group(1).strip() if label_match else ""
        raw_categories = cats_match.group(1) if cats_match else ""
        categories = self._parse_categories(raw_categories)
        had_category_text = bool(raw_categories.strip())
        category_mismatch = had_category_text and not categories

        evidence = self._clean_field(evidence_match.group(1)) if evidence_match else ""
        label = self._output_mapper.normalize_label(
            raw_label,
            fallback=(
                self._output_mapper.flagged_label()
                if categories
                else self._output_mapper.default_label
            ),
        )
        acc_match = self._ACCUSATION_PAT.search(raw)
        accusation = acc_match.group(1).strip() if acc_match else self._build_selection_summary(
            categories=categories,
            evidence=evidence,
        )
        if category_mismatch:
            accusation = (
                f"{accusation}\n"
                "[Parser warning: structured category output did not match valid PMPD IDs.]"
            )

        return ParsedAccusation(
            accusation=accusation,
            categories=categories,
            preliminary_label=label,
            raw_output="",
            parse_success=not category_mismatch,
            default_label=self._output_mapper.default_label,
            evidence=evidence,
            had_category_text=had_category_text,
            invalid_category_mismatch=category_mismatch,
        )

    def _parse_relaxed(self, raw: str) -> ParsedAccusation | None:
        """Relaxed parsing when the output is present but not perfectly formatted."""
        label_hints = re.findall(
            r"(?:preliminary_label|label|verdict|result|decision)[:\s]+([^\n]+)",
            raw,
            re.IGNORECASE,
        )
        if not label_hints:
            return None

        label = self._output_mapper.normalize_label(label_hints[-1])
        categories = self._scan_for_categories(raw)
        evidence_match = self._EVIDENCE_PAT.search(raw)
        evidence = self._clean_field(evidence_match.group(1)) if evidence_match else ""
        cats_match = self._CATEGORIES_PAT.search(raw)
        raw_categories = cats_match.group(1) if cats_match else ""
        had_category_text = bool(raw_categories.strip())
        category_mismatch = had_category_text and not categories
        parse_success = (bool(categories) or label == self._output_mapper.default_label) and not category_mismatch

        return ParsedAccusation(
            accusation=self._build_selection_summary(
                categories=categories,
                evidence=evidence,
                fallback=raw,
            ),
            categories=categories,
            preliminary_label=label,
            raw_output="",
            parse_success=parse_success,
            default_label=self._output_mapper.default_label,
            evidence=evidence,
            had_category_text=had_category_text,
            invalid_category_mismatch=category_mismatch,
        )

    def _parse_inferred(self, raw: str) -> ParsedAccusation:
        """Infer accusation state from categories and free-text violation keywords."""
        categories = self._scan_for_categories(raw)

        raw_lower = raw.lower()
        violation_keywords = [
            "violat", "unsafe", "harmful", "prohibited", "illegal",
            "non-compli", "breach", "forbidden",
        ]
        safe_keywords = [
            "safe", "compliant", "no violation", "allowed", "permitted",
        ]

        has_violation = any(k in raw_lower for k in violation_keywords)
        has_safe = any(k in raw_lower for k in safe_keywords)

        if categories or (has_violation and not has_safe):
            label = self._output_mapper.flagged_label()
        else:
            label = self._output_mapper.default_label

        return ParsedAccusation(
            accusation=self._build_selection_summary(
                categories=categories,
                evidence="",
                fallback=raw,
            ),
            categories=categories,
            preliminary_label=label,
            raw_output="",
            parse_success=False,
            default_label=self._output_mapper.default_label,
            evidence="",
            had_category_text=False,
            invalid_category_mismatch=False,
        )

    def _parse_categories(self, cats_str: str) -> list[str]:
        """Parse a category string like 'S9, S6' into a validated list."""
        if not cats_str or cats_str.strip().lower() in ("none", ""):
            return []

        raw_cats = re.split(r"[,;\s]+", cats_str.strip())
        return self._validate_categories(raw_cats)

    def _scan_for_categories(self, text: str) -> list[str]:
        """Scan free text for known category codes."""
        if not self._valid:
            matches = re.findall(r"\b([A-Z]{2,6}(?:_[A-Z]{2,6})?)\b", text)
            return list(dict.fromkeys(matches))

        found: list[str] = []
        for code in self._valid:
            if re.search(r"\b" + re.escape(code) + r"\b", text, re.IGNORECASE):
                found.append(code)
        return found

    def _validate_categories(self, raw: list[str]) -> list[str]:
        """Filter category strings against valid_categories."""
        cleaned = [c.strip().upper() for c in raw if c.strip()]

        if not self._valid:
            filtered = [c for c in cleaned if re.match(r"^[A-Z]{2,6}(_[A-Z]{2,6})?$", c)]
            return list(dict.fromkeys(filtered))[:2]

        filtered = [c for c in cleaned if c in self._valid]
        return list(dict.fromkeys(filtered))[:2]

    @staticmethod
    def _clean_field(value: str) -> str:
        """Normalise optional structured fields like EVIDENCE."""
        return value.strip().strip('"').strip("'").strip()

    @staticmethod
    def _build_selection_summary(
        categories: list[str],
        evidence: str,
        fallback: str = "",
    ) -> str:
        categories_text = ", ".join(categories) if categories else "(none)"
        evidence_text = AttackerOutputParser._clean_field(evidence) or "(none)"
        if categories or evidence_text != "(none)":
            return f"Selected categories: {categories_text}. Evidence spans: {evidence_text}"
        return fallback[:500]
