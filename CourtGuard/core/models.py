"""
Core Domain Models — PMPD Data Structures

Extracted from pmpd.py to satisfy SRP.  This module contains only
pure data classes and their rendering logic — zero I/O, zero persistence.

Classes
───────
  OneShot        — a concrete policy interaction example (shadow or live)
  GlobalModule   — system-wide rules that apply to every evaluation
  CategoryModule — domain-level risk category with rules, exceptions, shots

All three classes use @dataclass for construction and expose a
to_prompt_fragment() method for LLM prompt injection — the rendering
contract used by PMPDAssembler.

Enumerations used here (Verdict, OneShotSource) are defined in core/enums.py
so that both models and the wider application share the same types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.enums import OneShotPolarity, OneShotSource, Verdict


# ---------------------------------------------------------------------------
# OneShot
# ---------------------------------------------------------------------------


@dataclass
class OneShot:
    """
    A concrete example of a policy interaction — either a violation or a
    compliant response — used to ground LLM understanding of the policy.

    Attributes
    ----------
    example_id:   Unique identifier (e.g. "S1_001", "shadow_005")
    query:        The user query or attacker prompt that triggered the case
    response:     The AI response that was evaluated
    verdict:      "violation" | "compliant" | "borderline"
    reasoning:    Why the verdict was reached (from Judge Agent or author)
    severity:     1 (safe) – 5 (critical violation); 0 = not applicable
    source:       "shadow"  → pre-written seed example
                  "live"    → produced by Judge Agent during evaluation
    category_id:  Which CategoryModule this belongs to (e.g. "S1")
    timestamp:    Unix timestamp of when this was added
    """

    example_id: str
    query: str
    response: str
    verdict: str  # Verdict enum value
    reasoning: str
    severity: int = 0  # 1–5; 0 = unscored
    source: str = OneShotSource.SHADOW
    polarity: str = OneShotPolarity.POSITIVE
    category_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_prompt_fragment(self) -> str:
        """
        Render this One-Shot as a compact string suitable for injection
        into an LLM prompt.
        """
        lines = [
            f"[{self.source.upper()} ONE-SHOT | {self.category_id} | {self.verdict.upper()}"
            + (f" | Severity {self.severity}/5" if self.severity > 0 else "")
            + "]",
            f"Query    : {self.query}",
            f"Response : {self.response}",
            f"Reasoning: {self.reasoning}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GlobalModule
# ---------------------------------------------------------------------------


@dataclass
class GlobalModule:
    """
    System-wide rules that apply to every evaluation, regardless of category.

    Extracted once from the Governance Corpora by the PMPD Parser.

    Attributes
    ----------
    objective:   The primary goal of the safety evaluation
    principles:  Baseline ethical / operational standards always in force
    schema:      Required output format for the Judge Agent
                 (e.g. JSON keys, severity scale definition)
    source_doc:  Name / path of the governance document this was derived from
    last_updated: Unix timestamp of last update
    """

    objective: str
    principles: list[str]
    schema: str  # prose description of Judge output format
    source_doc: str = ""
    last_updated: float = field(default_factory=time.time)
    definitions: dict[str, str] = field(default_factory=dict)
    general_eval_principles: list[str] = field(default_factory=list)
    evaluation_protocol: list[str] = field(default_factory=list)
    positive_examples: list[dict[str, str]] = field(default_factory=list)
    negative_examples: list[dict[str, str]] = field(default_factory=list)

    def to_prompt_fragment(self) -> str:
        """Render for injection into prompts."""
        principles_text = "\n".join(f"  • {p}" for p in self.principles)
        protocol_text = "\n".join(f"  - {p}" for p in self.evaluation_protocol)
        lines = [
            "=== GLOBAL MODULE ===",
            f"OBJECTIVE  : {self.objective}",
            f"PRINCIPLES :\n{principles_text}",
        ]
        if protocol_text:
            lines.append(f"EVALUATION PROTOCOL :\n{protocol_text}")
        lines.append(f"OUTPUT SCHEMA:\n{self.schema}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CategoryModule
# ---------------------------------------------------------------------------


@dataclass
class CategoryModule:
    """
    Domain-level risk category extracted from the Governance Corpora.

    One CategoryModule corresponds to one identified risk/violation class
    (e.g. S1: Defamation, S2: Specialized Advice).

    Attributes
    ----------
    category_id:    Short label, e.g. "S1", "S2"
    name:           Human-readable name, e.g. "Defamation"
    general_rules:  Core definition and boundary of the violation
    exceptions:     Precise conditions where the rule does NOT apply
    one_shots:      List[OneShot] — seed (shadow) + accumulated live examples
    citations:      References back to the source Governance Corpora
    source_doc:     Which governance document this came from
    last_updated:   Unix timestamp of last update
    """

    category_id:   str
    name:          str
    general_rules: str
    exceptions:    list[str]
    one_shots:     list[OneShot]
    citations:     list[str]
    source_doc:    str   = ""
    last_updated:  float = field(default_factory=time.time)
    short_definition: str        = ""                           # ← new, after all required
    sub_categories:   list[dict] = field(default_factory=list)  # ← new, after all required


    # ------------------------------------------------------------------
    # One-Shot management
    # ------------------------------------------------------------------

    def add_one_shot(self, one_shot: OneShot) -> None:
        """Append a One-Shot and stamp the module as updated."""
        one_shot.category_id = self.category_id
        self.one_shots.append(one_shot)
        self.last_updated = time.time()

    def get_shadow_shots(self) -> list[OneShot]:
        """Return only pre-written seed examples."""
        return [s for s in self.one_shots if s.source == OneShotSource.SHADOW]

    def get_live_shots(self, limit: int = 5) -> list[OneShot]:
        """
        Return the most recent live examples (Judge Agent verdicts).
        Limited to `limit` most recent to keep prompts from growing unbounded.
        """
        live = [s for s in self.one_shots if s.source == OneShotSource.LIVE]
        return sorted(live, key=lambda s: s.timestamp, reverse=True)[:limit]

    def get_shots_for_prompt(
        self,
        max_shadow: int = 2,
        max_live: int = 3,
    ) -> list[OneShot]:
        """
        Return a balanced mix of shadow + live shots for prompt assembly.
        Newest live shots take priority.
        """
        return self.get_shadow_shots()[:max_shadow] + self.get_live_shots(max_live)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_prompt_fragment(
        self,
        include_one_shots: bool = True,
        max_shadow: int = 2,
        max_live: int = 3,
    ) -> str:
        """Render this module for injection into an LLM prompt."""
        exceptions_text = "\n".join(f"  • {e}" for e in self.exceptions) or "  (none)"
        citations_text = (
            "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(self.citations)) or "  (none)"
        )

        lines = [
            f"=== CATEGORY MODULE [{self.category_id}] : {self.name} ===",
            f"GENERAL RULES:\n{self.general_rules}",
            f"EXCEPTIONS:\n{exceptions_text}",
            f"CITATIONS:\n{citations_text}",
        ]

        if include_one_shots:
            shots = self.get_shots_for_prompt(max_shadow, max_live)
            if shots:
                lines.append("ONE-SHOTS:")
                for shot in shots:
                    lines.append(shot.to_prompt_fragment())

        return "\n\n".join(lines)
