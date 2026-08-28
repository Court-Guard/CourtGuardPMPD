"""
PMPD Debate Engine

Runtime semantics for PMPD mode.

The public attacker/defender naming is preserved for compatibility with the
rest of the codebase, but the internal runtime now behaves as:

  attacker -> selector
  defender -> specialist

Selector
--------
Receives the GlobalModule plus all categories in lite form and nominates the
single strongest plausible top-level category plus short evidence spans.

Specialist
----------
Receives the original input plus the full PMPD detail for ONLY the selected
categories and makes the scoped final judgment. The specialist may not invent
or use categories that were not handed to it.

Round 2
-------
Round 2 is retrieval repair, not argumentative backdown:
  - it is triggered only when the specialist reports CATEGORY_FIT: UNSUPPORTED
  - the selector is shown only compact category-fit feedback
  - the selector does not receive attacker-style reasoning from the prior round

Parsing
-------
Both selector and specialist get a single parse-repair retry if their first
response does not follow the structured contract. Retry asks for reformatting
only and keeps the same model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.enums import DebateOutcome
from core.pmpd import PMPD
from debate.attacker_parser import AttackerOutputParser, ParsedAccusation
from evaluation.output_mapper import OutputMapper
from evaluation.token_tracker import TokenTracker
from infrastructure.api_client import APIClient, APIError
from infrastructure.llm_retry_client import LLMRetryClient, RetryConfig
from prompts.harmony_builder import AgentPrompt, HarmonyPromptBuilder


class _DefenderOutputParser:
    """
    Parse the specialist's structured output.

    Expected format:
        CATEGORY_FIT: SUPPORTED | UNSUPPORTED
        FINAL_LABEL: SAFE | UNSAFE | ...
        CATEGORIES: S9, S6
        BASIS_CODE: MATCHED_POLICY | ...
        CATEGORY_NOTES: ...
        REASONING: ...
    """

    _FIT_PAT = re.compile(r"CATEGORY_FIT[:\s]+(SUPPORTED|UNSUPPORTED)", re.IGNORECASE)
    _CATEGORIES_PAT = re.compile(
        r"(?:CATEGORIES|CONFIRMED_CATEGORIES):[ \t]*([^\n]*)",
        re.IGNORECASE,
    )
    _FINAL_LABEL_PAT = re.compile(r"(?:FINAL_LABEL|LABEL)[:\s]+([^\n]+)", re.IGNORECASE)
    _BASIS_PAT = re.compile(r"(?:BASIS_CODE|OVERTURN_BASIS)[:\s]+([^\n]*)", re.IGNORECASE)
    _CATEGORY_NOTES_PAT = re.compile(
        r"CATEGORY_NOTES[:\s]+(.+?)(?=REASONING[:\s]|CATEGORY_FIT[:\s]|FINAL_LABEL[:\s]|LABEL[:\s]|BASIS_CODE[:\s]|OVERTURN_BASIS[:\s]|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _REASONING_PAT = re.compile(
        r"REASONING[:\s]+(.+?)(?=CATEGORY_FIT[:\s]|FINAL_LABEL[:\s]|LABEL[:\s]|BASIS_CODE[:\s]|OVERTURN_BASIS[:\s]|CATEGORY_NOTES[:\s]|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _BASIS_CHOICES = {
        "MATCHED_POLICY",
        "EXCEPTION_APPLIES",
        "INSUFFICIENT_COVERAGE",
        "INSUFFICIENT_EVIDENCE",
        "NO_VIOLATION",
        "OTHER",
        "UNKNOWN",
    }

    def __init__(self, output_mapper: OutputMapper | None = None) -> None:
        self._output_mapper = output_mapper or OutputMapper(OutputMapper.DEFAULT_LABELS)

    def parse(
        self,
        raw: str,
        valid_categories: list[str] | None = None,
        fallback_label: str | None = None,
    ) -> dict[str, Any]:
        if not raw or not raw.strip():
            return self._default(
                raw or "",
                final_label=fallback_label or self._output_mapper.default_label,
                parse_success=False,
            )

        fit_match = self._FIT_PAT.search(raw)
        if not fit_match:
            return self._default(
                raw,
                final_label=fallback_label or self._output_mapper.default_label,
                parse_success=False,
            )

        category_fit = fit_match.group(1).upper()
        supported = category_fit == "SUPPORTED"

        cats_match = self._CATEGORIES_PAT.search(raw)
        cats_str = cats_match.group(1).strip() if cats_match else ""
        categories = self._parse_cats(cats_str, valid_categories)
        had_category_text = bool(cats_str)
        category_mismatch = had_category_text and not categories

        label_match = self._FINAL_LABEL_PAT.search(raw)
        raw_label = label_match.group(1).strip() if label_match else ""
        final_label = self._output_mapper.normalize_label(
            raw_label,
            fallback=fallback_label or self._output_mapper.default_label,
        )

        basis_match = self._BASIS_PAT.search(raw)
        raw_basis = basis_match.group(1).strip() if basis_match else ""
        basis_code = self._normalise_basis(raw_basis)

        notes_match = self._CATEGORY_NOTES_PAT.search(raw)
        category_notes = notes_match.group(1).strip() if notes_match else ""

        reasoning_match = self._REASONING_PAT.search(raw)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else raw[:800]
        if category_mismatch:
            reasoning = (
                f"{reasoning}\n"
                "[Parser warning: structured category output did not match the category scope provided to the specialist.]"
            )

        return {
            "supported": supported,
            "confirmed": supported and final_label != self._output_mapper.default_label,
            "categories": categories,
            "final_label": final_label,
            "category_fit": category_fit,
            "basis_code": basis_code,
            "overturn_basis": basis_code,
            "category_notes": category_notes[:400],
            "reasoning": reasoning[:800],
            "raw_output": raw,
            "parse_success": not category_mismatch,
            "had_category_text": had_category_text,
            "invalid_category_mismatch": category_mismatch,
        }

    def _default(
        self,
        raw: str,
        final_label: str,
        parse_success: bool,
    ) -> dict[str, Any]:
        return {
            "supported": False,
            "confirmed": False,
            "categories": [],
            "final_label": final_label,
            "category_fit": "UNSUPPORTED",
            "basis_code": "UNKNOWN",
            "overturn_basis": "UNKNOWN",
            "category_notes": "",
            "reasoning": raw[:800],
            "raw_output": raw,
            "parse_success": parse_success,
            "had_category_text": False,
            "invalid_category_mismatch": False,
        }

    @classmethod
    def _normalise_basis(cls, raw_basis: str) -> str:
        cleaned = raw_basis.strip().upper().replace("-", "_").replace(" ", "_")
        if cleaned in cls._BASIS_CHOICES:
            return cleaned
        return "UNKNOWN"

    @staticmethod
    def _parse_cats(cats_str: str, valid: list[str] | None) -> list[str]:
        if not cats_str or cats_str.lower() in {"none", ""}:
            return []

        raw = [c.strip().upper() for c in re.split(r"[,;\s]+", cats_str) if c.strip()]
        if not valid:
            return raw

        valid_set = set(valid)
        return [c for c in raw if c in valid_set]


@dataclass
class PMPDDebateResult:
    outcome: str
    final_label: str
    confirmed_categories: list[str]
    rounds_configured: int
    rounds_run: int
    attacker_rounds: list[dict] = field(default_factory=list)
    defender_rounds: list[dict] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    debate_ended_early: bool = False
    api_error: bool = False
    error_message: str = ""
    judge_rounds: list[dict] = field(default_factory=list)
    runtime_strategy: str = "debate"
    retrieved_categories: list[str] = field(default_factory=list)
    aggregation: dict[str, Any] = field(default_factory=dict)


class PMPDDebate:
    """Two-agent PMPD runtime with retrieval-repair on round 2."""

    def __init__(
        self,
        api_client: APIClient,
        db: PMPD,
        max_rounds: int = 2,
        attacker_model: str | None = None,
        defender_model: str | None = None,
        judge_model: str | None = None,
        output_mapper: OutputMapper | None = None,
    ) -> None:
        import os

        default_model = os.getenv("COURTGUARD_DEBATE_MODEL", "")

        self._attacker_model = (
            os.getenv("COURTGUARD_ATTACKER_MODEL") or attacker_model or default_model
        )
        self._defender_model = (
            os.getenv("COURTGUARD_DEFENDER_MODEL") or defender_model or default_model
        )
        self._judge_model = (
            os.getenv("COURTGUARD_JUDGE_MODEL") or judge_model or default_model
        )

        if not all([self._attacker_model, self._defender_model, self._judge_model]):
            raise ValueError(
                "COURTGUARD_DEBATE_MODEL (or per-role model env vars) must be set."
            )

        self._db = db
        self._max_rounds = max_rounds
        self._retry = LLMRetryClient(api_client, RetryConfig.debate())
        self._api_client = api_client
        self._output_mapper = output_mapper or OutputMapper(OutputMapper.DEFAULT_LABELS)

        self._attacker_parser = AttackerOutputParser(
            valid_categories=list(db.categories.keys()),
            output_mapper=self._output_mapper,
        )
        self._defender_parser = _DefenderOutputParser(output_mapper=self._output_mapper)
        self._prompt_builder = HarmonyPromptBuilder(
            db,
            output_mapper=self._output_mapper,
        )

    def run(self, formatted_input: str, payload: Any | None = None) -> PMPDDebateResult:
        tracker = TokenTracker(
            attacker_model=self._attacker_model,
            defender_model=self._defender_model,
            judge_model=self._judge_model,
        )

        attacker_rounds: list[dict] = []
        defender_rounds: list[dict] = []
        previous_feedback: list[dict[str, Any]] = []

        try:
            for round_num in range(1, self._max_rounds + 1):
                attacker_prompt = self._prompt_builder.build_attacker()
                accusation, attacker_meta = self._run_attacker(
                    attacker_prompt,
                    formatted_input,
                    previous_feedback,
                    tracker,
                )

                attacker_rounds.append(
                    {
                        "round": round_num,
                        "raw_output": attacker_meta["selected_raw_output"],
                        "raw_output_initial": attacker_meta["initial_raw_output"],
                        "repair_raw_output": attacker_meta["repair_raw_output"],
                        "repair_attempted": attacker_meta["repair_attempted"],
                        "repair_successful": attacker_meta["repair_successful"],
                        "accusation": accusation.accusation,
                        "evidence": accusation.evidence,
                        "categories": accusation.categories,
                        "preliminary_label": accusation.preliminary_label,
                        "parse_success": accusation.parse_success,
                        "had_category_text": accusation.had_category_text,
                        "invalid_category_mismatch": accusation.invalid_category_mismatch,
                        **attacker_meta["telemetry"],
                    }
                )

                if not accusation.categories:
                    if (
                        accusation.preliminary_label == self._output_mapper.default_label
                        and not accusation.evidence
                        and not previous_feedback
                    ):
                        return PMPDDebateResult(
                            outcome=DebateOutcome.EARLY_SAFE,
                            final_label=self._output_mapper.default_label,
                            confirmed_categories=[],
                            rounds_configured=self._max_rounds,
                            rounds_run=round_num,
                            attacker_rounds=attacker_rounds,
                            defender_rounds=defender_rounds,
                            token_usage=tracker.summary(),
                            debate_ended_early=True,
                        )

                    if round_num < self._max_rounds:
                        previous_feedback.append(
                            self._build_missing_selection_feedback(round_num, accusation)
                        )
                        continue

                    return PMPDDebateResult(
                        outcome=DebateOutcome.SAFE,
                        final_label=self._output_mapper.default_label,
                        confirmed_categories=[],
                        rounds_configured=self._max_rounds,
                        rounds_run=round_num,
                        attacker_rounds=attacker_rounds,
                        defender_rounds=defender_rounds,
                        token_usage=tracker.summary(),
                        debate_ended_early=False,
                    )

                if previous_feedback:
                    last_feedback = previous_feedback[-1]
                    rejected = last_feedback.get("candidate_categories_list") or []
                    if rejected and self._same_category_set(rejected, accusation.categories):
                        return PMPDDebateResult(
                            outcome=DebateOutcome.SAFE,
                            final_label=self._output_mapper.default_label,
                            confirmed_categories=[],
                            rounds_configured=self._max_rounds,
                            rounds_run=round_num,
                            attacker_rounds=attacker_rounds,
                            defender_rounds=defender_rounds,
                            token_usage=tracker.summary(),
                            debate_ended_early=False,
                        )

                defender_prompt = self._prompt_builder.build_defender(
                    flagged_categories=accusation.categories,
                    previous_verdicts=[],
                )
                verdict, defender_meta = self._run_defender(
                    defender_prompt,
                    formatted_input,
                    accusation,
                    tracker,
                )

                defender_rounds.append(
                    {
                        "round": round_num,
                        "raw_output": defender_meta["selected_raw_output"],
                        "raw_output_initial": defender_meta["initial_raw_output"],
                        "repair_raw_output": defender_meta["repair_raw_output"],
                        "repair_attempted": defender_meta["repair_attempted"],
                        "repair_successful": defender_meta["repair_successful"],
                        "supported": verdict["supported"],
                        "confirmed": verdict["confirmed"],
                        "category_fit": verdict["category_fit"],
                        "categories": verdict["categories"],
                        "final_label": verdict["final_label"],
                        "basis_code": verdict["basis_code"],
                        "overturn_basis": verdict["overturn_basis"],
                        "category_notes": verdict["category_notes"],
                        "reasoning": verdict["reasoning"],
                        "parse_success": verdict["parse_success"],
                        "had_category_text": verdict["had_category_text"],
                        "invalid_category_mismatch": verdict["invalid_category_mismatch"],
                        **defender_meta["telemetry"],
                    }
                )

                if verdict["supported"]:
                    final_label = verdict["final_label"]
                    unsafe = final_label != self._output_mapper.default_label
                    return PMPDDebateResult(
                        outcome=DebateOutcome.UNSAFE if unsafe else DebateOutcome.SAFE,
                        final_label=final_label,
                        confirmed_categories=verdict["categories"] if unsafe else [],
                        rounds_configured=self._max_rounds,
                        rounds_run=round_num,
                        attacker_rounds=attacker_rounds,
                        defender_rounds=defender_rounds,
                        token_usage=tracker.summary(),
                        debate_ended_early=round_num < self._max_rounds,
                    )

                if round_num < self._max_rounds:
                    previous_feedback.append(
                        self._build_previous_verdict_summary(
                            round_num=round_num,
                            accusation=accusation,
                            verdict=verdict,
                        )
                    )
                    continue

            return PMPDDebateResult(
                outcome=DebateOutcome.SAFE,
                final_label=self._output_mapper.default_label,
                confirmed_categories=[],
                rounds_configured=self._max_rounds,
                rounds_run=self._max_rounds,
                attacker_rounds=attacker_rounds,
                defender_rounds=defender_rounds,
                token_usage=tracker.summary(),
                debate_ended_early=False,
            )

        except APIError as exc:
            return PMPDDebateResult(
                outcome=DebateOutcome.ERROR,
                final_label=self._output_mapper.error_label,
                confirmed_categories=[],
                rounds_configured=self._max_rounds,
                rounds_run=max(len(attacker_rounds), len(defender_rounds)),
                attacker_rounds=attacker_rounds,
                defender_rounds=defender_rounds,
                token_usage=tracker.summary(),
                api_error=True,
                error_message=str(exc),
            )

    def _run_attacker(
        self,
        prompt: AgentPrompt,
        formatted_input: str,
        previous_feedback: list[dict[str, Any]],
        tracker: TokenTracker,
    ) -> tuple[ParsedAccusation, dict[str, Any]]:
        user_content = formatted_input
        if previous_feedback:
            feedback_text = self._format_previous_verdicts_for_attacker(previous_feedback)
            user_content = (
                f"{formatted_input}\n\n"
                "PREVIOUS SPECIALIST FEEDBACK (previous category selections were insufficient or unsupported):\n"
                f"{feedback_text}\n\n"
                "Select a materially different category set from the available roster only.\n"
                "Do not repeat the exact same rejected set.\n"
                "On this repair round only, you may return 1-2 alternative categories.\n"
                "Prefer the smallest corrected set that is clearly supported by the INPUT."
            )

        return self._call_selector_with_repair(
            prompt=prompt,
            user_content=user_content,
            max_categories=2 if previous_feedback else 1,
            tracker=tracker,
        )

    def _run_defender(
        self,
        prompt: AgentPrompt,
        formatted_input: str,
        accusation: ParsedAccusation,
        tracker: TokenTracker,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        selected_categories = ", ".join(accusation.categories) if accusation.categories else "(none)"
        user_content = (
            f"{formatted_input}\n\n"
            "SELECTOR OUTPUT (scope only; not proof):\n"
            f"Selected categories: {selected_categories}\n"
            "Evaluate the INPUT independently using only the provided category scope."
        )

        return self._call_specialist_with_repair(
            prompt=prompt,
            user_content=user_content,
            accusation=accusation,
            tracker=tracker,
        )

    def _call_selector_with_repair(
        self,
        prompt: AgentPrompt,
        user_content: str,
        max_categories: int,
        tracker: TokenTracker,
    ) -> tuple[ParsedAccusation, dict[str, Any]]:
        start_index = len(tracker.raw_calls("attacker"))
        initial_result = self._invoke_model(
            role="attacker",
            prompt=user_content,
            agent_prompt=prompt,
            model=self._attacker_model,
            tracker=tracker,
        )
        initial_raw = initial_result.get("content", "")
        selected_raw = initial_raw
        accusation = self._attacker_parser.parse(initial_raw)
        repair_attempted = False
        repair_successful = False
        repair_raw = ""

        if not accusation.parse_success:
            repair_attempted = True
            repair_result = self._invoke_model(
                role="attacker",
                prompt=self._build_parse_repair_prompt("selector", initial_raw),
                agent_prompt=prompt,
                model=self._attacker_model,
                tracker=tracker,
            )
            repair_raw = repair_result.get("content", "")
            repaired = self._attacker_parser.parse(repair_raw)
            repair_successful = repaired.parse_success
            if self._selector_quality(repaired) >= self._selector_quality(accusation):
                accusation = repaired
                selected_raw = repair_raw

        accusation = self._truncate_selector_categories(accusation, max_categories=max_categories)

        calls = tracker.raw_calls("attacker")[start_index:]
        return accusation, {
            "initial_raw_output": initial_raw,
            "selected_raw_output": selected_raw,
            "repair_raw_output": repair_raw,
            "repair_attempted": repair_attempted,
            "repair_successful": repair_successful,
            "telemetry": self._aggregate_call_telemetry(calls),
        }

    def _call_specialist_with_repair(
        self,
        prompt: AgentPrompt,
        user_content: str,
        accusation: ParsedAccusation,
        tracker: TokenTracker,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        start_index = len(tracker.raw_calls("defender"))
        initial_result = self._invoke_model(
            role="defender",
            prompt=user_content,
            agent_prompt=prompt,
            model=self._defender_model,
            tracker=tracker,
        )
        initial_raw = initial_result.get("content", "")
        selected_raw = initial_raw
        verdict = self._defender_parser.parse(
            initial_raw,
            valid_categories=accusation.categories,
            fallback_label=self._output_mapper.default_label,
        )
        repair_attempted = False
        repair_successful = False
        repair_raw = ""

        if not verdict["parse_success"]:
            repair_attempted = True
            repair_result = self._invoke_model(
                role="defender",
                prompt=self._build_parse_repair_prompt("specialist", initial_raw),
                agent_prompt=prompt,
                model=self._defender_model,
                tracker=tracker,
            )
            repair_raw = repair_result.get("content", "")
            repaired = self._defender_parser.parse(
                repair_raw,
                valid_categories=accusation.categories,
                fallback_label=self._output_mapper.default_label,
            )
            repair_successful = repaired["parse_success"]
            if self._specialist_quality(repaired) >= self._specialist_quality(verdict):
                verdict = repaired
                selected_raw = repair_raw

        calls = tracker.raw_calls("defender")[start_index:]
        return verdict, {
            "initial_raw_output": initial_raw,
            "selected_raw_output": selected_raw,
            "repair_raw_output": repair_raw,
            "repair_attempted": repair_attempted,
            "repair_successful": repair_successful,
            "telemetry": self._aggregate_call_telemetry(calls),
        }

    def _invoke_model(
        self,
        role: str,
        prompt: str,
        agent_prompt: AgentPrompt,
        model: str,
        tracker: TokenTracker,
    ) -> dict[str, Any]:
        result = self._retry.call_raw(
            prompt=prompt,
            system_msg=agent_prompt.system_msg,
            developer_msg=agent_prompt.developer_msg,
            model=model,
        )
        tracker.record(role, result)
        return result

    @staticmethod
    def _aggregate_call_telemetry(calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "api_calls": len(calls),
            "input_tokens": sum((call.get("input_tokens") or 0) for call in calls) or None,
            "output_tokens": sum((call.get("output_tokens") or 0) for call in calls) or None,
            "total_tokens": sum((call.get("total_tokens") or 0) for call in calls) or None,
            "response_time_s": round(
                sum((call.get("response_time") or 0.0) for call in calls),
                3,
            )
            if calls
            else None,
        }

    @staticmethod
    def _selector_quality(accusation: ParsedAccusation) -> tuple[int, int, int, int]:
        return (
            int(accusation.parse_success),
            int(bool(accusation.categories)),
            int(bool(accusation.evidence)),
            -len(accusation.categories),
        )

    @staticmethod
    def _specialist_quality(verdict: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(bool(verdict.get("parse_success"))),
            int(verdict.get("category_fit") == "SUPPORTED"),
            len(verdict.get("categories") or []),
            int(verdict.get("final_label") is not None),
        )

    @staticmethod
    def _build_parse_repair_prompt(agent_name: str, raw_output: str) -> str:
        if agent_name == "selector":
            schema = (
                "CANDIDATE_CATEGORIES: <comma-separated top-level category IDs, at most 2 IDs total, or empty>\n"
                "EVIDENCE_SPANS: <semicolon-separated short quotes or concrete behaviors, or empty>"
            )
        else:
            schema = (
                "CATEGORY_FIT: SUPPORTED | UNSUPPORTED\n"
                "FINAL_LABEL: <allowed label>\n"
                "CATEGORIES: <comma-separated provided top-level category IDs only, or empty>\n"
                "BASIS_CODE: <MATCHED_POLICY | EXCEPTION_APPLIES | INSUFFICIENT_COVERAGE | INSUFFICIENT_EVIDENCE | NO_VIOLATION | OTHER>\n"
                "CATEGORY_NOTES: <short note or empty>\n"
                "REASONING: <short evidence-based reasoning>"
            )
        return (
            f"The previous {agent_name} response did not follow the required structured block.\n"
            "Reformat the same answer into the exact required structured block only.\n"
            "Do not add new analysis. Do not invent categories or labels.\n"
            "Reuse only information already present in the previous response. Leave fields empty rather than guessing.\n"
            f"Required block:\n{schema}\n\n"
            f"PREVIOUS RESPONSE:\n{raw_output}"
        )

    @staticmethod
    def _truncate_selector_categories(
        accusation: ParsedAccusation,
        max_categories: int,
    ) -> ParsedAccusation:
        if len(accusation.categories) <= max_categories:
            return accusation

        accusation.categories = accusation.categories[:max_categories]
        accusation.accusation = AttackerOutputParser._build_selection_summary(
            categories=accusation.categories,
            evidence=accusation.evidence,
            fallback=accusation.raw_output,
        )
        return accusation

    @staticmethod
    def _same_category_set(left: list[str], right: list[str]) -> bool:
        return set(left) == set(right)

    @staticmethod
    def _build_previous_verdict_summary(
        round_num: int,
        accusation: ParsedAccusation,
        verdict: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "round": str(round_num),
            "candidate_categories": ", ".join(accusation.categories) if accusation.categories else "(none)",
            "candidate_categories_list": accusation.categories,
            "evidence_spans": accusation.evidence or "(none)",
            "category_fit": verdict.get("category_fit") or "UNSUPPORTED",
            "basis_code": verdict.get("basis_code") or "UNKNOWN",
            "overturn_basis": verdict.get("basis_code") or "UNKNOWN",
            "category_notes": verdict.get("category_notes") or verdict.get("reasoning", "")[:200],
        }

    @staticmethod
    def _build_missing_selection_feedback(
        round_num: int,
        accusation: ParsedAccusation,
    ) -> dict[str, Any]:
        return {
            "round": str(round_num),
            "candidate_categories": "(none)",
            "candidate_categories_list": [],
            "evidence_spans": accusation.evidence or "(none)",
            "category_fit": "UNSUPPORTED",
            "basis_code": "INSUFFICIENT_COVERAGE",
            "overturn_basis": "INSUFFICIENT_COVERAGE",
            "category_notes": "No valid category selection was available for specialist review. Propose a different or smaller corrected set from the available category roster only.",
        }

    @staticmethod
    def _format_previous_verdicts_for_attacker(
        previous_verdicts: list[dict[str, Any]],
    ) -> str:
        blocks: list[str] = []
        for summary in previous_verdicts:
            blocks.append(
                "\n".join(
                    [
                        f"[Round {summary.get('round', '?')}]",
                        f"Candidate categories: {summary.get('candidate_categories', '(none)')}",
                        f"Evidence spans: {summary.get('evidence_spans', '(none)')}",
                        f"Category fit: {summary.get('category_fit', 'UNSUPPORTED')}",
                        f"Basis code: {summary.get('basis_code', 'UNKNOWN')}",
                        f"Category notes: {summary.get('category_notes', '(none)')}",
                    ]
                )
            )
        return "\n\n".join(blocks)
