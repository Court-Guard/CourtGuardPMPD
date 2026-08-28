"""
PMPD Mode Evaluator

Evaluates a single dataset record using the PMPD pipeline.

PMPD mode operation
-------------------
  For each record:
    1. PMPDAssembler.build(user_prompt, response) selects the most
       relevant CategoryModules via RAG or heuristic scoring,
       renders them with rules + exceptions + One-Shots into a
       structured context block, and builds role-specific system
       messages grounded in those exact categories.
    2. PolicyDebate.set_prompts() injects the assembled system messages.
    3. PolicyDebate.run_debate() runs the full Attacker / Defender / Judge
       cycle with the assembled context.

The key difference from RAG mode: both the context AND the system
prompts are policy-grounded and category-specific per evaluation.
"""

from __future__ import annotations

from data.dataset_loader import EvaluationRecord
from debate.pmpd_debate import PMPDDebate, PMPDDebateResult
from evaluation.input_mapper import InputMapper
from evaluation.output_mapper import OutputMapper
from evaluation.result_writer import EvaluationResult
from pmpd_pipeline.assembler import PMPDAssembler


class PMPDEvaluator:
    """
    Evaluates dataset records in PMPD mode.

    Usage
    -----
        evaluator = PMPDEvaluator(assembler, debate, model="openai/gpt-oss-20b")
        result    = evaluator.evaluate(record)
    """

    def __init__(
        self,
        assembler: PMPDAssembler,
        debate: PMPDDebate,
        model: str = "",
        k: int = 5,
        inspect: bool = False,
        input_mapper: InputMapper | None = None,
        output_mapper: OutputMapper | None = None,
    ) -> None:
        """
        Args:
            assembler:    PMPDAssembler with PMPD database and optional RAG.
            debate:       PMPDDebate instance.
            model:        Model identifier string (for result metadata).
            k:            Number of categories to retrieve per query.
            inspect:      Whether to print PMPD inspection output.
            input_mapper: Optional InputMapper for mapping input fields.
        """
        self._inspect = inspect
        self._assembler = assembler
        self._debate = debate
        self._model = model
        self._k = k
        self._input_mapper = input_mapper or InputMapper(InputMapper.DEFAULT_FIELDS)
        self._output_mapper = output_mapper or OutputMapper(OutputMapper.DEFAULT_LABELS)

    def evaluate(self, record: EvaluationRecord) -> EvaluationResult:
        print(f"\n  [PMPD] Evaluating index {record.index}...")

        resolved = self._input_mapper.map(record.raw)
        print(f"  Query    : {resolved.formatted[:80]}...")

        payload = None
        if getattr(self._debate, "requires_assembled_payload", True):
            payload = self._assembler.build(
                query=resolved.get("user_prompt", resolved.formatted),
                response=resolved.get("target_model_response", ""),
                k=self._k,
            )

        if self._inspect:
            self._print_inspection(record, payload)

        result: PMPDDebateResult = self._debate.run(resolved.formatted, payload=payload)
        if self._inspect:
            self._print_post_run_inspection(record, result)

        print(
            f"  Outcome  : {result.outcome} | "
            f"Rounds: {result.rounds_run}/{result.rounds_configured}"
        )

        attacker_cats = result.attacker_rounds[-1]["categories"] if result.attacker_rounds else []
        preliminary_label = (
            result.attacker_rounds[-1].get("preliminary_label", "")
            if result.attacker_rounds else ""
        )
        runtime_strategy = getattr(result, "runtime_strategy", "debate")
        if runtime_strategy == "multivote":
            winner = "MultiVote"
            reasoning = (
                result.aggregation.get("representative_reasoning", "")
                if getattr(result, "aggregation", None)
                else ""
            )
        else:
            winner = "Specialist" if result.defender_rounds else "Selector"
            reasoning = result.defender_rounds[-1]["reasoning"] if result.defender_rounds else ""

        return EvaluationResult(
            index=record.index,
            user_prompt=resolved.get("user_prompt", resolved.formatted),
            target_model_response=resolved.get("target_model_response", resolved.formatted),
            category=record.category,
            label=record.label,
            is_safe=record.is_safe,
            courtguard_mode="pmpd",
            courtguard_model=self._model,
            courtguard_verdict=result.final_label,
            courtguard_winner=winner,
            courtguard_violated_categories=result.confirmed_categories,
            courtguard_reasoning=reasoning,
            courtguard_severity_raw=result.final_label,
            debate_history=[
                f"ATTACKER Round {r['round']}: {r['raw_output'][:200]}"
                for r in result.attacker_rounds
            ]
            + [
                f"DEFENDER Round {r['round']}: {r['raw_output'][:200]}"
                for r in result.defender_rounds
            ]
            + [
                f"JUDGE {r['judge_index']}: {r['raw_output'][:200]}"
                for r in getattr(result, "judge_rounds", [])
            ],
            timing_info=[],
            api_error=result.api_error,
            outcome=result.outcome,
            final_label=result.final_label,
            preliminary_label=preliminary_label,
            attacker_categories=attacker_cats,
            confirmed_categories=result.confirmed_categories,
            rounds_configured=result.rounds_configured,
            rounds_run=result.rounds_run,
            debate_ended_early=result.debate_ended_early,
            token_usage=result.token_usage,
            attacker_raw_rounds=result.attacker_rounds,
            defender_raw_rounds=result.defender_rounds,
            judge_raw_rounds=getattr(result, "judge_rounds", []),
            runtime_strategy=runtime_strategy,
            retrieved_categories=getattr(result, "retrieved_categories", []),
            multivote_summary=getattr(result, "aggregation", {}),
            output_labels_used=list(self._output_mapper.labels),
            default_output_label=self._output_mapper.default_label,
            error_output_label=self._output_mapper.error_label,
        )

    def _print_inspection(
        self,
        record: EvaluationRecord,
        payload=None,
    ) -> None:
        """Print PMPD-built prompts and module contents for inspection."""
        sep = "=" * 70
        is_multivote = not getattr(self._debate, "requires_assembled_payload", True)

        print(f"\n{sep}")
        print(f"PMPD INSPECTION - Index {record.index}")
        print(sep)

        db = self._debate._db
        print("\nPMPD DATABASE SUMMARY")
        print(f"  Categories loaded: {db.list_categories()}")
        gm = db.global_module
        if gm:
            print(f"  Global objective : {gm.objective}")
            print(f"  Principles       : {len(gm.principles)} loaded")
            print(f"  Definitions      : {len(gm.definitions)} terms")
            print(f"  Eval principles  : {len(gm.general_eval_principles)} rules")
        else:
            print("  Warning: No GlobalModule loaded")

        print("\nALL CATEGORY MODULES (lite - what Attacker sees):")
        for cid in db.list_categories():
            cat = db.get_category(cid)
            if cat:
                shadow = len(cat.get_shadow_shots())
                live = len(cat.get_live_shots(limit=999))
                subs = len(cat.sub_categories)
                print(f"\n  [{cid}] {cat.name}")
                print(f"    Short def : {cat.short_definition or '(not set)'}")
                print(f"    Sub-cats  : {subs}")
                print(f"    Exceptions: {len(cat.exceptions)}")
                print(f"    One-Shots : {shadow} shadow, {live} live")

        builder = self._debate._prompt_builder
        if is_multivote:
            judge_count = getattr(self._debate, "_judge_count", 3)
            focus_notes = getattr(
                self._debate,
                "_FOCUS_NOTES",
                ("Prioritize direct harmful compliance and plain reading of the response.",),
            )
            gate_prompt = builder.build_multivote_category_gate()
            label_judge_prompt = builder.build_multivote_label_judge(
                judge_index=1,
                judge_count=judge_count,
                focus_note=focus_notes[0],
            )

            print("\nMULTIVOTE CATEGORY GATE SYSTEM PROMPT")
            print(gate_prompt.system_msg)

            if gate_prompt.developer_msg:
                print("\nMULTIVOTE CATEGORY GATE DEVELOPER MESSAGE")
                print(gate_prompt.developer_msg)

            print("\nMULTIVOTE LABEL JUDGE SYSTEM PROMPT")
            print(label_judge_prompt.system_msg)

            if label_judge_prompt.developer_msg:
                print("\nMULTIVOTE LABEL JUDGE DEVELOPER MESSAGE")
                print(label_judge_prompt.developer_msg)
        else:
            attacker_prompt = builder.build_attacker()

            print("\nATTACKER SYSTEM PROMPT (HarmonyPromptBuilder)")
            print(attacker_prompt.system_msg)

            if attacker_prompt.developer_msg:
                print("\nATTACKER DEVELOPER MESSAGE (harmony roles enabled)")
                print(attacker_prompt.developer_msg)

        if payload is not None:
            print("\nPREVIEWED CATEGORY SELECTION")
            print(f"  Assembler selected: {payload.category_ids}")
            print(f"  Context size      : {payload.total_chars} chars")

        if is_multivote:
            print("\nMULTIVOTE RUNTIME")
            print("  A category gate runs first and may return NONE.")
            print("  If categories are returned, independent label judges evaluate the INPUT.")
            print("  The gate itself only routes category relevance and does not judge safety.")
        else:
            print("\nSPECIALIST PROMPT")
            print("  Built dynamically after the selector nominates categories.")
            print("  Post-run inspection below will print the actual defender prompt when invoked.")

        print(f"\n{sep}\n")

    def _print_post_run_inspection(
        self,
        record: EvaluationRecord,
        result: PMPDDebateResult,
    ) -> None:
        """Print the actual defender prompt(s) used after the debate finishes."""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"PMPD POST-RUN INSPECTION - Index {record.index}")
        print(sep)

        if getattr(result, "runtime_strategy", "") == "multivote":
            gate_categories = result.attacker_rounds[-1].get("categories", []) if result.attacker_rounds else []
            print("\nMULTIVOTE POST-RUN SUMMARY")
            print(f"  Gate categories : {gate_categories or '(none)'}")
            print(f"  Final label     : {result.final_label}")
            print(f"  Final categories: {result.confirmed_categories or '(none)'}")
            if getattr(result, "aggregation", None):
                print(f"  Mean support    : {result.aggregation.get('mean_label_support', {})}")
            print(f"\n{sep}\n")
            return

        if not result.attacker_rounds:
            print("\nNo attacker rounds recorded.")
            print(f"\n{sep}\n")
            return

        if not any(round_data.get("categories") for round_data in result.attacker_rounds):
            print(
                "\nSpecialist was not invoked for this record because the selector returned no valid categories."
            )
            print(f"\n{sep}\n")
            return

        builder = self._debate._prompt_builder
        previous_verdicts: list[dict[str, str]] = []

        for idx, attacker_round in enumerate(result.attacker_rounds, start=1):
            flagged = attacker_round.get("categories", [])
            if not flagged:
                print(f"\nRound {idx}: attacker returned no valid categories; no defender prompt built.")
                continue

            defender_prompt = builder.build_defender(
                flagged_categories=flagged,
                previous_verdicts=previous_verdicts,
            )

            print(f"\nSPECIALIST SYSTEM PROMPT - Round {idx}")
            print(f"  Flagged categories: {flagged}")
            print(defender_prompt.system_msg)

            if defender_prompt.developer_msg:
                print(f"\nSPECIALIST DEVELOPER MESSAGE - Round {idx}")
                print(defender_prompt.developer_msg)

            if idx <= len(result.defender_rounds):
                defender_round = result.defender_rounds[idx - 1]
                previous_verdicts.append(
                    {
                        "round": str(idx),
                        "candidate_categories": ", ".join(flagged) if flagged else "(none)",
                        "evidence_spans": attacker_round.get("evidence", "") or "(none)",
                        "category_fit": defender_round.get("category_fit", "") or "UNSUPPORTED",
                        "basis_code": defender_round.get("basis_code", "") or "UNKNOWN",
                        "category_notes": defender_round.get("category_notes", "")
                        or defender_round.get("reasoning", "")[:200],
                    }
                )

        print(f"\n{sep}\n")
