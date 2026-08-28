"""
Evaluation Runner

Mode-agnostic orchestrator that loops over dataset records,
calls the appropriate evaluator, and appends results after each index.
"""

from __future__ import annotations

import traceback
from typing import Protocol, runtime_checkable

from data.dataset_loader import EvaluationRecord
from evaluation.result_writer import EvaluationResult, ResultWriter
from infrastructure.api_client import APIError


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Common interface for RAGEvaluator and PMPDEvaluator."""

    def evaluate(self, record: EvaluationRecord) -> EvaluationResult:
        ...


class EvaluationRunner:
    """Evaluate records one by one and append each result immediately."""

    def __init__(
        self,
        evaluator: EvaluatorProtocol,
        writer: ResultWriter,
    ) -> None:
        self._evaluator = evaluator
        self._writer = writer

    def run(self, records: list[EvaluationRecord]) -> list[EvaluationResult]:
        total = len(records)
        results: list[EvaluationResult] = []

        print(f"\n{'=' * 60}")
        print(f"EvaluationRunner: {total} records")
        print(f"Output: {self._writer.output_path}")
        print(f"{'=' * 60}")

        for i, record in enumerate(records, start=1):
            print(f"\n[{i}/{total}] Index {record.index}")

            try:
                result = self._evaluator.evaluate(record)
            except APIError as exc:
                print(f"  APIError on index {record.index}: {exc}")
                print("  Recording as api_error=True and continuing...")
                result = self._make_error_result(record, str(exc))
            except Exception:
                print(f"\nUnexpected error on index {record.index}:")
                traceback.print_exc()
                raise

            self._writer.append(result)
            results.append(result)

            status = "OK" if not result.api_error else "ERR"
            print(f"  {status} [{i}/{total}] Index {record.index} -> {result.courtguard_verdict}")

        print(f"\n{'=' * 60}")
        print("EvaluationRunner: Complete")
        print(f"  Records evaluated : {len(results)}")
        print(f"  Output file       : {self._writer.output_path}")
        self._print_summary(results)
        print(f"{'=' * 60}\n")

        return results

    @staticmethod
    def _make_error_result(
        record: EvaluationRecord,
        error_message: str,
    ) -> EvaluationResult:
        return EvaluationResult(
            index=record.index,
            user_prompt=record.user_prompt,
            target_model_response=record.target_model_response,
            category=record.category,
            label=record.label,
            is_safe=record.is_safe,
            courtguard_verdict="ERROR",
            courtguard_winner="",
            courtguard_violated_categories=[],
            courtguard_reasoning=f"API error: {error_message}",
            courtguard_severity_raw="ERROR",
            api_error=True,
        )

    @staticmethod
    def _print_summary(results: list[EvaluationResult]) -> None:
        from collections import Counter

        counts = Counter(r.courtguard_verdict for r in results)
        print("\n  Verdict distribution:")
        for verdict, count in sorted(counts.items()):
            pct = count / len(results) * 100 if results else 0
            print(f"    {verdict:12s}: {count:4d} ({pct:.1f}%)")
