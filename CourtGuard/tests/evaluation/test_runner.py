"""
Tests for evaluation/runner.py

All evaluator calls are mocked — no real API interaction.

Covers:
  - Happy path: all records evaluated and written
  - APIError per record: marked api_error=True, run continues
  - Unexpected exception: re-raised immediately
  - _print_summary verdict distribution
  - EvaluatorProtocol structural check
  - Results written after each record (not batched)
"""

import json
from unittest.mock import MagicMock

import pytest

from data.dataset_loader import EvaluationRecord
from evaluation.result_writer import EvaluationResult, ResultWriter
from evaluation.runner import EvaluationRunner, EvaluatorProtocol
from infrastructure.api_client import APIError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(index: int) -> EvaluationRecord:
    return EvaluationRecord(
        index=index,
        user_prompt=f"Query {index}",
        target_model_response=f"Response {index}",
        category="test",
        label="unharmful",
        is_safe=True,
    )


def _make_result(index: int, verdict: str = "SAFE") -> EvaluationResult:
    return EvaluationResult(
        index=index,
        user_prompt=f"Query {index}",
        target_model_response=f"Response {index}",
        courtguard_verdict=verdict,
        courtguard_mode="rag",
        courtguard_model="test-model",
        evaluated_at="2026-03-20T00:00:00",
    )


def _make_mock_evaluator(results: list[EvaluationResult]) -> MagicMock:
    evaluator = MagicMock(spec=EvaluatorProtocol)
    evaluator.evaluate.side_effect = results
    return evaluator


def _make_mock_writer(tmp_path) -> ResultWriter:
    return ResultWriter(
        mode="rag",
        model="test-model",
        dataset_name="test_dataset",
        output_dir=str(tmp_path),
        timestamp="20260101_000000",
    )


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------


class TestEvaluatorProtocol:
    def test_mock_satisfies_protocol(self):
        mock = MagicMock()
        mock.evaluate = MagicMock(return_value=_make_result(0))
        assert isinstance(mock, EvaluatorProtocol)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestEvaluationRunnerHappyPath:
    def test_all_records_evaluated(self, tmp_path):
        records = [_make_record(i) for i in range(3)]
        results = [_make_result(i) for i in range(3)]
        evaluator = _make_mock_evaluator(results)
        writer = _make_mock_writer(tmp_path)
        runner = EvaluationRunner(evaluator, writer)

        returned = runner.run(records)

        assert len(returned) == 3
        assert evaluator.evaluate.call_count == 3

    def test_results_written_after_each_record(self, tmp_path):
        """Each append should happen immediately, not batched."""
        records = [_make_record(i) for i in range(3)]
        results = [_make_result(i) for i in range(3)]
        evaluator = _make_mock_evaluator(results)
        writer = _make_mock_writer(tmp_path)

        append_calls = []
        original_append = writer.append

        def tracking_append(result):
            original_append(result)
            append_calls.append(result.index)

        writer.append = tracking_append
        runner = EvaluationRunner(evaluator, writer)
        runner.run(records)

        assert append_calls == [0, 1, 2]

    def test_output_file_contains_all_results(self, tmp_path):
        records = [_make_record(i) for i in range(3)]
        results = [_make_result(i, "SAFE") for i in range(3)]
        evaluator = _make_mock_evaluator(results)
        writer = _make_mock_writer(tmp_path)
        runner = EvaluationRunner(evaluator, writer)
        runner.run(records)

        with open(writer.output_path) as f:
            data = json.load(f)
        assert len(data) == 3


# ---------------------------------------------------------------------------
# APIError handling
# ---------------------------------------------------------------------------


class TestAPIErrorHandling:
    def test_api_error_does_not_stop_run(self, tmp_path):
        records = [_make_record(i) for i in range(3)]

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = [
            _make_result(0),
            APIError("rate limit hit"),
            _make_result(2),
        ]

        writer = _make_mock_writer(tmp_path)
        runner = EvaluationRunner(evaluator, writer)
        results = runner.run(records)

        assert len(results) == 3
        assert results[1].api_error is True
        assert results[1].courtguard_verdict == "ERROR"

    def test_api_error_record_still_written(self, tmp_path):
        records = [_make_record(0)]
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = APIError("server error")
        writer = _make_mock_writer(tmp_path)
        runner = EvaluationRunner(evaluator, writer)
        runner.run(records)

        with open(writer.output_path) as f:
            data = json.load(f)
        assert data[0]["api_error"] is True

    def test_other_exception_is_reraised(self, tmp_path):
        records = [_make_record(0)]
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = RuntimeError("unexpected crash")
        writer = _make_mock_writer(tmp_path)
        runner = EvaluationRunner(evaluator, writer)

        with pytest.raises(RuntimeError, match="unexpected crash"):
            runner.run(records)


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_summary_with_mixed_verdicts(self, capsys):
        results = [
            _make_result(0, "SAFE"),
            _make_result(1, "UNSAFE"),
            _make_result(2, "SAFE"),
            _make_result(3, "BORDERLINE"),
        ]
        EvaluationRunner._print_summary(results)
        captured = capsys.readouterr().out
        assert "SAFE" in captured
        assert "UNSAFE" in captured
        assert "BORDERLINE" in captured

    def test_summary_empty_results(self, capsys):
        EvaluationRunner._print_summary([])
        captured = capsys.readouterr().out
        # Should not crash on empty list
        assert "distribution" in captured.lower() or captured == ""
