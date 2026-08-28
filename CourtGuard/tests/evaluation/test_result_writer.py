"""
Tests for evaluation/result_writer.py

Covers:
  - Output filename format: CourtGuard_{mode}_{model}_{dataset}_{ts}.json
  - _slugify strips provider prefix
  - append() creates file on first call
  - append() accumulates records across multiple calls
  - Corrupt existing file handled gracefully (starts fresh)
  - record_count() reflects written records
  - EvaluationResult fields serialised correctly
"""

import json
import os

import pytest

from evaluation.result_writer import EvaluationResult, ResultWriter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def writer(tmp_path) -> ResultWriter:
    return ResultWriter(
        mode="pmpd",
        model="openai/gpt-oss-20b",
        dataset_name="PKU_SafeRLHF_180",
        output_dir=str(tmp_path),
        timestamp="20260320_143022",
    )


def _make_result(index: int, verdict: str = "SAFE") -> EvaluationResult:
    return EvaluationResult(
        index=index,
        user_prompt=f"Query {index}",
        target_model_response=f"Response {index}",
        category="test_category",
        label="unharmful",
        is_safe=True,
        courtguard_mode="pmpd",
        courtguard_model="openai/gpt-oss-20b",
        courtguard_verdict=verdict,
        courtguard_winner="Defender",
        courtguard_violated_categories=[],
        courtguard_reasoning="No violation found.",
        courtguard_severity_raw=verdict,
        debate_history=["ATTACKER Round 1: arg", "DEFENDER Round 1: arg"],
        timing_info=["Round 1: 1.2s"],
        evaluated_at="2026-03-20T14:30:22",
        api_error=False,
    )


# ---------------------------------------------------------------------------
# Filename format tests
# ---------------------------------------------------------------------------


class TestFilenameFormat:
    def test_filename_contains_all_components(self, writer):
        name = os.path.basename(writer.output_path)
        assert name.startswith("CourtGuard_pmpd_")
        assert "gpt-oss-20b" in name
        assert "PKU_SafeRLHF_180" in name
        assert "20260320_143022" in name
        assert name.endswith(".json")

    def test_rag_mode_in_filename(self, tmp_path):
        w = ResultWriter(
            mode="rag",
            model="openai/gpt-oss-20b",
            dataset_name="test_data",
            output_dir=str(tmp_path),
            timestamp="20260101_000000",
        )
        assert "rag" in os.path.basename(w.output_path)

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "nested" / "results")
        w = ResultWriter(
            mode="rag",
            model="m",
            dataset_name="d",
            output_dir=new_dir,
            timestamp="20260101_000000",
        )
        assert os.path.exists(new_dir)


# ---------------------------------------------------------------------------
# Slugify tests
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_strips_provider_prefix(self):
        assert ResultWriter._slugify("openai/gpt-oss-20b") == "gpt-oss-20b"

    def test_strips_meta_prefix(self):
        result = ResultWriter._slugify("meta-llama/llama-3.3-70b-instruct")
        assert "/" not in result
        assert result.startswith("llama")

    def test_strips_provider_prefix_with_free_suffix(self):
        result = ResultWriter._slugify("meta-llama/llama-3.3-70b-instruct:free")
        assert "/" not in result
        assert ":" not in result
        assert result.startswith("llama")

    def test_no_prefix(self):
        assert ResultWriter._slugify("mymodel") == "mymodel"

    def test_special_chars_replaced(self):
        slug = ResultWriter._slugify("model:v1.0")
        assert ":" not in slug


# ---------------------------------------------------------------------------
# Append tests
# ---------------------------------------------------------------------------


class TestAppend:
    def test_creates_file_on_first_append(self, writer):
        assert not os.path.exists(writer.output_path)
        writer.append(_make_result(0))
        assert os.path.exists(writer.output_path)

    def test_file_contains_valid_json_array(self, writer):
        writer.append(_make_result(0))
        with open(writer.output_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_multiple_appends_accumulate(self, writer):
        writer.append(_make_result(0))
        writer.append(_make_result(1))
        writer.append(_make_result(2))
        with open(writer.output_path) as f:
            data = json.load(f)
        assert len(data) == 3
        assert data[0]["index"] == 0
        assert data[2]["index"] == 2

    def test_record_count_reflects_appended(self, writer):
        assert writer.record_count() == 0
        writer.append(_make_result(0))
        assert writer.record_count() == 1
        writer.append(_make_result(1))
        assert writer.record_count() == 2

    def test_all_result_fields_serialised(self, writer):
        writer.append(_make_result(0, verdict="UNSAFE"))
        with open(writer.output_path) as f:
            record = json.load(f)[0]

        assert record["index"] == 0
        assert record["courtguard_verdict"] == "UNSAFE"
        assert record["courtguard_mode"] == "pmpd"
        assert record["courtguard_violated_categories"] == []
        assert isinstance(record["debate_history"], list)
        assert record["api_error"] is False

    def test_corrupt_existing_file_starts_fresh(self, writer, tmp_path):
        # Write garbage to the output file before appending
        with open(writer.output_path, "w") as f:
            f.write("not valid json {{{")
        # Should not crash — starts with empty list
        writer.append(_make_result(0))
        with open(writer.output_path) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_appends_across_separate_writer_instances(self, tmp_path):
        """Two writer instances pointing at the same file both append correctly."""
        w1 = ResultWriter(
            mode="rag",
            model="m",
            dataset_name="d",
            output_dir=str(tmp_path),
            timestamp="20260101_000000",
        )
        w2 = ResultWriter(
            mode="rag",
            model="m",
            dataset_name="d",
            output_dir=str(tmp_path),
            timestamp="20260101_000000",
        )
        w1.append(_make_result(0))
        w2.append(_make_result(1))

        with open(w1.output_path) as f:
            data = json.load(f)
        assert len(data) == 2
