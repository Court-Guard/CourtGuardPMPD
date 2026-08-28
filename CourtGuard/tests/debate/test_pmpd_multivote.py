"""Tests for the category-gated PMPD multivote runtime."""

from unittest.mock import MagicMock

from core.enums import DebateOutcome
from core.models import CategoryModule, GlobalModule
from core.pmpd import PMPD
from debate.pmpd_multivote import PMPDMultiVote
from evaluation.output_mapper import OutputMapper
from infrastructure.api_client import APIClient


def _make_pmpd() -> PMPD:
    db = PMPD.from_scratch()
    db.set_global_module(
        GlobalModule(
            objective="Test objective.",
            principles=["Evaluate only the designated artifact."],
            schema="Output label.",
            definitions={"Hazard": "Harmful enabling content."},
            general_eval_principles=["Assume benign intent in ambiguous cases."],
            evaluation_protocol=["Use companion fields only as context unless policy says otherwise."],
        )
    )
    db.add_category(
        CategoryModule(
            category_id="S9",
            name="Violent Crimes",
            general_rules="No violence.",
            exceptions=["Historical educational content is allowed."],
            one_shots=[],
            citations=[],
            short_definition="Violent criminal activity.",
        )
    )
    db.add_category(
        CategoryModule(
            category_id="S2",
            name="Defamation",
            general_rules="No false harmful claims about real people.",
            exceptions=[],
            one_shots=[],
            citations=[],
            short_definition="Defamation and false allegations.",
        )
    )
    db.add_category(
        CategoryModule(
            category_id="S5",
            name="Privacy",
            general_rules="No exposure of sensitive private information.",
            exceptions=[],
            one_shots=[],
            citations=[],
            short_definition="Privacy and personal data exposure.",
        )
    )
    return db


def _make_mock_client() -> MagicMock:
    client = MagicMock(spec=APIClient)
    client.api_key = "sk-test"
    client.get_usage_stats.return_value = {"total_requests": 1}
    return client


def _api_result(content: str, input_tokens: int = 100, output_tokens: int = 50) -> dict:
    return {
        "content": content,
        "success": True,
        "response_time": 0.5,
        "model_used": "test-model",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_used": input_tokens + output_tokens,
    }


GATE_S9 = """
CANDIDATE_CATEGORIES: S9
"""

GATE_THREE = """
CANDIDATE_CATEGORIES: S9, S2, S5
"""

GATE_NONE = """
CANDIDATE_CATEGORIES: NONE
"""

GATE_BAD = """
Maybe violent.
"""

JUDGE_UNSAFE_S9 = """
FINAL_LABEL: UNSAFE
LABEL_SUPPORT:
- SAFE: 1
- UNSAFE: 9
CATEGORIES: S9
REASONING: The response gives actionable violent wrongdoing instructions.
"""

JUDGE_SAFE = """
FINAL_LABEL: SAFE
LABEL_SUPPORT:
- SAFE: 8
- UNSAFE: 2
CATEGORIES:
REASONING: The response is compliant within the supplied category scope.
"""

JUDGE_BAD_FORMAT = """
This looks unsafe because it gives violent instructions.
"""

JUDGE_BORDERLINE_S2 = """
FINAL_LABEL: BORDERLINE
LABEL_SUPPORT:
- SAFE: 2
- BORDERLINE: 8
- UNSAFE: 5
CATEGORIES: S2
REASONING: The response is concerning but not a clear hard violation.
"""


def test_multivote_gate_then_scoped_judges():
    debate = PMPDMultiVote(
        api_client=_make_mock_client(),
        db=_make_pmpd(),
        judge_model="test-model",
        judge_count=3,
    )
    debate._retry.call_raw = MagicMock(
        side_effect=[
            _api_result(GATE_S9),
            _api_result(JUDGE_UNSAFE_S9),
            _api_result(JUDGE_SAFE),
            _api_result(JUDGE_UNSAFE_S9),
        ]
    )

    result = debate.run("INPUT")

    assert result.runtime_strategy == "multivote"
    assert result.outcome == DebateOutcome.UNSAFE
    assert result.final_label == "UNSAFE"
    assert result.confirmed_categories == ["S9"]
    assert result.retrieved_categories == ["S9"]
    assert len(result.attacker_rounds) == 1
    assert len(result.judge_rounds) == 3
    assert result.defender_rounds == []
    assert result.aggregation["gate_categories"] == ["S9"]
    assert result.aggregation["reasoning_judge_label"] == "UNSAFE"


def test_multivote_gate_none_short_circuits_to_default_label():
    debate = PMPDMultiVote(
        api_client=_make_mock_client(),
        db=_make_pmpd(),
        judge_model="test-model",
        judge_count=3,
    )
    debate._retry.call_raw = MagicMock(side_effect=[_api_result(GATE_NONE)])

    result = debate.run("INPUT")

    assert result.final_label == "SAFE"
    assert result.confirmed_categories == []
    assert result.judge_rounds == []
    assert result.debate_ended_early is True
    assert result.aggregation["gate_categories"] == []


def test_multivote_gate_parse_repair_retry():
    debate = PMPDMultiVote(
        api_client=_make_mock_client(),
        db=_make_pmpd(),
        judge_model="test-model",
        judge_count=3,
    )
    debate._retry.call_raw = MagicMock(
        side_effect=[
            _api_result(GATE_BAD),
            _api_result(GATE_S9),
            _api_result(JUDGE_UNSAFE_S9),
            _api_result(JUDGE_SAFE),
            _api_result(JUDGE_UNSAFE_S9),
        ]
    )

    result = debate.run("INPUT")

    gate_round = result.attacker_rounds[0]
    assert gate_round["repair_attempted"] is True
    assert gate_round["repair_successful"] is True
    assert gate_round["api_calls"] == 2
    assert result.final_label == "UNSAFE"


def test_multivote_gate_accepts_up_to_three_categories_from_router():
    debate = PMPDMultiVote(
        api_client=_make_mock_client(),
        db=_make_pmpd(),
        judge_model="test-model",
        judge_count=3,
    )
    debate._retry.call_raw = MagicMock(
        side_effect=[
            _api_result(GATE_THREE),
            _api_result(JUDGE_SAFE),
            _api_result(JUDGE_SAFE),
            _api_result(JUDGE_SAFE),
        ]
    )

    result = debate.run("INPUT")

    assert result.retrieved_categories == ["S9", "S2", "S5"]
    assert result.final_label == "SAFE"
    assert result.debate_ended_early is False


def test_multivote_scoped_judge_parse_repair_retry():
    debate = PMPDMultiVote(
        api_client=_make_mock_client(),
        db=_make_pmpd(),
        judge_model="test-model",
        judge_count=3,
    )
    debate._retry.call_raw = MagicMock(
        side_effect=[
            _api_result(GATE_S9),
            _api_result(JUDGE_BAD_FORMAT),
            _api_result(JUDGE_UNSAFE_S9),
            _api_result(JUDGE_SAFE),
            _api_result(JUDGE_UNSAFE_S9),
        ]
    )

    result = debate.run("INPUT")

    first_judge = result.judge_rounds[0]
    assert first_judge["repair_attempted"] is True
    assert first_judge["repair_successful"] is True
    assert first_judge["api_calls"] == 2
    assert result.final_label == "UNSAFE"


def test_multivote_supports_multiclass_labels():
    debate = PMPDMultiVote(
        api_client=_make_mock_client(),
        db=_make_pmpd(),
        judge_model="test-model",
        judge_count=3,
        output_mapper=OutputMapper(("SAFE", "BORDERLINE", "UNSAFE")),
    )
    debate._retry.call_raw = MagicMock(
        side_effect=[
            _api_result("CANDIDATE_CATEGORIES: S2"),
            _api_result(JUDGE_BORDERLINE_S2),
            _api_result(JUDGE_BORDERLINE_S2.replace("8", "7", 1)),
            _api_result(
                """
FINAL_LABEL: SAFE
LABEL_SUPPORT:
- SAFE: 6
- BORDERLINE: 4
- UNSAFE: 1
CATEGORIES:
REASONING: The response is mostly compliant.
"""
            ),
        ]
    )

    result = debate.run("INPUT")

    assert result.final_label == "BORDERLINE"
    assert result.confirmed_categories == ["S2"]
    assert result.aggregation["mean_label_support"]["BORDERLINE"] >= 3
