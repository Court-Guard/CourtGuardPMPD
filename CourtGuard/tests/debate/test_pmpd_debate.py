"""Tests for the PMPD selector/specialist runtime."""

from unittest.mock import MagicMock

import pytest

from core.enums import DebateOutcome
from core.models import CategoryModule, GlobalModule
from core.pmpd import PMPD
from debate.pmpd_debate import PMPDDebate
from infrastructure.api_client import APIClient, APIError


def _make_pmpd() -> PMPD:
    db = PMPD.from_scratch()
    db.set_global_module(
        GlobalModule(
            objective="Test objective.",
            principles=[],
            schema="Output label.",
        )
    )
    db.add_category(
        CategoryModule(
            category_id="S9",
            name="Violent Crimes",
            general_rules="No violence.",
            exceptions=[],
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


SELECTOR_UNSAFE_S9 = """
CANDIDATE_CATEGORIES: S9
EVIDENCE_SPANS: "instructions for violence"
"""

SELECTOR_UNSAFE_S2 = """
CANDIDATE_CATEGORIES: S2
EVIDENCE_SPANS: "false allegation"
"""

SELECTOR_MULTI_CATEGORY = """
CANDIDATE_CATEGORIES: S9, S2
EVIDENCE_SPANS: "instructions for violence"; "false allegation"
"""

SELECTOR_SAFE = """
CANDIDATE_CATEGORIES:
EVIDENCE_SPANS:
"""

SELECTOR_INVALID = """
CANDIDATE_CATEGORIES: VCR
EVIDENCE_SPANS: "instructions for violence"
"""

SELECTOR_BAD_FORMAT = """
This likely violates S9 because it gives instructions for violence.
"""

SPECIALIST_SUPPORTED_UNSAFE = """
CATEGORY_FIT: SUPPORTED
FINAL_LABEL: UNSAFE
CATEGORIES: S9
BASIS_CODE: MATCHED_POLICY
CATEGORY_NOTES: The supplied category directly matches the input.
REASONING: The response gives violent wrongdoing instructions.
"""

SPECIALIST_SUPPORTED_SAFE = """
CATEGORY_FIT: SUPPORTED
FINAL_LABEL: SAFE
CATEGORIES:
BASIS_CODE: NO_VIOLATION
CATEGORY_NOTES: The supplied category is sufficient and no violation is present.
REASONING: The response is compliant within the provided category scope.
"""

SPECIALIST_UNSUPPORTED = """
CATEGORY_FIT: UNSUPPORTED
FINAL_LABEL: SAFE
CATEGORIES:
BASIS_CODE: INSUFFICIENT_COVERAGE
CATEGORY_NOTES: The supplied categories do not align closely enough with the input.
REASONING: The category scope is insufficient for a reliable judgment.
"""

SPECIALIST_INVALID = """
CATEGORY_FIT: SUPPORTED
FINAL_LABEL: UNSAFE
CATEGORIES: VCR
BASIS_CODE: MATCHED_POLICY
CATEGORY_NOTES: The supplied category directly matches the input.
REASONING: The response gives violent wrongdoing instructions.
"""


@pytest.fixture
def db() -> PMPD:
    return _make_pmpd()


def _make_debate(db, max_rounds=2, call_raw_side_effect=None) -> PMPDDebate:
    client = _make_mock_client()
    debate = PMPDDebate(
        api_client=client,
        db=db,
        max_rounds=max_rounds,
        attacker_model="test-model",
        defender_model="test-model",
        judge_model="test-model",
    )
    if call_raw_side_effect is not None:
        debate._retry.call_raw = MagicMock(side_effect=call_raw_side_effect)
    return debate


class TestSelectorAndSpecialistHappyPath:
    def test_selector_safe_skips_specialist(self, db):
        debate = _make_debate(db, call_raw_side_effect=[_api_result(SELECTOR_SAFE)])
        result = debate.run("USER PROMPT:\ntest query")

        assert result.outcome == DebateOutcome.EARLY_SAFE
        assert result.final_label == "SAFE"
        assert result.rounds_run == 1
        assert len(result.attacker_rounds) == 1
        assert len(result.defender_rounds) == 0

    def test_supported_unsafe_round_one_returns_unsafe(self, db):
        debate = _make_debate(
            db,
            call_raw_side_effect=[
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_SUPPORTED_UNSAFE),
            ],
        )
        result = debate.run("input")

        assert result.outcome == DebateOutcome.UNSAFE
        assert result.final_label == "UNSAFE"
        assert result.confirmed_categories == ["S9"]
        assert result.debate_ended_early is True

    def test_supported_safe_round_one_returns_safe(self, db):
        debate = _make_debate(
            db,
            call_raw_side_effect=[
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_SUPPORTED_SAFE),
            ],
        )
        result = debate.run("input")

        assert result.outcome == DebateOutcome.SAFE
        assert result.final_label == "SAFE"
        assert result.confirmed_categories == []

    def test_round_one_selector_is_truncated_to_single_category(self, db):
        debate = _make_debate(
            db,
            call_raw_side_effect=[
                _api_result(SELECTOR_MULTI_CATEGORY),
                _api_result(SPECIALIST_SUPPORTED_UNSAFE),
            ],
        )
        result = debate.run("input")

        assert result.attacker_rounds[0]["categories"] == ["S9"]
        specialist_prompt = debate._retry.call_raw.call_args_list[1].kwargs["prompt"]
        assert "Selected categories: S9" in specialist_prompt
        assert "Selected categories: S9, S2" not in specialist_prompt


class TestRoundTwoCoverageRepair:
    def test_unsupported_then_new_categories_triggers_round_two_specialist(self, db):
        debate = _make_debate(
            db,
            max_rounds=2,
            call_raw_side_effect=[
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_UNSUPPORTED),
                _api_result(SELECTOR_UNSAFE_S2),
                _api_result(
                    """
                    CATEGORY_FIT: SUPPORTED
                    FINAL_LABEL: UNSAFE
                    CATEGORIES: S2
                    BASIS_CODE: MATCHED_POLICY
                    CATEGORY_NOTES: The supplied category now matches the input.
                    REASONING: The response contains false harmful allegations.
                    """
                ),
            ],
        )
        result = debate.run("input")

        assert result.outcome == DebateOutcome.UNSAFE
        assert result.final_label == "UNSAFE"
        assert result.confirmed_categories == ["S2"]
        assert result.rounds_run == 2
        assert len(result.defender_rounds) == 2

    def test_same_categories_after_unsupported_skips_second_specialist_and_defaults_safe(self, db):
        debate = _make_debate(
            db,
            max_rounds=2,
            call_raw_side_effect=[
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_UNSUPPORTED),
                _api_result(SELECTOR_UNSAFE_S9),
            ],
        )
        result = debate.run("input")

        assert result.outcome == DebateOutcome.SAFE
        assert result.final_label == "SAFE"
        assert result.rounds_run == 2
        assert len(result.attacker_rounds) == 2
        assert len(result.defender_rounds) == 1

    def test_round_two_selector_receives_specialist_feedback_not_attacker_reasoning(self, db):
        debate = _make_debate(
            db,
            max_rounds=2,
            call_raw_side_effect=[
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_UNSUPPORTED),
                _api_result(SELECTOR_UNSAFE_S2),
                _api_result(
                    """
                    CATEGORY_FIT: SUPPORTED
                    FINAL_LABEL: UNSAFE
                    CATEGORIES: S2
                    BASIS_CODE: MATCHED_POLICY
                    CATEGORY_NOTES: The supplied category now matches the input.
                    REASONING: The response contains false harmful allegations.
                    """
                ),
            ],
        )
        debate.run("input")

        round_two_prompt = debate._retry.call_raw.call_args_list[2].kwargs["prompt"]
        assert "PREVIOUS SPECIALIST FEEDBACK" in round_two_prompt
        assert "Category fit: UNSUPPORTED" in round_two_prompt
        assert "Basis code: INSUFFICIENT_COVERAGE" in round_two_prompt
        assert "Do not repeat the exact same rejected set." in round_two_prompt
        assert "you may return 1-2 alternative categories" in round_two_prompt
        assert "ATTACKER ACCUSATION" not in round_two_prompt

    def test_specialist_handoff_contains_only_selector_hints(self, db):
        debate = _make_debate(
            db,
            call_raw_side_effect=[
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_SUPPORTED_UNSAFE),
            ],
        )
        debate.run("input")

        specialist_prompt = debate._retry.call_raw.call_args_list[1].kwargs["prompt"]
        assert "SELECTOR OUTPUT" in specialist_prompt
        assert "ATTACKER ACCUSATION" not in specialist_prompt
        assert "ATTACKER PRELIMINARY LABEL" not in specialist_prompt
        assert "Evidence spans:" not in specialist_prompt
        assert "scope only; not proof" in specialist_prompt


class TestParseRepairAndTelemetry:
    def test_selector_parse_repair_retry(self, db):
        debate = _make_debate(
            db,
            call_raw_side_effect=[
                _api_result(SELECTOR_BAD_FORMAT),
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_SUPPORTED_UNSAFE),
            ],
        )
        result = debate.run("input")

        round_one = result.attacker_rounds[0]
        assert round_one["repair_attempted"] is True
        assert round_one["repair_successful"] is True
        assert round_one["api_calls"] == 2
        assert round_one["total_tokens"] == 300
        assert result.outcome == DebateOutcome.UNSAFE

    def test_specialist_parse_repair_retry(self, db):
        debate = _make_debate(
            db,
            call_raw_side_effect=[
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_INVALID),
                _api_result(SPECIALIST_SUPPORTED_UNSAFE),
            ],
        )
        result = debate.run("input")

        round_one = result.defender_rounds[0]
        assert round_one["repair_attempted"] is True
        assert round_one["repair_successful"] is True
        assert round_one["api_calls"] == 2
        assert round_one["categories"] == ["S9"]
        assert result.outcome == DebateOutcome.UNSAFE

    def test_selector_invalid_category_without_repair_marks_parse_unsuccessful(self, db):
        debate = _make_debate(
            db,
            max_rounds=1,
            call_raw_side_effect=[
                _api_result(SELECTOR_INVALID),
                _api_result(SELECTOR_INVALID),
            ],
        )
        result = debate.run("input")

        round_one = result.attacker_rounds[0]
        assert round_one["categories"] == []
        assert round_one["parse_success"] is False
        assert round_one["invalid_category_mismatch"] is True
        assert result.outcome == DebateOutcome.SAFE

    def test_round_telemetry_aggregates_repair_calls(self, db):
        debate = _make_debate(
            db,
            call_raw_side_effect=[
                _api_result(SELECTOR_BAD_FORMAT, input_tokens=80, output_tokens=20),
                _api_result(SELECTOR_UNSAFE_S9, input_tokens=50, output_tokens=10),
                _api_result(SPECIALIST_SUPPORTED_UNSAFE, input_tokens=60, output_tokens=40),
            ],
        )
        result = debate.run("input")

        assert result.attacker_rounds[0]["input_tokens"] == 130
        assert result.attacker_rounds[0]["output_tokens"] == 30
        assert result.attacker_rounds[0]["total_tokens"] == 160
        assert result.attacker_rounds[0]["response_time_s"] == 1.0
        assert result.defender_rounds[0]["input_tokens"] == 60
        assert result.defender_rounds[0]["output_tokens"] == 40


class TestAPIErrorHandling:
    def test_api_error_returns_error_outcome(self, db):
        debate = _make_debate(db)
        debate._retry.call_raw = MagicMock(side_effect=APIError("rate limit"))
        result = debate.run("input")

        assert result.outcome == DebateOutcome.ERROR
        assert result.api_error is True
        assert "rate limit" in result.error_message

    def test_partial_rounds_preserved_on_later_error(self, db):
        debate = _make_debate(
            db,
            max_rounds=2,
            call_raw_side_effect=[
                _api_result(SELECTOR_UNSAFE_S9),
                _api_result(SPECIALIST_UNSUPPORTED),
                APIError("server error"),
            ],
        )
        result = debate.run("input")

        assert len(result.attacker_rounds) == 1
        assert len(result.defender_rounds) == 1
