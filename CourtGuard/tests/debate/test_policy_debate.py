"""
Tests for debate/policy_debate.py

All LLM calls are mocked — no real API interaction.

Covers:
  - set_prompts() updates internal prompts
  - debate_history reset between run_debate() calls
  - detect_resume_point() with various history states
  - _is_valid_content() edge cases
  - _handle_debate_api_error() placeholder filling
  - run_debate() happy path (fully mocked)
"""

from unittest.mock import MagicMock, patch

from debate.policy_debate import PolicyDebate
from infrastructure.api_client import APIClient, APIError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Build a mock APIClient that returns success on call_model."""
    client = MagicMock(spec=APIClient)
    client.api_key = "sk-test"
    client.get_usage_stats.return_value = {"total_requests": 1, "model": "test"}
    return client


def _make_debate(mock_client=None) -> PolicyDebate:
    client = mock_client or _make_mock_client()
    return PolicyDebate(api_client=client)


def _mock_retry_call_raw(content: str):
    """Return a result dict as LLMRetryClient.call_raw() would."""
    return {"content": content, "success": True, "response_time": 0.1}


# ---------------------------------------------------------------------------
# set_prompts
# ---------------------------------------------------------------------------


class TestSetPrompts:
    def test_set_prompts_updates_all_three(self):
        debate = _make_debate()
        debate.set_prompts("new_attacker", "new_defender", "new_judge")
        assert debate._prompts["attacker_system"] == "new_attacker"
        assert debate._prompts["defender_system"] == "new_defender"
        assert debate._prompts["judge_system"] == "new_judge"

    def test_set_prompts_replaces_previous(self):
        debate = _make_debate()
        debate.set_prompts("a1", "d1", "j1")
        debate.set_prompts("a2", "d2", "j2")
        assert debate._prompts["attacker_system"] == "a2"


# ---------------------------------------------------------------------------
# _is_valid_content
# ---------------------------------------------------------------------------


class TestIsValidContent:
    def test_valid_content(self):
        debate = _make_debate()
        assert (
            debate._is_valid_content("This is a valid argument about policy violations.") is True
        )

    def test_too_short(self):
        debate = _make_debate()
        assert debate._is_valid_content("Short") is False

    def test_empty_string(self):
        assert _make_debate()._is_valid_content("") is False

    def test_none_falsy(self):
        assert _make_debate()._is_valid_content(None) is False

    def test_invalid_marker_detected(self):
        debate = _make_debate()
        assert debate._is_valid_content("[API Error - attack argument unavailable]") is False

    def test_unavailable_marker(self):
        debate = _make_debate()
        assert debate._is_valid_content("argument unavailable for round 1") is False


# ---------------------------------------------------------------------------
# detect_resume_point
# ---------------------------------------------------------------------------


class TestDetectResumePoint:
    def test_empty_history_starts_at_round_1(self):
        debate = _make_debate()
        round_num, last_a, last_d = debate.detect_resume_point([])
        assert round_num == 1
        assert last_a == ""
        assert last_d == ""

    def test_only_attacker_round_1_resumes_at_1(self):
        debate = _make_debate()
        history = ["ATTACKER Round 1: This is a valid attacker argument about policy."]
        round_num, last_a, last_d = debate.detect_resume_point(history)
        assert round_num == 1
        assert "valid attacker argument" in last_a

    def test_full_round_1_resumes_at_round_2(self):
        debate = _make_debate()
        history = [
            "ATTACKER Round 1: Valid attacker argument with policy citations.",
            "DEFENDER Round 1: Valid defender argument with policy compliance.",
        ]
        round_num, _, _ = debate.detect_resume_point(history)
        assert round_num == 2

    def test_complete_debate_resumes_past_num_rounds(self):
        debate = _make_debate()
        history = [
            "ATTACKER Round 1: Valid attacker round one argument here.",
            "DEFENDER Round 1: Valid defender round one argument here.",
            "ATTACKER Round 2: Valid attacker round two argument here.",
            "DEFENDER Round 2: Valid defender round two argument here.",
        ]
        round_num, _, _ = debate.detect_resume_point(history)
        assert round_num == PolicyDebate.NUM_ROUNDS + 1

    def test_invalid_content_not_counted(self):
        debate = _make_debate()
        history = ["ATTACKER Round 1: [API Error - Attack argument unavailable]"]
        round_num, last_a, _ = debate.detect_resume_point(history)
        assert round_num == 1
        assert last_a == ""


# ---------------------------------------------------------------------------
# debate_history reset
# ---------------------------------------------------------------------------


class TestDebateHistoryReset:
    def test_history_reset_between_runs(self):
        """debate_history must be empty at the start of each run_debate() call."""
        debate = _make_debate()

        # Manually dirty the history
        debate.debate_history = ["leftover from previous run"]

        attacker_content = "This is a substantial attacker argument about violations."
        defender_content = "This is a substantial defender argument about compliance."
        judge_content = (
            "After careful analysis:\n"
            "---VERDICT---\n"
            "SEVERITY: SAFE\n"
            "WINNER: Defender\n"
            "VIOLATED_CATEGORIES: None\n"
            "---END VERDICT---"
        )

        with patch.object(
            debate._retry,
            "call_raw",
            side_effect=[
                _mock_retry_call_raw(attacker_content),
                _mock_retry_call_raw(defender_content),
                _mock_retry_call_raw(attacker_content),
                _mock_retry_call_raw(defender_content),
                _mock_retry_call_raw(judge_content),
            ],
        ):
            debate.run_debate("query", "response", "context")

        # "leftover from previous run" must not appear
        assert not any("leftover" in e for e in debate.debate_history)


# ---------------------------------------------------------------------------
# _handle_debate_api_error
# ---------------------------------------------------------------------------


class TestHandleDebateAPIError:
    def test_fills_missing_rounds_with_placeholders(self):
        debate = _make_debate()
        debate.debate_history = []

        history, result = debate._handle_debate_api_error(APIError("test error"))

        # Should have placeholder entries for all rounds
        for round_num in range(1, PolicyDebate.NUM_ROUNDS + 1):
            assert any(f"ATTACKER Round {round_num}:" in e for e in history)
            assert any(f"DEFENDER Round {round_num}:" in e for e in history)

    def test_result_has_api_error_flag(self):
        debate = _make_debate()
        _, result = debate._handle_debate_api_error(APIError("err"))
        assert result.get("api_error") is True

    def test_judge_evaluation_unsafe(self):
        debate = _make_debate()
        _, result = debate._handle_debate_api_error(APIError("err"))
        assert result["judge_evaluation"]["severity"] == "UNSAFE"

    def test_does_not_duplicate_existing_placeholders(self):
        """If a round already has a placeholder, don't add a second one."""
        debate = _make_debate()
        debate.debate_history = ["ATTACKER Round 1: [API Error - Attack argument unavailable]"]
        history, _ = debate._handle_debate_api_error(APIError("err"))
        attacker_r1 = [e for e in history if e.startswith("ATTACKER Round 1:")]
        assert len(attacker_r1) == 1


# ---------------------------------------------------------------------------
# run_debate happy path
# ---------------------------------------------------------------------------


class TestRunDebateHappyPath:
    def test_run_debate_returns_history_and_result(self):
        debate = _make_debate()

        attacker_content = "This is a detailed attacker argument about policy violations."
        defender_content = "This is a detailed defender argument about policy compliance."
        judge_content = (
            "Analysis complete.\n"
            "---VERDICT---\n"
            "SEVERITY: UNSAFE\n"
            "WINNER: Attacker\n"
            "VIOLATED_CATEGORIES: Defamation\n"
            "---END VERDICT---"
        )

        responses = (
            [_mock_retry_call_raw(attacker_content)]
            + [_mock_retry_call_raw(defender_content)]
            + [_mock_retry_call_raw(attacker_content)]
            + [_mock_retry_call_raw(defender_content)]
            + [_mock_retry_call_raw(judge_content)]
        )

        with patch.object(debate._retry, "call_raw", side_effect=responses):
            history, result = debate.run_debate("query", "response", "context")

        assert isinstance(history, list)
        assert len(history) == PolicyDebate.NUM_ROUNDS * 2
        assert "judge_evaluation" in result
        assert result["judge_evaluation"]["severity"] == "UNSAFE"
        assert result["judge_evaluation"]["winner"] == "Attacker"

    def test_run_debate_result_has_api_stats(self):
        debate = _make_debate()

        content = "Sufficiently long argument about policy violations in detail."
        judge = (
            "---VERDICT---\nSEVERITY: SAFE\nWINNER: Defender\n"
            "VIOLATED_CATEGORIES: None\n---END VERDICT---"
        )
        side_eff = [_mock_retry_call_raw(content)] * 4 + [_mock_retry_call_raw(judge)]

        with patch.object(debate._retry, "call_raw", side_effect=side_eff):
            _, result = debate.run_debate("q", "r", "ctx")

        assert "api_stats" in result
