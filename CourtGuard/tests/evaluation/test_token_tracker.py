"""
Tests for evaluation/token_tracker.py

Create as: tests/evaluation/test_token_tracker.py

Covers:
  - AgentTokenUsage.record_call() accumulates correctly
  - AgentTokenUsage handles None token values gracefully
  - AgentTokenUsage.total_tokens property
  - AgentTokenUsage.to_dict() shape
  - TokenTracker.record() dispatches to correct agent
  - TokenTracker.summary() totals correct
  - TokenTracker.raw_calls() returns stored calls
  - TokenTracker unknown role raises ValueError
"""

import pytest
import time

from evaluation.token_tracker import AgentTokenUsage, TokenTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    input_tokens=100,
    output_tokens=50,
    response_time=1.5,
    content="test response",
    success=True,
    model="openai/gpt-oss-20b",
):
    return {
        "content":       content,
        "success":       success,
        "response_time": response_time,
        "model_used":    model,
        "tokens_used":   input_tokens + output_tokens if input_tokens else None,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# AgentTokenUsage tests
# ---------------------------------------------------------------------------

class TestAgentTokenUsage:
    def test_initial_state_zero(self):
        agent = AgentTokenUsage(model="test-model", role="attacker")
        assert agent.input_tokens  == 0
        assert agent.output_tokens == 0
        assert agent.total_tokens  == 0
        assert agent.rounds_run    == 0
        assert agent.raw_calls     == []

    def test_record_call_accumulates(self):
        agent = AgentTokenUsage(model="test", role="attacker")
        agent.record_call(_make_result(100, 50))
        assert agent.input_tokens  == 100
        assert agent.output_tokens == 50
        assert agent.total_tokens  == 150
        assert agent.rounds_run    == 1

    def test_multiple_calls_accumulate(self):
        agent = AgentTokenUsage(model="test", role="attacker")
        agent.record_call(_make_result(100, 50))
        agent.record_call(_make_result(200, 80))
        assert agent.input_tokens  == 300
        assert agent.output_tokens == 130
        assert agent.rounds_run    == 2

    def test_none_tokens_handled_gracefully(self):
        agent = AgentTokenUsage(model="test", role="attacker")
        agent.record_call(_make_result(input_tokens=None, output_tokens=None))
        assert agent.input_tokens  == 0
        assert agent.output_tokens == 0
        assert agent.rounds_run    == 1

    def test_raw_calls_stored(self):
        agent = AgentTokenUsage(model="test", role="attacker")
        agent.record_call(_make_result(100, 50, content="the response"))
        assert len(agent.raw_calls)              == 1
        assert agent.raw_calls[0]["raw_content"] == "the response"
        assert agent.raw_calls[0]["round"]       == 1

    def test_raw_calls_always_stored_on_failure(self):
        agent = AgentTokenUsage(model="test", role="attacker")
        agent.record_call(_make_result(success=False, content=""))
        assert len(agent.raw_calls)         == 1
        assert agent.raw_calls[0]["success"] is False

    def test_to_dict_shape(self):
        agent = AgentTokenUsage(model="openai/gpt-oss-20b", role="defender")
        agent.record_call(_make_result(100, 50))
        d = agent.to_dict()
        assert d["model"]         == "openai/gpt-oss-20b"
        assert d["role"]          == "defender"
        assert d["rounds_run"]    == 1
        assert d["input_tokens"]  == 100
        assert d["output_tokens"] == 50
        assert d["total_tokens"]  == 150
        assert "total_time_s" in d


# ---------------------------------------------------------------------------
# TokenTracker tests
# ---------------------------------------------------------------------------

class TestTokenTracker:
    def test_record_dispatches_to_correct_agent(self):
        tracker = TokenTracker(
            attacker_model="model-a",
            defender_model="model-d",
        )
        tracker.record("attacker", _make_result(100, 50))
        tracker.record("defender", _make_result(200, 80))

        summary = tracker.summary()
        assert summary["attacker"]["input_tokens"]  == 100
        assert summary["defender"]["input_tokens"]  == 200

    def test_summary_totals(self):
        tracker = TokenTracker(
            attacker_model="model-a",
            defender_model="model-d",
        )
        tracker.record("attacker", _make_result(100, 50))
        tracker.record("defender", _make_result(200, 80))

        summary = tracker.summary()
        assert summary["total_input_tokens"]  == 300
        assert summary["total_output_tokens"] == 130
        assert summary["total_tokens"]        == 430

    def test_summary_estimated_cost_is_none(self):
        tracker = TokenTracker()
        summary = tracker.summary()
        assert summary["estimated_cost_usd"] is None

    def test_summary_has_all_agents(self):
        tracker = TokenTracker()
        summary = tracker.summary()
        assert "attacker" in summary
        assert "defender" in summary
        assert "judge"    in summary

    def test_judge_zero_when_not_called(self):
        tracker = TokenTracker(judge_model="judge-model")
        summary = tracker.summary()
        assert summary["judge"]["rounds_run"]   == 0
        assert summary["judge"]["total_tokens"] == 0

    def test_raw_calls_accessible(self):
        tracker = TokenTracker(attacker_model="model-a")
        tracker.record("attacker", _make_result(100, 50, content="raw text"))
        calls = tracker.raw_calls("attacker")
        assert len(calls)             == 1
        assert calls[0]["raw_content"] == "raw text"

    def test_unknown_role_raises(self):
        tracker = TokenTracker()
        with pytest.raises(ValueError, match="Unknown agent role"):
            tracker.record("unknown_role", _make_result())

    def test_rounds_run_tracking(self):
        tracker = TokenTracker(attacker_model="m")
        tracker.record("attacker", _make_result())
        tracker.record("attacker", _make_result())
        assert tracker.rounds_run("attacker") == 2
        assert tracker.rounds_run("defender") == 0

    def test_multiple_rounds_per_agent(self):
        tracker = TokenTracker(attacker_model="m", defender_model="d")
        for _ in range(3):
            tracker.record("attacker", _make_result(50, 25))
        tracker.record("defender", _make_result(300, 100))

        summary = tracker.summary()
        assert summary["attacker"]["rounds_run"]   == 3
        assert summary["attacker"]["input_tokens"] == 150
        assert summary["defender"]["rounds_run"]   == 1