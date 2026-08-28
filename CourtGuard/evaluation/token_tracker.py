"""
Token Tracker

Accumulates per-agent token counts and timing across all debate rounds.

Design
──────
TokenTracker is instantiated once per evaluation record.
After each API call, the caller updates the tracker with the result dict
from APIClient.call_model() which now includes input_tokens and output_tokens.

At the end of the evaluation, tracker.summary() returns the complete
token_usage and timing blocks written to the result JSON.

This is the ground truth for cost analysis — the structure is designed
so that adding price_per_token per model in the future requires only
adding a calculation to summary(), not changing any schema.

Token fields in APIClient result dict
──────────────────────────────────────
  result["input_tokens"]   ← prompt_tokens  from OpenAI usage
  result["output_tokens"]  ← completion_tokens from OpenAI usage
  result["tokens_used"]    ← total_tokens (kept for backwards compat)
  result["response_time"]  ← wall clock seconds for this call
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Per-agent usage record
# ---------------------------------------------------------------------------

@dataclass
class AgentTokenUsage:
    """
    Token consumption and timing for one debate agent across all its rounds.

    Attributes
    ----------
    model          : Model identifier string used by this agent.
    role           : "attacker" | "defender" | "judge"
    rounds_run     : How many rounds this agent actually executed.
    input_tokens   : Total prompt tokens consumed across all rounds.
    output_tokens  : Total completion tokens generated across all rounds.
    total_tokens   : input_tokens + output_tokens.
    total_time_s   : Total wall-clock seconds across all rounds.
    raw_calls      : List of per-call result dicts — includes raw response
                     content and token counts for each individual call.
                     Always stored even if parsing failed.
    """

    model:         str
    role:          str
    rounds_run:    int             = 0
    api_calls:     int             = 0
    input_tokens:  int             = 0
    output_tokens: int             = 0
    total_time_s:  float           = 0.0
    raw_calls:     list[dict]      = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def record_call(self, api_result: dict[str, Any]) -> None:
        """
        Accumulate token counts and timing from one APIClient.call_model() result.

        Safe to call even when tokens are None (API didn't report usage).

        Args:
            api_result: Dict returned by APIClient.call_model().
        """
        self.rounds_run   += 1
        self.api_calls    += 1
        self.input_tokens  += api_result.get("input_tokens")  or 0
        self.output_tokens += api_result.get("output_tokens") or 0
        self.total_time_s  += api_result.get("response_time") or 0.0

        # Always store the raw call — safety net for parsing failures
        self.raw_calls.append({
            "round":          self.rounds_run,
            "model":          api_result.get("model_used", self.model),
            "input_tokens":   api_result.get("input_tokens"),
            "output_tokens":  api_result.get("output_tokens"),
            "total_tokens":   api_result.get("tokens_used"),
            "response_time":  api_result.get("response_time"),
            "success":        api_result.get("success", False),
            "raw_content":    api_result.get("content", ""),
        })

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        return {
            "model":         self.model,
            "role":          self.role,
            "rounds_run":    self.rounds_run,
            "api_calls":     self.api_calls,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.total_tokens,
            "total_time_s":  round(self.total_time_s, 3),
        }


# ---------------------------------------------------------------------------
# Token Tracker
# ---------------------------------------------------------------------------

class TokenTracker:
    """
    Tracks token consumption and timing for all agents in one evaluation.

    Usage
    -----
        tracker = TokenTracker(
            attacker_model="openai/gpt-oss-20b",
            defender_model="openai/gpt-oss-20b",
        )

        # After each API call:
        tracker.record("attacker", api_result)
        tracker.record("defender", api_result)

        # At the end:
        usage = tracker.summary()
        # usage is ready to be written to EvaluationResult.token_usage
    """

    def __init__(
        self,
        attacker_model: str = "",
        defender_model: str = "",
        judge_model:    str = "",
    ) -> None:
        """
        Args:
            attacker_model: Model ID for the Attacker agent.
            defender_model: Model ID for the Defender agent.
            judge_model:    Model ID for the Judge agent (empty if not used).
        """
        self._start_time = time.time()

        self._agents: dict[str, AgentTokenUsage] = {
            "attacker": AgentTokenUsage(model=attacker_model, role="attacker"),
            "defender": AgentTokenUsage(model=defender_model, role="defender"),
            "judge":    AgentTokenUsage(model=judge_model,    role="judge"),
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, role: str, api_result: dict[str, Any]) -> None:
        """
        Record one API call result for the given agent role.

        Args:
            role:       "attacker" | "defender" | "judge"
            api_result: Dict from APIClient.call_model().
        """
        if role not in self._agents:
            raise ValueError(
                f"Unknown agent role '{role}'. "
                f"Expected: attacker, defender, judge."
            )
        self._agents[role].record_call(api_result)

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Build the complete token_usage and timing summary for a result record.

        Returns:
            Dict ready to be stored as EvaluationResult.token_usage.
            Structure:
            {
              "attacker": {model, role, rounds_run, input_tokens, ...},
              "defender": {model, role, rounds_run, input_tokens, ...},
              "judge":    {model, role, rounds_run, input_tokens, ...},
              "total_input_tokens":  int,
              "total_output_tokens": int,
              "total_tokens":        int,
              "total_time_s":        float,
              "estimated_cost_usd":  null    ← reserved for future pricing
            }
        """
        total_input  = sum(a.input_tokens  for a in self._agents.values())
        total_output = sum(a.output_tokens for a in self._agents.values())
        total_time   = time.time() - self._start_time

        return {
            "attacker":            self._agents["attacker"].to_dict(),
            "defender":            self._agents["defender"].to_dict(),
            "judge":               self._agents["judge"].to_dict(),
            "total_input_tokens":  total_input,
            "total_output_tokens": total_output,
            "total_tokens":        total_input + total_output,
            "total_time_s":        round(total_time, 3),
            "estimated_cost_usd":  None,   # reserved for future pricing
        }

    def raw_calls(self, role: str) -> list[dict]:
        """
        Return the list of raw per-call dicts for the given agent role.

        Used to store complete raw outputs in the result JSON —
        safety net for token-limit-induced parsing failures.

        Args:
            role: "attacker" | "defender" | "judge"
        """
        return self._agents[role].raw_calls if role in self._agents else []

    def rounds_run(self, role: str) -> int:
        """Number of rounds actually executed for this agent."""
        return self._agents[role].rounds_run if role in self._agents else 0

    def last_call(self, role: str) -> dict[str, Any]:
        """Return the most recent raw call dict for the given role, if any."""
        if role not in self._agents or not self._agents[role].raw_calls:
            return {}
        return self._agents[role].raw_calls[-1]
