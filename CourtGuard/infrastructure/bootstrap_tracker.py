"""
Bootstrap Tracker

Accumulates per-stage token counts, API call counts, and latency for
the CourtGuard bootstrap pipeline (Stage 3: Prompt Generation,
Stage 4: PMPD Parsing).

Design
──────
BootstrapTracker is injected into PromptGenerator and PMPDParser.
After each LLM call the component reports the raw APIClient result dict
via tracker.record(stage, api_result).

At the end of the bootstrap run, BootstrapOrchestrator calls
tracker.save(path) which persists the stats to bootstrap_stats.json.

This mirrors the design of evaluation/token_tracker.py but covers the
bootstrap pipeline instead of the debate loop.

Stats structure saved to JSON
──────────────────────────────
{
  "saved_at": "2026-04-04T22:00:00",
  "run_id": "2026-04-04T22:00:00",
  "stages": {
    "pmpd_parsing": {
      "api_calls": 7,
      "input_tokens": 42100,
      "output_tokens": 8300,
      "total_time_s": 14.2,
      "errors": 0
    },
    "prompt_generation": {
      "api_calls": 3,
      "input_tokens": 9400,
      "output_tokens": 1200,
      "total_time_s": 4.1,
      "errors": 0
    }
  },
  "totals": {
    "api_calls": 10,
    "input_tokens": 51500,
    "output_tokens": 9500,
    "total_time_s": 18.3,
    "errors": 0
  }
}
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Per-stage stats
# ---------------------------------------------------------------------------


@dataclass
class StageStats:
    """
    Accumulated usage statistics for one bootstrap stage.

    Attributes
    ----------
    stage         : Stage identifier (e.g. "pmpd_parsing")
    api_calls     : Number of successful LLM calls made
    input_tokens  : Total prompt tokens consumed
    output_tokens : Total completion tokens generated
    total_time_s  : Total wall-clock seconds for all calls
    errors        : Calls that returned success=False or raised
    """

    stage:         str
    api_calls:     int   = 0
    input_tokens:  int   = 0
    output_tokens: int   = 0
    total_time_s:  float = 0.0
    errors:        int   = 0

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def record(self, api_result: dict[str, Any]) -> None:
        """
        Accumulate one API call result.

        Args:
            api_result: Dict returned by APIClient.call_model() or
                        LLMRetryClient.call_raw().
        """
        if api_result.get("success", False):
            self.api_calls += 1
        else:
            self.errors += 1

        self.input_tokens  += api_result.get("input_tokens")  or 0
        self.output_tokens += api_result.get("output_tokens") or 0
        self.total_time_s  += api_result.get("response_time") or 0.0

    def record_error(self) -> None:
        """Record a call that raised an exception (no result dict available)."""
        self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        return {
            "api_calls":     self.api_calls,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.total_tokens,
            "total_time_s":  round(self.total_time_s, 3),
            "errors":        self.errors,
        }


# ---------------------------------------------------------------------------
# Bootstrap Tracker
# ---------------------------------------------------------------------------


# Canonical stage identifiers used across the pipeline
STAGE_PMPD_PARSING      = "pmpd_parsing"
STAGE_PROMPT_GENERATION = "prompt_generation"
ALL_STAGES              = [STAGE_PMPD_PARSING, STAGE_PROMPT_GENERATION]


class BootstrapTracker:
    """
    Tracks token consumption, API calls, and latency for bootstrap stages.

    Usage
    -----
        tracker = BootstrapTracker()

        # Inside PMPDParser, after each LLM call:
        tracker.record("pmpd_parsing", raw_result)

        # Inside PromptGenerator, after each LLM call:
        tracker.record("prompt_generation", raw_result)

        # After bootstrap completes:
        tracker.save("bootstrap_stats.json")
    """

    def __init__(self) -> None:
        self._start_time = time.time()
        self._run_id = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        self._stages: dict[str, StageStats] = {
            stage: StageStats(stage=stage) for stage in ALL_STAGES
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, stage: str, api_result: dict[str, Any]) -> None:
        """
        Record one API call result for the given stage.

        Silently creates a new StageStats entry if `stage` is not one of
        the canonical stages — this makes the tracker forward-compatible.

        Args:
            stage:      Stage identifier (use STAGE_* constants).
            api_result: Dict from APIClient.call_model() or call_raw().
        """
        if stage not in self._stages:
            self._stages[stage] = StageStats(stage=stage)
        self._stages[stage].record(api_result)

    def record_error(self, stage: str) -> None:
        """
        Record a call that raised an exception before producing a result dict.

        Args:
            stage: Stage identifier.
        """
        if stage not in self._stages:
            self._stages[stage] = StageStats(stage=stage)
        self._stages[stage].record_error()

    # ------------------------------------------------------------------
    # Summary & persistence
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Build the full stats dict ready for JSON serialisation.

        Returns:
            Dict with per-stage stats and aggregated totals.
        """
        stages_dict = {stage: stats.to_dict() for stage, stats in self._stages.items()}

        total_calls   = sum(s.api_calls    for s in self._stages.values())
        total_in      = sum(s.input_tokens  for s in self._stages.values())
        total_out     = sum(s.output_tokens for s in self._stages.values())
        total_time    = time.time() - self._start_time
        total_errors  = sum(s.errors        for s in self._stages.values())

        return {
            "run_id":   self._run_id,
            "saved_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "stages":   stages_dict,
            "totals": {
                "api_calls":     total_calls,
                "input_tokens":  total_in,
                "output_tokens": total_out,
                "total_tokens":  total_in + total_out,
                "total_time_s":  round(total_time, 3),
                "errors":        total_errors,
            },
        }

    def save(self, path: str) -> None:
        """
        Persist stats to a JSON file.

        If the file already exists, the new run is **appended** to a
        ``"runs"`` list so the history of bootstrap runs is preserved.

        Args:
            path: File path for the JSON stats file.
        """
        data = self.summary()

        existing: dict = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (json.JSONDecodeError, OSError):
                existing = {}

        runs: list = existing.get("runs", [])
        runs.append(data)

        output = {
            "description": (
                "CourtGuard bootstrap pipeline usage statistics. "
                "Each entry in 'runs' covers one full bootstrap execution."
            ),
            "runs": runs,
            "latest": data,
        }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)

        print(
            f"\n  📊 Bootstrap stats saved → {path}"
            f"\n     API calls : {data['totals']['api_calls']}"
            f"\n     Tokens in : {data['totals']['input_tokens']:,}"
            f"\n     Tokens out: {data['totals']['output_tokens']:,}"
            f"\n     Total time: {data['totals']['total_time_s']}s"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted bootstrap usage summary to stdout."""
        data = self.summary()
        print(f"\n{'─'*50}")
        print("  Bootstrap Usage Summary")
        print(f"{'─'*50}")
        for stage, stats in data["stages"].items():
            if stats["api_calls"] > 0 or stats["errors"] > 0:
                print(f"  [{stage}]")
                print(f"    API calls  : {stats['api_calls']}")
                print(f"    Tokens in  : {stats['input_tokens']:,}")
                print(f"    Tokens out : {stats['output_tokens']:,}")
                print(f"    Time       : {stats['total_time_s']}s")
                if stats["errors"]:
                    print(f"    Errors     : {stats['errors']}")
        t = data["totals"]
        print(f"{'─'*50}")
        print(f"  TOTAL  api_calls={t['api_calls']}  "
              f"tokens={t['total_tokens']:,}  time={t['total_time_s']}s")
        print(f"{'─'*50}\n")
