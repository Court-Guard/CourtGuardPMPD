"""
Result Writer

Append-only JSON result writer for CourtGuard evaluation runs.

Design
──────
Results are appended record-by-record so a crash at index N does not
lose records 0 to N-1.  Each write:
  1. Reads the existing file (or starts with empty list)
  2. Appends the new record
  3. Writes the full list back

Output filename format
──────────────────────
    CourtGuard_{mode}_{model_slug}_{dataset_name}_{YYYYMMDD_HHMMSS}.json

    e.g.
    CourtGuard_rag_gpt-oss-20b_PKU_SafeRLHF_180_20260320_143022.json
    CourtGuard_pmpd_llama-3.3-70b_PKU_SafeRLHF_180_20260320_143022.json
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Evaluation result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """
    Complete result for one evaluated dataset record.

    Attributes
    ----------
    index                        : Original dataset index.
    user_prompt                  : The evaluated user query.
    target_model_response        : The evaluated AI response.
    category                     : Dataset category label.
    label                        : Dataset ground-truth label.
    is_safe                      : Dataset ground-truth safety flag.
    courtguard_mode              : "rag" or "pmpd".
    courtguard_model             : Model identifier string used for debate.
    courtguard_verdict           : "SAFE" | "BORDERLINE" | "UNSAFE".
    courtguard_winner            : "Attacker" | "Defender".
    courtguard_violated_categories: Categories flagged by Judge.
    courtguard_reasoning         : Judge's detailed reasoning.
    courtguard_severity_raw      : Raw severity string from verdict block.
    debate_history               : Full debate turn history list.
    timing_info                  : Per-turn timing strings.
    evaluated_at                 : ISO timestamp of evaluation.
    api_error                    : True if debate was incomplete due to API error.
    """

    index: int
    user_prompt: str
    target_model_response: str
    category: str = ""
    label: str = ""
    is_safe: bool = True
    courtguard_mode: str = ""
    courtguard_model: str = ""
    courtguard_verdict: str = ""
    courtguard_winner: str = ""
    courtguard_violated_categories: list[str] = field(default_factory=list)
    courtguard_reasoning: str = ""
    courtguard_severity_raw: str = ""
    debate_history: list[str] = field(default_factory=list)
    timing_info: list[str] = field(default_factory=list)
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    api_error: bool = False
    outcome: str = ""
    final_label: str = ""
    preliminary_label: str = ""
    attacker_categories: list[str] = field(default_factory=list)
    confirmed_categories: list[str] = field(default_factory=list)
    rounds_configured: int = 0
    rounds_run: int = 0
    debate_ended_early: bool = False
    token_usage: dict = field(default_factory=dict)
    attacker_raw_rounds: list[dict] = field(default_factory=list)
    defender_raw_rounds: list[dict] = field(default_factory=list)
    judge_raw_rounds: list[dict] = field(default_factory=list)
    runtime_strategy: str = ""
    retrieved_categories: list[str] = field(default_factory=list)
    multivote_summary: dict = field(default_factory=dict)
    output_labels_used: list[str] = field(default_factory=list)
    default_output_label: str = ""
    error_output_label: str = ""


# ---------------------------------------------------------------------------
# Result Writer
# ---------------------------------------------------------------------------


class ResultWriter:
    """
    Append-only JSON result writer.

    Constructs the output filename automatically from mode, model,
    dataset name, and timestamp.  Creates the output directory if needed.

    Usage
    -----
        writer = ResultWriter(
            mode=         "pmpd",
            model=        "openai/gpt-oss-20b",
            dataset_name= "PKU_SafeRLHF_180",
            output_dir=   "data/results",
        )
        writer.append(result)   # called after each index
        print(writer.output_path)
    """

    def __init__(
        self,
        mode: str,
        model: str,
        dataset_name: str,
        output_dir: str = "data/results",
        output_file: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        """
        Args:
            mode:         "rag" or "pmpd".
            model:        Model identifier, e.g. "openai/gpt-oss-20b".
            dataset_name: Dataset filename stem.
            output_dir:   Directory to write results into.
            output_file:  Optional exact filename to bypass timestamp generation.
            timestamp:    Optional fixed timestamp string (for testing).
                          Defaults to current time as YYYYMMDD_HHMMSS.
        """
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if output_file:
            self._path = self._resolve_output_path(output_dir, output_file)
        else:
            ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            model_slug = self._slugify(model)
            dataset_slug = self._slugify(dataset_name)
            filename = f"CourtGuard_{mode}_{model_slug}_{dataset_slug}_{ts}.json"
            self._path = os.path.join(output_dir, filename)

        parent_dir = os.path.dirname(self._path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    @property
    def output_path(self) -> str:
        """Absolute path to the output file."""
        return self._path

    def append(self, result: EvaluationResult) -> None:
        """
        Append a single result to the output file.

        Reads the existing array, appends the new record, and writes back.
        Safe to call after every index — partial results are never lost.

        Args:
            result: EvaluationResult to append.
        """
        existing = self._load_existing()
        existing.append(self._serialise(result))

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def record_count(self) -> int:
        """Number of records already written to the output file."""
        return len(self._load_existing())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing(self) -> list[dict]:
        """Load the existing results array, or return empty list."""
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def _serialise(result: EvaluationResult) -> dict:
        """Convert EvaluationResult to a plain dict for JSON output."""
        return asdict(result)

    @staticmethod
    def _resolve_output_path(output_dir: str, output_file: str) -> str:
        """
        Resolve explicit output paths consistently.

        Bare filenames are written under output_dir.
        Relative paths that already include directories are respected as-is.
        Absolute paths are respected as-is.
        """
        if os.path.isabs(output_file):
            return output_file
        if os.path.dirname(output_file):
            return output_file
        return os.path.join(output_dir, output_file)

    @staticmethod
    def _slugify(text: str) -> str:
        """
        Convert a string to a safe filename slug.

        e.g. "openai/gpt-oss-20b" → "gpt-oss-20b"
             "meta-llama/llama-3.3-70b-instruct" → "llama-3.3-70b-instruct"
        """
        # Take only the part after the last '/' (provider prefix removal)
        slug = text.split("/")[-1]
        # Replace any remaining unsafe characters with hyphens
        slug = re.sub(r"[^\w\-]", "-", slug)
        return slug
