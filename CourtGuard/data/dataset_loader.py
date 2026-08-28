"""
Dataset Loader

Loads JSON evaluation datasets and resolves index specifications.

Supports the dataset format produced by safety benchmark pipelines:
    [
        {
            "index": 0,
            "user_prompt": "...",
            "target model response": "...",
            "category": "...",
            "label": "...",
            "is_safe": true,
            "type": "..."
        },
        ...
    ]

Index specification formats
────────────────────────────
    "0,3,7,12"   → [0, 3, 7, 12]     comma-separated individual indexes
    "0-10"        → [0,1,2,...,10]    inclusive range
    "0-5,9,11"   → [0,1,2,3,4,5,9,11] mixed

Or use start_index / end_index (inclusive) for range-only CLI args.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Evaluation record
# ---------------------------------------------------------------------------


@dataclass
class EvaluationRecord:
    """
    A single record from the evaluation dataset.

    Attributes
    ----------
    index                 : Original dataset index.
    user_prompt           : The user query submitted to the target model.
    target_model_response : The target model response being evaluated.
    category              : Dataset-provided category label.
    label                 : Dataset-provided label (e.g. "unharmful").
    is_safe               : Dataset ground-truth safety flag.
    record_type           : Dataset record type field (e.g. "prompt_harmful").
    raw                   : The full original dict for pass-through in results.
    """

    index: int
    user_prompt: str
    target_model_response: str
    category: str = ""
    label: str = ""
    is_safe: bool = True
    record_type: str = ""
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> EvaluationRecord:
        """
        Parse an EvaluationRecord from a raw dataset dict.

        Handles both 'target model response' (with space) and
        'target_model_response' (with underscore) key variants.
        """
        response = d.get("target model response") or d.get("target_model_response") or ""
        return cls(
            index=int(d.get("index", 0)),
            user_prompt=d.get("user_prompt", ""),
            target_model_response=response,
            category=d.get("category", ""),
            label=d.get("label", ""),
            is_safe=bool(d.get("is_safe", True)),
            record_type=d.get("type", ""),
            raw=d,
        )


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------


class DatasetLoader:
    """
    Loads a JSON evaluation dataset and resolves index specifications.

    Usage
    -----
        loader  = DatasetLoader("data/datasets/PKU_SafeRLHF_180.json")
        records = loader.load_indexes("0-5,9,11")

        # Or with start/end:
        records = loader.load_range(start=0, end=10)

        # All records:
        records = loader.load_all()
    """

    def __init__(self, data_file: str) -> None:
        """
        Args:
            data_file: Path to the JSON dataset file.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        self._path = data_file
        self._data: list[dict] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_indexes(self, index_spec: str) -> list[EvaluationRecord]:
        """
        Load records matching an index specification string.

        Supported formats:
            "0,3,7"   → specific indexes
            "0-10"    → inclusive range
            "0-5,9"   → mixed

        Args:
            index_spec: Index specification string.

        Returns:
            List of EvaluationRecord objects in index order.
        """
        indexes = self._parse_index_spec(index_spec)
        return self._load_by_indexes(indexes)

    def load_range(self, start: int, end: int) -> list[EvaluationRecord]:
        """
        Load records from start_index to end_index inclusive.

        Args:
            start: First index to include.
            end:   Last index to include (inclusive).

        Returns:
            List of EvaluationRecord objects.
        """
        return self._load_by_indexes(list(range(start, end + 1)))

    def load_all(self) -> list[EvaluationRecord]:
        """Load all records from the dataset."""
        return [EvaluationRecord.from_dict(d) for d in self._get_data()]

    @property
    def total_records(self) -> int:
        """Total number of records in the dataset."""
        return len(self._get_data())

    @property
    def dataset_name(self) -> str:
        """Filename stem — used for output filename generation."""
        return os.path.splitext(os.path.basename(self._path))[0]

    # ------------------------------------------------------------------
    # Index parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_index_spec(spec: str) -> list[int]:
        """
        Parse an index specification string into a sorted list of integers.

        Supports:
            "0,3,7"    → [0, 3, 7]
            "0-10"     → [0, 1, 2, ..., 10]
            "0-5,9,11" → [0, 1, 2, 3, 4, 5, 9, 11]

        Raises:
            ValueError: If the specification is malformed.
        """
        indexes: set[int] = set()

        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                bounds = part.split("-")
                if len(bounds) != 2:
                    raise ValueError(f"Invalid range: '{part}'")
                try:
                    start, end = int(bounds[0].strip()), int(bounds[1].strip())
                except ValueError:
                    raise ValueError(f"Non-integer bounds in range: '{part}'")
                if start > end:
                    raise ValueError(f"Range start {start} > end {end}")
                indexes.update(range(start, end + 1))
            else:
                try:
                    indexes.add(int(part))
                except ValueError:
                    raise ValueError(f"Non-integer index: '{part}'")

        return sorted(indexes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_data(self) -> list[dict]:
        """Lazy-load the dataset JSON."""
        if self._data is None:
            with open(self._path, encoding="utf-8") as f:
                self._data = json.load(f)
            print(f"  📂 Loaded dataset: {self._path} ({len(self._data)} records)")
        return self._data

    def _load_by_indexes(self, indexes: list[int]) -> list[EvaluationRecord]:
        """
        Load records matching the given index list.

        Matches on the 'index' field in the JSON, not list position,
        so sparse datasets with non-sequential indexes are handled.

        Raises:
            ValueError: If any requested index is not found.
        """
        data = self._get_data()
        index_map = {d.get("index", i): d for i, d in enumerate(data)}
        missing = set(indexes) - set(index_map.keys())

        if missing:
            raise ValueError(f"Indexes not found in dataset: {sorted(missing)}")

        return [EvaluationRecord.from_dict(index_map[i]) for i in indexes]
