"""
Data package.

Handles dataset loading and index parsing for CourtGuard evaluations.

Public API
──────────
    from data.dataset_loader import DatasetLoader, EvaluationRecord
"""

from data.dataset_loader import DatasetLoader, EvaluationRecord

__all__ = ["DatasetLoader", "EvaluationRecord"]
