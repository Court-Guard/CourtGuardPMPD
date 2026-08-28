"""
Core domain package.

Contains pure domain models and enumerations with zero I/O dependencies.
Nothing in this package should import from infrastructure, rag, debate,
bootstrap, or any other application layer.

Re-exports
──────────
Consumers can import from the package directly:

    from core import PMPD, OneShot, GlobalModule, CategoryModule
    from core import Verdict, OneShotSource, JudgeSeverity
"""

from core.enums import (
    SEVERITY_TO_INT,
    SEVERITY_TO_VERDICT,
    BootstrapModel,
    JudgeSeverity,
    OneShotSource,
    Verdict,
)
from core.models import CategoryModule, GlobalModule, OneShot
from core.pmpd import PMPD

__all__ = [
    # enums
    "BootstrapModel",
    "JudgeSeverity",
    "OneShotSource",
    "SEVERITY_TO_INT",
    "SEVERITY_TO_VERDICT",
    "Verdict",
    # models
    "CategoryModule",
    "GlobalModule",
    "OneShot",
    # database
    "PMPD",
]
