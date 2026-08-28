"""
Bootstrap package.

Manages CourtGuard startup orchestration — state persistence, stage
sequencing, and API key rotation across all pipeline stages.

Public API
──────────
    from bootstrap import BootstrapOrchestrator, BootstrapResult
    from bootstrap.state_manager import BootstrapStateManager
"""

from bootstrap.orchestrator import BootstrapOrchestrator, BootstrapResult

__all__ = ["BootstrapOrchestrator", "BootstrapResult"]
