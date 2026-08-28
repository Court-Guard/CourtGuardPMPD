"""
Debate package.

Orchestrates multi-round adversarial debates evaluating AI responses
against policy documents.

Public API
──────────
    from debate import PolicyDebate
"""

from debate.policy_debate import PolicyDebate
from debate.pmpd_debate import PMPDDebate, PMPDDebateResult


__all__ = ["PolicyDebate"]
