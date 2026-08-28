"""
Prompts package.

Generates and manages domain-aware system prompts for the debate agents.

Pipeline
────────
  PromptGenerator — orchestrates generation from a Markdown tree
  PromptValidator — rule-based validation with injectable rules
  PromptWriter    — persists prompts as JSON (replaces .py codegen)
  PromptLoader    — loads prompts from JSON, falls back to defaults

Public API
──────────
    from prompts import PromptGenerator, PromptLoader
"""

from prompts.generator import PromptGenerator
from prompts.loader import PromptLoader
from prompts.harmony_builder import HarmonyPromptBuilder


__all__ = ["PromptGenerator", "PromptLoader"]
