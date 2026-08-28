"""
Bootstrap State Manager

Handles persistence of the bootstrap pipeline state and detection of
policy artefacts on disk.

Extracted from duplicated free functions in main.py and pmpd_main.py:
  _load_bootstrap_state()   × 2
  _save_bootstrap_state()   × 2
  _detect_policy_pdf()      × 2
  _load_rag_config()        × 2
  _save_rag_config()        × 2

The state file (.bootstrap_state.json) tracks which stages have
completed so subsequent runs skip already-finished work.

State schema
────────────
{
  "ingested":          bool,
  "rag_tuned":         bool,
  "prompts_generated": bool,
  "pmpd_parsed":       bool,
  "policy_file":       str | null
}
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass

from infrastructure.config import PathConfig

# ---------------------------------------------------------------------------
# Bootstrap state dataclass
# ---------------------------------------------------------------------------


@dataclass
class BootstrapState:
    """
    Typed representation of the bootstrap pipeline completion state.

    Attributes
    ----------
    ingested          : Stage 0 (PDF → Markdown tree) complete.
    rag_tuned         : Stage 1 (RAG parameter tuning) complete.
    prompts_generated : Stage 3 (role prompt generation) complete.
    pmpd_parsed       : Stage 4 (PMPD database build) complete.
    policy_file       : Absolute path to the PDF that was ingested,
                        or None if no ingestion has run yet.
                        Used to detect when a new PDF is dropped.
    """

    ingested: bool = False
    rag_tuned: bool = False
    prompts_generated: bool = False
    pmpd_parsed: bool = False
    policy_file: str | None = None

    def reset(self) -> None:
        """Reset all stage flags (force-bootstrap mode)."""
        self.ingested = False
        self.rag_tuned = False
        self.prompts_generated = False
        self.pmpd_parsed = False
        self.policy_file = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BootstrapState:
        return cls(
            ingested=d.get("ingested", False),
            rag_tuned=d.get("rag_tuned", False),
            prompts_generated=d.get("prompts_generated", False),
            pmpd_parsed=d.get("pmpd_parsed", False),
            policy_file=d.get("policy_file"),
        )


# ---------------------------------------------------------------------------
# Bootstrap State Manager
# ---------------------------------------------------------------------------


class BootstrapStateManager:
    """
    Persists and restores bootstrap pipeline state between runs.

    Also handles detection of the policy PDF and RAG config file —
    artefacts that live alongside the state file.

    Usage
    -----
        paths   = PathConfig.default()
        manager = BootstrapStateManager(paths)
        state   = manager.load_state()
        # ... run stages ...
        state.ingested = True
        manager.save_state(state)
    """

    def __init__(self, paths: PathConfig | None = None) -> None:
        """
        Args:
            paths: PathConfig instance. Defaults to PathConfig.default().
        """
        self._paths = paths or PathConfig.default()

    # ------------------------------------------------------------------
    # Bootstrap state
    # ------------------------------------------------------------------

    def load_state(self) -> BootstrapState:
        """
        Load bootstrap state from disk.

        Returns a fresh default BootstrapState if the file does not
        exist or is corrupt.
        """
        if not os.path.exists(self._paths.bootstrap_state):
            return BootstrapState()
        try:
            with open(self._paths.bootstrap_state, encoding="utf-8") as f:
                return BootstrapState.from_dict(json.load(f))
        except Exception:
            return BootstrapState()

    def save_state(self, state: BootstrapState) -> None:
        """
        Persist bootstrap state to disk.

        Args:
            state: Current BootstrapState to persist.
        """
        with open(self._paths.bootstrap_state, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # Policy Document detection
    # ------------------------------------------------------------------

    def detect_policy_document(self) -> str | None:
        """
        Scan the policy input directory for a supported policy document.
        Supported formats: .pdf, .docx, .md

        Returns:
            Absolute path to the first document found (sorted), or None.
        """
        os.makedirs(self._paths.policy_input_dir, exist_ok=True)
        docs = []
        for ext in ["*.pdf", "*.docx", "*.md"]:
            docs.extend(glob.glob(os.path.join(self._paths.policy_input_dir, ext)))
        
        docs.sort()
        return docs[0] if docs else None

    # ------------------------------------------------------------------
    # RAG config
    # ------------------------------------------------------------------

    def load_rag_config(self) -> dict | None:
        """
        Load RAG tuning parameters from rag_config.json.

        Returns:
            Config dict, or None if file missing or corrupt.
        """
        if not os.path.exists(self._paths.rag_config_file):
            return None
        try:
            with open(self._paths.rag_config_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"  ⚠ Could not load RAG config: {exc}")
            return None

    def save_rag_config(self, config: dict) -> None:
        """
        Persist RAG tuning parameters to rag_config.json.

        Args:
            config: Dict with chunk_size, chunk_overlap, k, rationale.
        """
        with open(self._paths.rag_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"  💾 RAG config saved to {self._paths.rag_config_file}")
