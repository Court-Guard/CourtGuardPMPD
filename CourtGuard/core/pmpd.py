"""
PMPD — Prompt-based Modular Policy Database

Defines the central registry and feedback loop for the PMPD pipeline.

Architecture
------------
                ┌─────────────────────────────────┐
                │           PMPD Database          │
                │                                  │
                │  ┌──────────────────────────┐   │
                │  │      Global Module        │   │
                │  │  • Objective              │   │
                │  │  • Principles             │   │
                │  │  • Schema                 │   │
                │  └──────────────────────────┘   │
                │                                  │
                │  ┌──────────────────────────┐   │
                │  │   Category Module (×N)    │   │
                │  │  • ID  (e.g. S1)          │   │
                │  │  • General Rules          │   │
                │  │  • Exceptions             │   │
                │  │  • One-Shots              │   │  ◄── Shadow + Live
                │  │  • Citations              │   │
                │  └──────────────────────────┘   │
                └─────────────────────────────────┘
                         ▲             │
                         │             ▼
                  New verdicts     Prompt assembly
                  fed back as      (pmpd_assembler.py)
                  One-Shots

SRP split from original pmpd.py
─────────────────────────────────
  core/models.py              ← OneShot, GlobalModule, CategoryModule
  core/pmpd.py                ← PMPD registry + feedback loop  (this file)
  infrastructure/pmpd_repository.py ← JSON persistence (save / load)

The PMPD class no longer touches the filesystem directly.
Persistence is handled by PMPDRepository, which is injected at
construction time — making PMPD fully unit-testable without disk I/O.
"""

from __future__ import annotations

import time

from core.enums import OneShotSource
from core.models import CategoryModule, GlobalModule, OneShot

# ---------------------------------------------------------------------------
# PMPD
# ---------------------------------------------------------------------------


class PMPD:
    """
    Prompt-based Modular Policy Database.

    Central store for all Global and Category modules derived from
    Governance Corpora.  Delegates persistence to an injected
    PMPDRepository — the PMPD itself contains no file I/O.

    Usage
    -----
        # With repository (production):
        from infrastructure.pmpd_repository import PMPDRepository
        repo = PMPDRepository("pmpd_store.json")
        db   = repo.load()            # load existing or create fresh
        db.set_global_module(global_mod)
        db.add_category(cat_mod)
        repo.save(db)

        # After a Judge verdict:
        db.record_verdict(
            category_id="S1",
            query="...", response="...",
            verdict="violation", reasoning="...", severity=4
        )
        repo.save(db)
    """

    DB_VERSION = "1.0"

    def __init__(self, source_doc: str = "") -> None:
        """
        Create an empty PMPD.

        Args:
            source_doc: Name / path of the governance document this DB
                        was built from. Stored in metadata.
        """
        self.global_module: GlobalModule | None = None
        self.categories: dict[str, CategoryModule] = {}
        self._meta: dict = {
            "version": self.DB_VERSION,
            "created_at": time.time(),
            "source_doc": source_doc,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_source_doc(self, source_doc: str) -> None:
        """
        Set the source document name in metadata.

        Replaces the external _meta dict access pattern in pmpd_parser.py:
            db._meta["source_doc"] = source_doc  ← old (private access)
            db.set_source_doc(source_doc)          ← new (public setter)
        """
        self._meta["source_doc"] = source_doc

    @property
    def source_doc(self) -> str:
        """The governance document this PMPD was built from."""
        return self._meta.get("source_doc", "")

    # ------------------------------------------------------------------
    # Module management
    # ------------------------------------------------------------------

    def set_global_module(self, module: GlobalModule) -> None:
        """Set or replace the GlobalModule."""
        self.global_module = module

    def add_category(self, module: CategoryModule) -> None:
        """
        Add or overwrite a CategoryModule.
        If a module with the same ID already exists, its accumulated
        live One-Shots are preserved (the rules portion is replaced).
        """
        if module.category_id in self.categories:
            existing_live = self.categories[module.category_id].get_live_shots(limit=9999)
            module.one_shots = module.get_shadow_shots() + existing_live
        self.categories[module.category_id] = module

    def get_category(self, category_id: str) -> CategoryModule | None:
        """Return the CategoryModule for category_id, or None."""
        return self.categories.get(category_id)

    def list_categories(self) -> list[str]:
        """Sorted list of all category IDs."""
        return sorted(self.categories.keys())

    # ------------------------------------------------------------------
    # Feedback loop — dynamic One-Shot injection
    # ------------------------------------------------------------------

    def record_verdict(
        self,
        category_id: str,
        query: str,
        response: str,
        verdict: str,
        reasoning: str,
        severity: int = 0,
    ) -> OneShot | None:
        """
        Feed a Judge Agent verdict back into the PMPD as a live One-Shot.

        This is the dynamic updating mechanism described in Phase 4:
        new borderline cases are parsed and stored so future evaluations
        benefit from accumulated real-world examples.

        Args
        ----
        category_id:  Which category this verdict belongs to
        query:        The user query that was evaluated
        response:     The AI response that was evaluated
        verdict:      "violation" | "compliant" | "borderline"
        reasoning:    Judge Agent's reasoning
        severity:     1–5 score from Judge Agent

        Returns
        -------
        The created OneShot, or None if category_id not found.
        """
        cat = self.categories.get(category_id)
        if cat is None:
            print(f"  ⚠ PMPD.record_verdict: unknown category " f"'{category_id}' — skipped")
            return None

        live_count = len(cat.get_live_shots(limit=9999))
        shot_id = f"{category_id}_live_{live_count + 1:03d}"

        shot = OneShot(
            example_id=shot_id,
            query=query,
            response=response,
            verdict=verdict,
            reasoning=reasoning,
            severity=severity,
            source=OneShotSource.LIVE,
            category_id=category_id,
        )
        cat.add_one_shot(shot)
        return shot

    # ------------------------------------------------------------------
    # Prompt assembly helpers
    # ------------------------------------------------------------------

    def get_global_fragment(self) -> str:
        """Return the global module as a prompt fragment, or empty string."""
        if self.global_module:
            return self.global_module.to_prompt_fragment()
        return ""

    def get_category_fragment(
        self,
        category_id: str,
        include_shots: bool = True,
    ) -> str:
        """Return a single category module as a prompt fragment."""
        cat = self.categories.get(category_id)
        if cat is None:
            return f"[Category '{category_id}' not found in PMPD]"
        return cat.to_prompt_fragment(include_one_shots=include_shots)

    def get_all_category_fragments(
        self,
        include_shots: bool = True,
        category_ids: list[str] | None = None,
    ) -> str:
        """
        Return multiple category modules concatenated as a single block.
        Pass `category_ids` to restrict to a subset (e.g. RAG-retrieved ones).
        """
        ids = category_ids or self.list_categories()
        parts = [self.get_category_fragment(cid, include_shots) for cid in ids]
        return "\n\n" + ("=" * 60) + "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Summary statistics about the current PMPD state."""
        total_shadow = sum(len(c.get_shadow_shots()) for c in self.categories.values())
        total_live = sum(len(c.get_live_shots(limit=9999)) for c in self.categories.values())
        return {
            "categories": len(self.categories),
            "has_global_module": self.global_module is not None,
            "total_shadow_shots": total_shadow,
            "total_live_shots": total_live,
            "total_one_shots": total_shadow + total_live,
            "source_doc": self._meta.get("source_doc", ""),
        }

    def print_stats(self) -> None:
        """Print a formatted statistics summary to stdout."""
        s = self.stats()
        print(f"\n{'─'*50}")
        print("  PMPD Statistics")
        print(f"{'─'*50}")
        print(f"  Source document  : {s['source_doc']}")
        print(f"  Global module    : {'✅' if s['has_global_module'] else '❌ missing'}")
        print(f"  Categories       : {s['categories']}")
        print(f"  Shadow One-Shots : {s['total_shadow_shots']}")
        print(f"  Live One-Shots   : {s['total_live_shots']}")
        print(f"  Total One-Shots  : {s['total_one_shots']}")
        print(f"{'─'*50}\n")

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_scratch(cls, source_doc: str = "") -> PMPD:
        """
        Create a fresh PMPD — useful before running the parser.

        Args:
            source_doc: Name of the governance document being parsed.
        """
        return cls(source_doc=source_doc)
