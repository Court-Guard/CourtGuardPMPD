"""
PMPD Repository — Persistence Layer

Handles all JSON serialisation and deserialisation for the PMPD database.

Extracted from the original pmpd.py PMPD class to satisfy SRP:
  core/pmpd.py                     ← registry + feedback loop (no I/O)
  infrastructure/pmpd_repository.py ← save / load / serialise  (this file)

The repository pattern means:
  • PMPD is fully unit-testable without touching the filesystem.
  • Swapping the storage format (e.g. MessagePack, SQLite) only requires
    a new PMPDRepository implementation — PMPD itself is unchanged.
  • Atomic writes (tmp → os.replace) are isolated here, not scattered
    across the domain model.

Usage
─────
    repo = PMPDRepository("pmpd_store.json")

    # Load existing or create fresh
    db = repo.load()

    # ... modify db ...

    repo.save(db)

    # Create a fresh DB without loading from disk
    db = PMPDRepository.create_empty(source_doc="AILuminate v1.1")
"""

from __future__ import annotations

import json
import os
import time

from core.enums import OneShotSource
from core.models import CategoryModule, GlobalModule, OneShot
from core.pmpd import PMPD

# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class PMPDRepository:
    """
    Persists and restores PMPD instances as JSON files.

    Responsibilities
    ────────────────
    • save(db)  — serialise PMPD to disk atomically
    • load()    — deserialise PMPD from disk; returns empty PMPD on missing file
    • Serialisation helpers (private) — CategoryModule ↔ dict

    The db_path is fixed at construction time so the repository is a
    stable handle to one specific store file.
    """

    def __init__(self, db_path: str = "pmpd_store.json") -> None:
        """
        Args:
            db_path: Filesystem path to the PMPD JSON store.
        """
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, db: PMPD) -> None:
        """
        Serialise the full PMPD to disk as JSON.

        Uses an atomic write (tmp file → os.replace) so a crash mid-write
        never leaves a corrupt store file.

        Args:
            db: The PMPD instance to persist.
        """
        data = {
            "_meta": db._meta,
            "global_module": self._serialise_global(db.global_module),
            "categories": {
                cid: self._serialise_category(cat) for cid, cat in db.categories.items()
            },
        }

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        tmp_path = self.db_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        os.replace(tmp_path, self.db_path)  # atomic write
        print(f"  💾 PMPD saved → {self.db_path}")

    def load(self) -> PMPD:
        """
        Deserialise a PMPD from disk.

        If the file does not exist, returns a fresh empty PMPD.
        If the file exists but is corrupt, prints a warning and returns
        a fresh empty PMPD so the pipeline can continue.

        Returns:
            Populated PMPD instance (or empty if file missing / corrupt).
        """
        if not os.path.exists(self.db_path):
            return PMPD.from_scratch()

        try:
            with open(self.db_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ⚠ Could not load PMPD from {self.db_path}: {exc}")
            return PMPD.from_scratch()

        return self._deserialise(data)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_empty(source_doc: str = "") -> PMPD:
        """
        Create a fresh PMPD instance without touching the filesystem.

        Convenience wrapper around PMPD.from_scratch() — useful when the
        bootstrap pipeline needs a blank slate before the parser runs.

        Args:
            source_doc: Name of the governance document being parsed.

        Returns:
            Empty PMPD instance.
        """
        return PMPD.from_scratch(source_doc=source_doc)

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------

    def _deserialise(self, data: dict) -> PMPD:
        """Reconstruct a PMPD from a raw JSON dict."""
        db = PMPD.from_scratch()
        db._meta = data.get("_meta", db._meta)

        gm_data = data.get("global_module")
        if gm_data:
            db.global_module = self._deserialise_global(gm_data)

        for cid, cdata in data.get("categories", {}).items():
            db.categories[cid] = self._deserialise_category(cid, cdata)

        print(f"  📂 PMPD loaded from {self.db_path} " f"({len(db.categories)} categories)")
        return db

    @staticmethod
    def _deserialise_global(data: dict) -> GlobalModule:
        """Reconstruct a GlobalModule from a dict."""
        return GlobalModule(
            objective=               data["objective"],
            principles=              data["principles"],
            schema=                  data["schema"],
            source_doc=              data.get("source_doc", ""),
            last_updated=            data.get("last_updated", time.time()),
            definitions=             data.get("definitions", {}),
            general_eval_principles= data.get("general_eval_principles", []),
            evaluation_protocol=     data.get("evaluation_protocol", []),
            positive_examples=       data.get("positive_examples", []),
            negative_examples=       data.get("negative_examples", []),
        )

    @staticmethod
    def _deserialise_category(cid: str, data: dict) -> CategoryModule:
        """Reconstruct a CategoryModule (and its OneShots) from a dict."""
        one_shots = [
            OneShot(
                example_id=s["example_id"],
                query=s["query"],
                response=s["response"],
                verdict=s["verdict"],
                reasoning=s["reasoning"],
                severity=s.get("severity", 0),
                source=s.get("source", OneShotSource.SHADOW),
                category_id=s.get("category_id", cid),
                timestamp=s.get("timestamp", 0.0),
                polarity=    s.get("polarity",  "positive"),
            )
            for s in data.get("one_shots", [])
        ]
        return CategoryModule(
            category_id=      cid,
            name=             data["name"],
            general_rules=    data["general_rules"],
            exceptions=       data.get("exceptions", []),
            one_shots=        one_shots,
            citations=        data.get("citations", []),
            source_doc=       data.get("source_doc", ""),
            last_updated=     data.get("last_updated", time.time()),
            short_definition= data.get("short_definition", ""),
            sub_categories=   data.get("sub_categories", []),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_global(module: GlobalModule | None) -> dict | None:
        """Convert a GlobalModule to a plain dict, or None."""
        if module is None:
            return None
        return {
            "objective":               module.objective,
            "principles":              module.principles,
            "schema":                  module.schema,
            "source_doc":              module.source_doc,
            "last_updated":            module.last_updated,
            "definitions":             module.definitions,
            "general_eval_principles": module.general_eval_principles,
            "evaluation_protocol":     module.evaluation_protocol,
            "positive_examples":       module.positive_examples,
            "negative_examples":       module.negative_examples,
        }

    @staticmethod
    def _serialise_category(cat: CategoryModule) -> dict:
        """Convert a CategoryModule to a plain dict for JSON serialisation."""
        return {
            "name": cat.name,
            "general_rules": cat.general_rules,
            "exceptions": cat.exceptions,
            "citations": cat.citations,
            "source_doc": cat.source_doc,
            "last_updated": cat.last_updated,
            "short_definition": cat.short_definition,
            "sub_categories":   cat.sub_categories,
            "one_shots": [
                {
                    "example_id": s.example_id,
                    "query": s.query,
                    "response": s.response,
                    "verdict": s.verdict,
                    "reasoning": s.reasoning,
                    "severity": s.severity,
                    "source": s.source,
                    "category_id": s.category_id,
                    "timestamp": s.timestamp,
                    "polarity": s.polarity,

                }
                for s in cat.one_shots
            ],
        }
