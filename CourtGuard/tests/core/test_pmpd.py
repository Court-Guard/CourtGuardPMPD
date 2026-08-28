"""
Tests for core/pmpd.py

Covers:
  - PMPD module registration and lookup
  - add_category() preserves live shots on overwrite
  - record_verdict() feedback loop
  - Prompt fragment assembly helpers
  - stats()
  - from_scratch() factory
"""

import pytest

from core.enums import OneShotSource, Verdict
from core.models import CategoryModule, GlobalModule, OneShot
from core.pmpd import PMPD

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_category(cid: str, name: str) -> CategoryModule:
    return CategoryModule(
        category_id=cid,
        name=name,
        general_rules=f"Rules for {name}.",
        exceptions=[f"Exception for {name}."],
        one_shots=[
            OneShot(
                example_id=f"{cid}_shadow_001",
                query="test query",
                response="test response",
                verdict=Verdict.VIOLATION,
                reasoning="test reasoning",
                source=OneShotSource.SHADOW,
                category_id=cid,
            )
        ],
        citations=[f"Ref {cid}"],
        source_doc="test_doc",
    )


def _make_global() -> GlobalModule:
    return GlobalModule(
        objective="Test objective.",
        principles=["Principle A.", "Principle B."],
        schema="Output verdict and severity.",
        source_doc="test_doc",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> PMPD:
    pmpd = PMPD.from_scratch(source_doc="test_doc")
    pmpd.set_global_module(_make_global())
    pmpd.add_category(_make_category("S1", "Defamation"))
    pmpd.add_category(_make_category("S2", "Privacy"))
    return pmpd


# ---------------------------------------------------------------------------
# Construction and metadata
# ---------------------------------------------------------------------------


class TestPMPDConstruction:
    def test_from_scratch_empty(self):
        pmpd = PMPD.from_scratch()
        assert pmpd.global_module is None
        assert pmpd.categories == {}

    def test_from_scratch_sets_source_doc(self):
        pmpd = PMPD.from_scratch(source_doc="MyPolicy")
        assert pmpd.source_doc == "MyPolicy"

    def test_set_source_doc(self):
        pmpd = PMPD.from_scratch()
        pmpd.set_source_doc("NewDoc")
        assert pmpd.source_doc == "NewDoc"


# ---------------------------------------------------------------------------
# Module management
# ---------------------------------------------------------------------------


class TestModuleManagement:
    def test_set_global_module(self, db):
        assert db.global_module is not None
        assert db.global_module.objective == "Test objective."

    def test_add_category_stores_module(self, db):
        assert "S1" in db.categories
        assert "S2" in db.categories

    def test_get_category_returns_module(self, db):
        cat = db.get_category("S1")
        assert cat is not None
        assert cat.name == "Defamation"

    def test_get_category_returns_none_for_unknown(self, db):
        assert db.get_category("S99") is None

    def test_list_categories_sorted(self, db):
        assert db.list_categories() == ["S1", "S2"]

    def test_add_category_preserves_live_shots_on_overwrite(self, db):
        """Replacing a category keeps accumulated live one-shots."""
        # First add a live shot to S1
        db.record_verdict(
            category_id="S1",
            query="live query",
            response="live response",
            verdict=Verdict.VIOLATION,
            reasoning="live reasoning",
            severity=4,
        )
        live_count_before = len(db.get_category("S1").get_live_shots(limit=999))
        assert live_count_before == 1

        # Now overwrite S1 with a new module (as the parser would on re-run)
        new_cat = _make_category("S1", "Defamation Updated")
        db.add_category(new_cat)

        # Live shots preserved
        cat = db.get_category("S1")
        assert cat.name == "Defamation Updated"
        assert len(cat.get_live_shots(limit=999)) == 1


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------


class TestRecordVerdict:
    def test_record_verdict_adds_live_shot(self, db):
        shot = db.record_verdict(
            category_id="S1",
            query="test query",
            response="test response",
            verdict=Verdict.BORDERLINE,
            reasoning="borderline case",
            severity=3,
        )
        assert shot is not None
        assert shot.source == OneShotSource.LIVE
        assert shot.verdict == Verdict.BORDERLINE
        assert shot.severity == 3
        assert shot.category_id == "S1"

    def test_record_verdict_sequential_ids(self, db):
        shot1 = db.record_verdict("S1", "q1", "r1", Verdict.VIOLATION, "r1")
        shot2 = db.record_verdict("S1", "q2", "r2", Verdict.COMPLIANT, "r2")
        assert shot1.example_id == "S1_live_001"
        assert shot2.example_id == "S1_live_002"

    def test_record_verdict_unknown_category_returns_none(self, db):
        result = db.record_verdict(
            category_id="S99",
            query="q",
            response="r",
            verdict=Verdict.VIOLATION,
            reasoning="r",
        )
        assert result is None

    def test_record_verdict_appended_to_category(self, db):
        db.record_verdict("S2", "q", "r", Verdict.VIOLATION, "r")
        cat = db.get_category("S2")
        live = cat.get_live_shots(limit=999)
        assert len(live) == 1


# ---------------------------------------------------------------------------
# Prompt assembly helpers
# ---------------------------------------------------------------------------


class TestPromptHelpers:
    def test_get_global_fragment_returns_text(self, db):
        fragment = db.get_global_fragment()
        assert "Test objective." in fragment
        assert "Principle A." in fragment

    def test_get_global_fragment_empty_when_no_global(self, db):
        pmpd = PMPD.from_scratch()
        assert db.get_global_fragment() != ""
        assert pmpd.get_global_fragment() == ""

    def test_get_category_fragment_returns_text(self, db):
        fragment = db.get_category_fragment("S1")
        assert "Defamation" in fragment
        assert "Rules for Defamation" in fragment

    def test_get_category_fragment_unknown_returns_error_string(self, db):
        fragment = db.get_category_fragment("S99")
        assert "not found" in fragment

    def test_get_all_category_fragments_includes_all(self, db):
        all_frags = db.get_all_category_fragments()
        assert "Defamation" in all_frags
        assert "Privacy" in all_frags

    def test_get_all_category_fragments_subset(self, db):
        frag = db.get_all_category_fragments(category_ids=["S1"])
        assert "Defamation" in frag
        assert "Privacy" not in frag


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_counts_correct(self, db):
        s = db.stats()
        assert s["categories"] == 2
        assert s["has_global_module"] is True
        assert s["total_shadow_shots"] == 2  # 1 per category
        assert s["total_live_shots"] == 0

    def test_stats_live_shots_counted_after_verdict(self, db):
        db.record_verdict("S1", "q", "r", Verdict.VIOLATION, "r")
        s = db.stats()
        assert s["total_live_shots"] == 1
        assert s["total_one_shots"] == 3  # 2 shadow + 1 live
