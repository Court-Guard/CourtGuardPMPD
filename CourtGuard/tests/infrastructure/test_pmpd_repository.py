"""
Tests for infrastructure/pmpd_repository.py

Covers:
  - save() / load() roundtrip
  - Atomic write (tmp file replaced)
  - Corrupt file returns empty PMPD
  - Missing file returns empty PMPD
  - GlobalModule serialisation / deserialisation
  - CategoryModule serialisation / deserialisation
  - Live shots preserved across save/load
"""

import os

import pytest

from core.enums import OneShotSource, Verdict
from core.models import CategoryModule, GlobalModule, OneShot
from core.pmpd import PMPD
from infrastructure.pmpd_repository import PMPDRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> PMPD:
    db = PMPD.from_scratch(source_doc="TestPolicy")
    db.set_global_module(
        GlobalModule(
            objective="Test objective.",
            principles=["Principle A.", "Principle B."],
            schema="Output verdict.",
            source_doc="TestPolicy",
        )
    )
    cat = CategoryModule(
        category_id="S1",
        name="Defamation",
        general_rules="No false statements.",
        exceptions=["Historical education."],
        one_shots=[
            OneShot(
                example_id="S1_shadow_001",
                query="q",
                response="r",
                verdict=Verdict.VIOLATION,
                reasoning="reason",
                source=OneShotSource.SHADOW,
                category_id="S1",
            )
        ],
        citations=["Section 3.1"],
        source_doc="TestPolicy",
    )
    db.add_category(cat)
    return db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    return str(tmp_path / "pmpd_store.json")


@pytest.fixture
def repo(tmp_db_path) -> PMPDRepository:
    return PMPDRepository(tmp_db_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPMPDRepositorySaveLoad:
    def test_save_creates_file(self, repo, tmp_db_path):
        repo.save(_make_db())
        assert os.path.exists(tmp_db_path)

    def test_load_missing_returns_empty(self, repo):
        db = repo.load()
        assert db.global_module is None
        assert db.categories == {}

    def test_roundtrip_global_module(self, repo):
        original = _make_db()
        repo.save(original)
        loaded = repo.load()

        assert loaded.global_module is not None
        assert loaded.global_module.objective == "Test objective."
        assert loaded.global_module.principles == ["Principle A.", "Principle B."]
        assert loaded.global_module.schema == "Output verdict."
        assert loaded.global_module.source_doc == "TestPolicy"

    def test_roundtrip_category_module(self, repo):
        original = _make_db()
        repo.save(original)
        loaded = repo.load()

        assert "S1" in loaded.categories
        cat = loaded.categories["S1"]
        assert cat.name == "Defamation"
        assert cat.general_rules == "No false statements."
        assert cat.exceptions == ["Historical education."]
        assert cat.citations == ["Section 3.1"]

    def test_roundtrip_shadow_shots(self, repo):
        original = _make_db()
        repo.save(original)
        loaded = repo.load()

        shots = loaded.categories["S1"].one_shots
        assert len(shots) == 1
        assert shots[0].source == OneShotSource.SHADOW
        assert shots[0].verdict == Verdict.VIOLATION
        assert shots[0].example_id == "S1_shadow_001"

    def test_roundtrip_live_shots(self, repo):
        db = _make_db()
        db.record_verdict(
            category_id="S1",
            query="live query",
            response="live response",
            verdict=Verdict.BORDERLINE,
            reasoning="borderline",
            severity=3,
        )
        repo.save(db)
        loaded = repo.load()

        live = loaded.categories["S1"].get_live_shots(limit=10)
        assert len(live) == 1
        assert live[0].verdict == Verdict.BORDERLINE
        assert live[0].severity == 3

    def test_roundtrip_source_doc(self, repo):
        original = _make_db()
        repo.save(original)
        loaded = repo.load()
        assert loaded.source_doc == "TestPolicy"

    def test_corrupt_file_returns_empty(self, repo, tmp_db_path):
        with open(tmp_db_path, "w") as f:
            f.write("this is not valid json {{{")
        db = repo.load()
        assert db.global_module is None
        assert db.categories == {}

    def test_atomic_write_no_tmp_file_left(self, repo, tmp_db_path):
        repo.save(_make_db())
        tmp_file = tmp_db_path + ".tmp"
        assert not os.path.exists(tmp_file)

    def test_multiple_categories_roundtrip(self, repo):
        db = _make_db()
        db.add_category(
            CategoryModule(
                category_id="S2",
                name="Privacy",
                general_rules="Protect user data.",
                exceptions=[],
                one_shots=[],
                citations=[],
            )
        )
        repo.save(db)
        loaded = repo.load()
        assert set(loaded.categories.keys()) == {"S1", "S2"}


class TestCreateEmpty:
    def test_creates_empty_pmpd(self):
        db = PMPDRepository.create_empty(source_doc="Test")
        assert db.global_module is None
        assert db.categories == {}
        assert db.source_doc == "Test"
