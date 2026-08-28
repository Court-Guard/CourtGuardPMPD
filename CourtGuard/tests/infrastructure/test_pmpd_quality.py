"""Artifact quality tests for the local PMPD store."""

from pathlib import Path

import pytest

from infrastructure.pmpd_repository import PMPDRepository


PMPD_PATH = Path("F:/Projects/SDP-II/CourtGuard/pmpd_store.json")


@pytest.mark.skipif(not PMPD_PATH.exists(), reason="local pmpd_store.json not present")
def test_all_categories_have_nonempty_rules():
    db = PMPDRepository(str(PMPD_PATH)).load()

    for cid, cat in db.categories.items():
        assert cat.name.strip(), f"{cid} missing name"
        assert cat.general_rules.strip(), f"{cid} missing general_rules"


@pytest.mark.skipif(not PMPD_PATH.exists(), reason="local pmpd_store.json not present")
def test_no_placeholder_general_rules():
    db = PMPDRepository(str(PMPD_PATH)).load()

    bad_fragments = [
        "See source document for details",
        "(not set)",
    ]

    offenders = []
    for cid, cat in db.categories.items():
        text = f"{cat.short_definition} {cat.general_rules}"
        if any(fragment in text for fragment in bad_fragments):
            offenders.append(cid)

    assert offenders == [], f"Placeholder PMPD content found in: {offenders}"

