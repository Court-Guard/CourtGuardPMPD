"""
Optional tests for Phase 2 _record_verdict() feedback loop.

Verifies that app._record_verdict() stores the configured taxonomy label
directly in the OneShot verdict field instead of converting it to legacy
'violation'/'compliant'/'borderline' strings.

All tests are marked with pytest.mark.optional and are skipped in default
CI runs. No API calls, no file I/O -- uses in-memory PMPD objects.

Run with:
    pytest tests/evaluation/test_feedback_loop.py -v
or include optional tests with:
    pytest -m optional
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pmpd_with_category():
    """Build a minimal in-memory PMPD with one category for testing."""
    from core.models import CategoryModule, GlobalModule
    from core.pmpd import PMPD

    cat = CategoryModule(
        category_id="S1",
        name="Test Category",
        short_definition="Test",
        general_rules="No violations.",
        exceptions=[],
        one_shots=[],
        citations=[],
    )
    glb = GlobalModule(
        objective="Policy compliance",
        principles=[],
        schema="verdict: BLOCK | ALLOW",
        source_doc="test.md",
    )
    db = PMPD()
    db.set_global_module(glb)
    db.add_category(cat)
    return db


def _make_assembled_prompt():
    """Build a minimal AssembledPrompt for testing _record_verdict."""
    from pmpd_pipeline.assembler import AssembledPrompt

    return AssembledPrompt(
        attacker_system="Attacker system prompt",
        defender_system="Defender system prompt",
        judge_system="Judge system prompt",
        context="Policy context",
        category_ids=["S1"],
    )


def _make_output_mapper(labels):
    """Return an OutputMapper with the given labels."""
    from evaluation.output_mapper import OutputMapper
    return OutputMapper(labels)


def _make_paths():
    """Return a PathConfig with a temp pmpd_db_path."""
    from infrastructure.config import PathConfig
    return PathConfig(pmpd_db_path="/tmp/test_pmpd_store.json")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.optional
def test_configured_label_stored_directly():
    """
    When the judge returns label='BLOCK', verdict is stored as 'BLOCK',
    not 'violation' (the pre-Phase-2 legacy conversion).
    """
    from core.pmpd import PMPD
    from evaluation.output_mapper import OutputMapper

    db = _make_pmpd_with_category()
    mapper = OutputMapper(("ALLOW", "BLOCK"))
    payload = _make_assembled_prompt()

    result = {
        "judge_evaluation": {
            "label": "BLOCK",
            "detailed_analysis": "This is clearly a block case.",
            "violated_categories": [],
        }
    }

    # Capture what was passed to db.record_verdict
    recorded_verdict = []

    original_record = db.record_verdict

    def capture_record(category_id, query, response, verdict, reasoning, severity):
        recorded_verdict.append(verdict)
        return original_record(
            category_id=category_id,
            query=query,
            response=response,
            verdict=verdict,
            reasoning=reasoning,
            severity=severity,
        )

    db.record_verdict = capture_record

    # We call the static method directly — no need for a full CourtGuardApp instance
    from app import CourtGuardApp

    with patch("app.PMPDRepository") as mock_repo:
        mock_repo.return_value.save = MagicMock()
        app = CourtGuardApp.__new__(CourtGuardApp)
        app._paths = _make_paths()
        app._record_verdict(
            db=db,
            query="test query",
            response="test response",
            result=result,
            payload=payload,
            output_mapper=mapper,
        )

    assert recorded_verdict, "Expected at least one verdict to be recorded"
    assert recorded_verdict[0] == "BLOCK", (
        f"Expected stored verdict 'BLOCK', got '{recorded_verdict[0]}'"
    )


@pytest.mark.optional
def test_default_label_stored_as_configured():
    """
    When the judge returns label='ALLOW' (the default/compliant label),
    it is stored as 'ALLOW', not 'compliant'.
    """
    from evaluation.output_mapper import OutputMapper

    db = _make_pmpd_with_category()
    mapper = OutputMapper(("ALLOW", "BLOCK"))
    payload = _make_assembled_prompt()

    result = {
        "judge_evaluation": {
            "label": "ALLOW",
            "detailed_analysis": "This response is fine.",
            "violated_categories": [],
        }
    }

    recorded_verdict = []
    original_record = db.record_verdict

    def capture_record(category_id, query, response, verdict, reasoning, severity):
        recorded_verdict.append(verdict)
        return original_record(
            category_id=category_id, query=query, response=response,
            verdict=verdict, reasoning=reasoning, severity=severity,
        )

    db.record_verdict = capture_record

    from app import CourtGuardApp

    with patch("app.PMPDRepository") as mock_repo:
        mock_repo.return_value.save = MagicMock()
        app = CourtGuardApp.__new__(CourtGuardApp)
        app._paths = _make_paths()
        app._record_verdict(
            db=db, query="q", response="r",
            result=result, payload=payload, output_mapper=mapper,
        )

    assert recorded_verdict and recorded_verdict[0] == "ALLOW", (
        f"Expected 'ALLOW', got '{recorded_verdict[0] if recorded_verdict else None}'"
    )


@pytest.mark.optional
def test_severity_stored_as_zero():
    """Phase 2: severity is always stored as 0 (legacy 1-5 scale is not used)."""
    from evaluation.output_mapper import OutputMapper

    db = _make_pmpd_with_category()
    mapper = OutputMapper(("ALLOW", "BLOCK"))
    payload = _make_assembled_prompt()

    result = {
        "judge_evaluation": {
            "label": "BLOCK",
            "detailed_analysis": "Flagged.",
            "violated_categories": [],
        }
    }

    recorded_severity = []
    original_record = db.record_verdict

    def capture_record(category_id, query, response, verdict, reasoning, severity):
        recorded_severity.append(severity)
        return original_record(
            category_id=category_id, query=query, response=response,
            verdict=verdict, reasoning=reasoning, severity=severity,
        )

    db.record_verdict = capture_record

    from app import CourtGuardApp

    with patch("app.PMPDRepository") as mock_repo:
        mock_repo.return_value.save = MagicMock()
        app = CourtGuardApp.__new__(CourtGuardApp)
        app._paths = _make_paths()
        app._record_verdict(
            db=db, query="q", response="r",
            result=result, payload=payload, output_mapper=mapper,
        )

    assert recorded_severity and recorded_severity[0] == 0, (
        f"Expected severity 0, got {recorded_severity[0] if recorded_severity else 'nothing'}"
    )


@pytest.mark.optional
def test_api_error_skips_recording():
    """When result has api_error=True, no verdict is recorded."""
    from evaluation.output_mapper import OutputMapper

    db = _make_pmpd_with_category()
    mapper = OutputMapper(("ALLOW", "BLOCK"))
    payload = _make_assembled_prompt()

    result = {
        "api_error": True,
        "judge_evaluation": {"label": "BLOCK"},
    }

    recorded_verdict = []
    original_record = db.record_verdict

    def capture_record(**kwargs):
        recorded_verdict.append(kwargs.get("verdict"))
        return original_record(**kwargs)

    db.record_verdict = capture_record

    from app import CourtGuardApp

    app = CourtGuardApp.__new__(CourtGuardApp)
    app._paths = _make_paths()
    app._record_verdict(
        db=db, query="q", response="r",
        result=result, payload=payload, output_mapper=mapper,
    )

    assert not recorded_verdict, "No verdict should be recorded when api_error=True"
