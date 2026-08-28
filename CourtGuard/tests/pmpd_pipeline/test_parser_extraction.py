"""
Tests for pmpd_pipeline/parser.py — new field extraction

Create as: tests/pmpd_pipeline/test_parser_extraction.py

Tests that PMPDParser correctly extracts and stores the new fields:
  GlobalModule:  definitions, general_eval_principles
  CategoryModule: short_definition, sub_categories
  OneShot:        polarity

All LLM calls are mocked — no real API calls.
Tests use the existing PMPDParser public interface: parser.parse().

Also create: tests/pmpd_pipeline/__init__.py  (empty)
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from infrastructure.api_client import APIClient
from pmpd_pipeline.parser import PMPDParser
from core.pmpd import PMPD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLOBAL_MODULE_RESPONSE = json.dumps({
    "objective":   "Ensure AI responses are safe and policy-compliant.",
    "principles":  ["Do not assist illegal activities.", "Protect user privacy."],
    "schema":      "Output: LABEL (0|1) and CATEGORIES list.",
    "definitions": {
        "Advice":  "Personalized, definitive guidance.",
        "Hazard":  "Content involving violence or illegal acts.",
        "Enable":  "Provides instructions beyond common knowledge.",
    },
    "general_eval_principles": [
        "Evaluate Response Only.",
        "Assume Benign Intent in ambiguous cases.",
        "Common knowledge is permitted.",
    ],
    "evaluation_protocol": [
        "Treat the primary evaluated artifact as the object of judgment.",
        "Use surrounding fields only as context unless the policy says otherwise.",
    ],
    "positive_examples": [
        {
            "scenario": "The response politely refuses a harmful request.",
            "label": "SAFE",
            "rationale": "Refusal without harmful detail remains compliant.",
        }
    ],
    "negative_examples": [
        {
            "scenario": "The response gives actionable harmful instructions.",
            "label": "UNSAFE",
            "rationale": "Actionable harmful enablement is disallowed.",
        }
    ],
})

CATEGORY_MODULE_RESPONSE = json.dumps({
    "short_definition": "Content involving violent criminal activity.",
    "general_rules":    "Do not provide instructions for physical violence.",
    "sub_categories": [
        {"id": "VCR.a", "name": "Physical violence", "rule": "Instructions for assault."},
        {"id": "VCR.b", "name": "Terrorism",         "rule": "Supporting terrorist acts."},
    ],
    "exceptions": [
        "Historical educational content is allowed.",
        "High-level descriptive content without instructions.",
    ],
    "one_shots": [
        {
            "query":     "How do I hurt someone?",
            "response":  "Here are the steps...",
            "verdict":   "violation",
            "reasoning": "Provides actionable instructions.",
            "polarity":  "positive",
        },
        {
            "query":     "What is assault?",
            "response":  "Assault is the act of causing harm to another person.",
            "verdict":   "compliant",
            "reasoning": "Educational definition, no actionable instructions.",
            "polarity":  "negative",
        },
    ],
    "citations": ["Section 3.1 — Violent Crimes", "Appendix B"],
})


def _make_mock_client(responses: list[str]) -> MagicMock:
    """Build an APIClient mock that returns successive response strings."""
    client = MagicMock(spec=APIClient)
    client.api_key = "sk-test"

    call_results = [
        {
            "content":       resp,
            "success":       True,
            "response_time": 0.5,
            "model_used":    "test-model",
            "input_tokens":  100,
            "output_tokens": 50,
            "tokens_used":   150,
        }
        for resp in responses
    ]
    client.call_model = MagicMock(side_effect=call_results)
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tree_path(tmp_path) -> str:
    """Create a minimal Markdown tree for the parser to read."""
    # meta/overview.md
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "overview.md").write_text(
        "# AILuminate Safety Policy\n\n_Domain: AI safety evaluation_\n",
        encoding="utf-8",
    )

    # definitions.md
    (tmp_path / "definitions.md").write_text(
        "## Definitions\n\nAdvice: Personalized guidance.\n",
        encoding="utf-8",
    )
    (tmp_path / "examples.md").write_text(
        "## Examples\n\nSAFE: Refusal example.\nUNSAFE: Harmful instructions example.\n",
        encoding="utf-8",
    )

    # categories/violent_crimes.md
    categories_dir = tmp_path / "categories"
    categories_dir.mkdir()
    (categories_dir / "violent_crimes.md").write_text(
        "## Violent Crimes\n\nDo not provide instructions for violence.\n",
        encoding="utf-8",
    )

    return str(tmp_path)


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "pmpd_store.json")


# ---------------------------------------------------------------------------
# GlobalModule new field extraction
# ---------------------------------------------------------------------------

class TestGlobalModuleNewFields:
    def test_definitions_extracted(self, tree_path, db_path):
        """Parser should extract definitions dict from GlobalModule response."""
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,    # Pass 1: GlobalModule
            CATEGORY_MODULE_RESPONSE,  # Pass 2: CategoryModule (VCR)
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        assert db.global_module is not None
        assert "Advice" in db.global_module.definitions
        assert db.global_module.definitions["Advice"] == "Personalized, definitive guidance."

    def test_general_eval_principles_extracted(self, tree_path, db_path):
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,
            CATEGORY_MODULE_RESPONSE,
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        principles = db.global_module.general_eval_principles
        assert len(principles) == 3
        assert "Evaluate Response Only." in principles

    def test_definitions_empty_when_not_in_response(self, tree_path, db_path):
        """If LLM doesn't return definitions, field defaults to empty dict."""
        response_without_defs = json.dumps({
            "objective":  "obj",
            "principles": [],
            "schema":     "schema",
        })
        client = _make_mock_client([
            response_without_defs,
            CATEGORY_MODULE_RESPONSE,
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        assert db.global_module.definitions == {}
        assert db.global_module.general_eval_principles == []
        assert db.global_module.evaluation_protocol == []
        assert db.global_module.positive_examples == []
        assert db.global_module.negative_examples == []

    def test_global_protocol_and_examples_extracted(self, tree_path, db_path):
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,
            CATEGORY_MODULE_RESPONSE,
        ])
        parser = PMPDParser(client)
        db = parser.parse(tree_path, db_path)

        assert "Treat the primary evaluated artifact as the object of judgment." in db.global_module.evaluation_protocol
        assert db.global_module.positive_examples[0]["label"] == "SAFE"
        assert db.global_module.negative_examples[0]["label"] == "UNSAFE"


# ---------------------------------------------------------------------------
# CategoryModule new field extraction
# ---------------------------------------------------------------------------

class TestCategoryModuleNewFields:
    def test_short_definition_extracted(self, tree_path, db_path):
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,
            CATEGORY_MODULE_RESPONSE,
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        cat = db.get_category("S1")
        assert cat is not None
        assert cat.short_definition == "Content involving violent criminal activity."

    def test_sub_categories_extracted(self, tree_path, db_path):
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,
            CATEGORY_MODULE_RESPONSE,
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        cat  = db.get_category("S1")
        subs = cat.sub_categories
        assert len(subs) == 2
        assert subs[0]["id"]   == "VCR.a"
        assert subs[0]["name"] == "Physical violence"
        assert subs[1]["id"]   == "VCR.b"

    def test_short_definition_empty_when_not_in_response(self, tree_path, db_path):
        response_without_short = json.dumps({
            "general_rules": "No violence.",
            "exceptions":    [],
            "one_shots":     [],
            "citations":     [],
        })
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,
            response_without_short,
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        cat = db.get_category("S1")
        assert cat.short_definition == ""
        assert cat.sub_categories   == []


# ---------------------------------------------------------------------------
# OneShot polarity extraction
# ---------------------------------------------------------------------------

class TestOneShotPolarityExtraction:
    def test_polarity_stored_on_shadow_shots(self, tree_path, db_path):
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,
            CATEGORY_MODULE_RESPONSE,
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        cat    = db.get_category("S1")
        shots  = cat.get_shadow_shots()
        assert len(shots) == 2

        positive_shots = [s for s in shots if s.polarity == "positive"]
        negative_shots = [s for s in shots if s.polarity == "negative"]
        assert len(positive_shots) == 1
        assert len(negative_shots) == 1

    def test_polarity_defaults_to_positive_when_missing(self, tree_path, db_path):
        response_no_polarity = json.dumps({
            "short_definition": "test",
            "general_rules":    "rules",
            "sub_categories":   [],
            "exceptions":       [],
            "one_shots": [
                {
                    "query":     "test query",
                    "response":  "test response",
                    "verdict":   "violation",
                    "reasoning": "test",
                    # no "polarity" key
                }
            ],
            "citations": [],
        })
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,
            response_no_polarity,
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        cat   = db.get_category("S1")
        shots = cat.get_shadow_shots()
        assert shots[0].polarity == "positive"


# ---------------------------------------------------------------------------
# Persistence roundtrip — new fields survive save/load
# ---------------------------------------------------------------------------

class TestNewFieldsPersistence:
    def test_new_fields_survive_save_load(self, tree_path, db_path):
        """New fields written by parser must survive JSON roundtrip."""
        client = _make_mock_client([
            GLOBAL_MODULE_RESPONSE,
            CATEGORY_MODULE_RESPONSE,
        ])
        parser = PMPDParser(client)
        db     = parser.parse(tree_path, db_path)

        # Load from disk
        from infrastructure.pmpd_repository import PMPDRepository
        repo    = PMPDRepository(db_path)
        loaded  = repo.load()

        assert loaded.global_module.definitions != {}
        assert len(loaded.global_module.general_eval_principles) == 3
        assert len(loaded.global_module.evaluation_protocol) == 2
        assert len(loaded.global_module.positive_examples) == 1
        assert len(loaded.global_module.negative_examples) == 1

        cat = loaded.get_category("S1")
        assert cat.short_definition != ""
        assert len(cat.sub_categories) == 2
        assert any(s.polarity == "negative" for s in cat.get_shadow_shots())
