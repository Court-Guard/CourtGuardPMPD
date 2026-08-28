"""
Tests for ingestion/structure_planner.py

All LLM calls are mocked — zero API cost.
Covers:
  - Category extraction from valid JSON
  - Fallback on empty / garbage responses
  - Prompt content validation (anti sub-category leakage)
  - Full-text preview (no windowing / no truncation for <120K)
  - StructurePlan field naming (categories, not hazard_categories)
  - Documents with no sub-categories
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from infrastructure.api_client import APIClient
from ingestion.structure_planner import DocumentStructurePlanner, StructurePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_12_CATEGORY_RESPONSE = json.dumps({
    "title": "The AILuminate Assessment Standard v1.1",
    "domain": "Assessment of content hazards in AI responses",
    "categories": [
        "Violent Crimes", "Nonviolent Crimes", "Sex-Related Crimes",
        "Child Sexual Exploitation", "Indiscriminate Weapons",
        "Suicide and Self-Harm", "Intellectual Property", "Privacy",
        "Defamation", "Hate", "Sexual Content", "Specialized Advice",
    ],
    "has_definitions": True,
    "has_examples": True,
})

FLAT_CATEGORIES_RESPONSE = json.dumps({
    "title": "Wikipedia Vandalism Policy",
    "domain": "Detection of vandalism in Wikipedia edits",
    "categories": ["Good Faith Edits", "Bad Faith Edits", "Spam"],
    "has_definitions": False,
    "has_examples": True,
})


def _make_mock_client(response_content: str) -> MagicMock:
    """Build an APIClient mock that returns the given content."""
    client = MagicMock(spec=APIClient)
    client.api_key = "sk-test"
    client.call_model = MagicMock(return_value={
        "content": response_content,
        "success": True,
        "response_time": 0.3,
        "model_used": "test-model",
        "input_tokens": 500,
        "output_tokens": 100,
        "tokens_used": 600,
    })
    return client


def _make_planner(response_content: str) -> DocumentStructurePlanner:
    """Build a planner wired to a mocked APIClient."""
    mock_client = _make_mock_client(response_content)
    planner = DocumentStructurePlanner(mock_client)
    return planner


# ---------------------------------------------------------------------------
# Category extraction
# ---------------------------------------------------------------------------

class TestCategoryExtraction:
    def test_plan_extracts_all_categories(self):
        """12-category JSON -> all 12 present in plan.categories."""
        planner = _make_planner(VALID_12_CATEGORY_RESPONSE)
        plan = planner.plan("A" * 50000)  # 50K dummy text

        assert len(plan.categories) == 12
        assert "Violent Crimes" in plan.categories
        assert "Sexual Content" in plan.categories
        assert "Specialized Advice" in plan.categories

    def test_plan_extracts_title_and_domain(self):
        """Title and domain are correctly extracted."""
        planner = _make_planner(VALID_12_CATEGORY_RESPONSE)
        plan = planner.plan("A" * 50000)

        assert plan.title == "The AILuminate Assessment Standard v1.1"
        assert "content hazards" in plan.domain.lower() or "ai" in plan.domain.lower()

    def test_zero_subcategories_valid(self):
        """A document with flat categories (no sub-categories) is perfectly valid."""
        planner = _make_planner(FLAT_CATEGORIES_RESPONSE)
        plan = planner.plan("B" * 30000)

        assert len(plan.categories) == 3
        assert "Good Faith Edits" in plan.categories
        assert "Spam" in plan.categories


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    def test_plan_fallback_on_empty_response(self):
        """Mock LLM returns empty string -> fallback plan."""
        planner = _make_planner("")
        plan = planner.plan("C" * 10000)

        assert plan.categories == ["general"]
        assert plan.title == "Policy Document"

    def test_plan_fallback_on_non_json(self):
        """Mock LLM returns prose -> fallback plan."""
        planner = _make_planner("I cannot parse this document because it is too complex.")
        plan = planner.plan("D" * 10000)

        assert plan.categories == ["general"]
        assert plan.title == "Policy Document"

    def test_plan_fallback_on_malformed_json(self):
        """Mock LLM returns truncated JSON -> fallback plan."""
        planner = _make_planner('{"title": "Test", "domain": "Test", "categories": ["A", "B"')
        plan = planner.plan("E" * 10000)

        assert plan.categories == ["general"]


# ---------------------------------------------------------------------------
# Prompt content validation
# ---------------------------------------------------------------------------

class TestPromptContent:
    def test_prompt_forbids_sub_types(self):
        """The prompt must instruct the LLM not to include sub-types."""
        prompt = DocumentStructurePlanner._build_prompt("sample text")
        assert "NOT the sub-types" in prompt or "not include sub-types" in prompt.lower()

    def test_prompt_forbids_umbrella_terms(self):
        """The prompt must instruct the LLM not to include umbrella groupings."""
        prompt = DocumentStructurePlanner._build_prompt("sample text")
        assert "umbrella" in prompt.lower()

    def test_prompt_uses_categories_key(self):
        """JSON template in the prompt must use 'categories', not 'hazard_categories'."""
        prompt = DocumentStructurePlanner._build_prompt("sample text")
        assert '"categories"' in prompt
        assert "hazard_categories" not in prompt

    def test_prompt_handles_similar_named_categories(self):
        """Prompt must mention that similar names can be distinct categories."""
        prompt = DocumentStructurePlanner._build_prompt("sample text")
        assert "Sex-Related Crimes" in prompt and "Sexual Content" in prompt


# ---------------------------------------------------------------------------
# Full-text preview (no windowing)
# ---------------------------------------------------------------------------

class TestPreview:
    def test_full_text_under_limit(self):
        """Documents under 120K chars are sent in full — no truncation."""
        text = "X" * 95000
        planner = _make_planner(VALID_12_CATEGORY_RESPONSE)
        preview = planner._build_preview(text)

        assert preview == text
        assert len(preview) == 95000

    def test_truncation_at_limit(self):
        """Documents over 120K chars are truncated with a notice."""
        text = "Y" * 150000
        planner = _make_planner(VALID_12_CATEGORY_RESPONSE)
        preview = planner._build_preview(text)

        assert len(preview) < 150000
        assert "truncated" in preview.lower()
        assert preview.startswith("Y" * 1000)

    def test_short_document_untouched(self):
        """A short document is returned as-is."""
        text = "Hello world, this is a short policy."
        planner = _make_planner(VALID_12_CATEGORY_RESPONSE)
        preview = planner._build_preview(text)

        assert preview == text


# ---------------------------------------------------------------------------
# StructurePlan dataclass
# ---------------------------------------------------------------------------

class TestStructurePlanDataclass:
    def test_categories_field_exists(self):
        """StructurePlan must have 'categories' attribute, not 'hazard_categories'."""
        plan = StructurePlan(title="T", domain="D", categories=["A"])
        assert hasattr(plan, "categories")
        assert not hasattr(plan, "hazard_categories")

    def test_to_dict_uses_categories_key(self):
        """to_dict() must use 'categories' key."""
        plan = StructurePlan(title="T", domain="D", categories=["A", "B"])
        d = plan.to_dict()
        assert "categories" in d
        assert "hazard_categories" not in d
        assert d["categories"] == ["A", "B"]

    def test_fallback_uses_general(self):
        """Fallback plan uses 'general' as a category."""
        plan = StructurePlan.fallback()
        assert plan.categories == ["general"]
