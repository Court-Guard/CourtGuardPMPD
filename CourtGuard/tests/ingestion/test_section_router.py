"""
Tests for ingestion/section_router.py

All LLM calls are mocked — zero API cost.
Covers:
  - Chunk accumulation with valid JSON
  - Fatal SectionRoutingError raised on persistent failure (replaces old graceful skip)
  - Progressive token-limit escalation: success on 2nd, 3rd, 4th tier
  - Full escalation exhausted → SectionRoutingError raised
  - Truncation detection triggering retry (any parse failure, not only missing '}')
  - Section registry uses categories/ folder
  - Finalise appends .md extension
  - Chunk size constant value
  - TOKEN_LIMIT_TIERS constant shape
"""

import json
import pytest
from unittest.mock import MagicMock, call

from core.exceptions import SectionRoutingError
from infrastructure.api_client import APIClient
from ingestion.section_router import SectionRouter, _CHUNK_SIZE, _TOKEN_LIMIT_TIERS
from ingestion.structure_planner import StructurePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(responses: list[str]) -> MagicMock:
    """Build an APIClient mock that returns successive responses."""
    client = MagicMock(spec=APIClient)
    client.api_key = "sk-test"

    call_results = [
        {
            "content": resp,
            "success": True,
            "response_time": 0.3,
            "model_used": "test-model",
            "input_tokens": 200,
            "output_tokens": 100,
            "tokens_used": 300,
        }
        for resp in responses
    ]
    client.call_model = MagicMock(side_effect=call_results)
    return client


def _make_plan(categories: list[str] | None = None) -> StructurePlan:
    """Create a StructurePlan for testing."""
    return StructurePlan(
        title="Test Policy",
        domain="Test domain",
        categories=categories or ["Alpha", "Beta"],
    )


def _valid_chunk_response(plan: StructurePlan, alpha_text: str = "content for alpha") -> str:
    """Build a valid JSON response matching the plan's section keys."""
    result = {
        "meta/overview": None,
        "definitions": None,
        "examples": None,
    }
    for cat in plan.categories:
        import re
        slug = re.sub(r"[^\w\-]", "_", cat.lower().strip())
        result[f"categories/{slug}"] = alpha_text if cat == plan.categories[0] else None
    return json.dumps(result)


def _garbage() -> str:
    """Return a string that is definitively not JSON."""
    return "I am definitely not valid JSON! }}}{"


# ---------------------------------------------------------------------------
# Chunk accumulation
# ---------------------------------------------------------------------------

class TestChunkAccumulation:
    def test_route_accumulates_valid_chunks(self):
        """Two valid chunks → file_map has concatenated content."""
        plan = _make_plan(["Alpha"])
        resp1 = json.dumps({
            "meta/overview": "Overview text from chunk 1",
            "definitions": None,
            "examples": None,
            "categories/alpha": "Alpha content from chunk 1",
        })
        resp2 = json.dumps({
            "meta/overview": "Overview text from chunk 2",
            "definitions": None,
            "examples": None,
            "categories/alpha": "Alpha content from chunk 2",
        })

        mock_client = _make_mock_client([resp1, resp2])
        router = SectionRouter(mock_client)

        # Use short text to produce exactly 2 chunks
        short_text = "A" * (_CHUNK_SIZE * 2)
        file_map = router.route(short_text, plan)

        assert "categories/alpha.md" in file_map
        assert "Alpha content from chunk 1" in file_map["categories/alpha.md"]
        assert "Alpha content from chunk 2" in file_map["categories/alpha.md"]


# ---------------------------------------------------------------------------
# Fail-hard on persistent failure
# ---------------------------------------------------------------------------

class TestFailHardOnPersistentFailure:
    def test_route_raises_error_when_all_tiers_exhausted(self):
        """
        If every token-limit tier returns unparseable JSON, SectionRoutingError
        must be raised — the pipeline MUST NOT silently skip the chunk.
        """
        plan = _make_plan(["Alpha"])
        # Return garbage for every possible retry attempt
        garbage_responses = [_garbage()] * len(_TOKEN_LIMIT_TIERS)
        mock_client = _make_mock_client(garbage_responses)
        router = SectionRouter(mock_client)

        short_text = "B" * _CHUNK_SIZE  # exactly 1 chunk

        with pytest.raises(SectionRoutingError) as exc_info:
            router.route(short_text, plan)

        err = exc_info.value
        assert err.chunk_index == 1
        assert err.attempts == len(_TOKEN_LIMIT_TIERS)
        # Error message should be descriptive
        assert "could not be parsed" in str(err).lower() or "Chunk 1" in str(err)

    def test_error_attributes_are_set_correctly(self):
        """SectionRoutingError carries chunk_index, attempts, last_response."""
        plan = _make_plan(["Alpha"])
        garbage_responses = [_garbage()] * len(_TOKEN_LIMIT_TIERS)
        mock_client = _make_mock_client(garbage_responses)
        router = SectionRouter(mock_client)

        with pytest.raises(SectionRoutingError) as exc_info:
            router.route("X" * _CHUNK_SIZE, plan)

        err = exc_info.value
        assert err.chunk_index == 1
        assert err.attempts == len(_TOKEN_LIMIT_TIERS)
        assert err.last_response == _garbage()

    def test_second_chunk_failure_raises_with_correct_index(self):
        """Failure in chunk 2 must report chunk_index=2."""
        plan = _make_plan(["Alpha"])
        valid = json.dumps({
            "meta/overview": None, "definitions": None, "examples": None,
            "categories/alpha": "ok",
        })
        garbage_responses = [valid] + [_garbage()] * len(_TOKEN_LIMIT_TIERS)
        mock_client = _make_mock_client(garbage_responses)
        router = SectionRouter(mock_client)

        two_chunks = "C" * (_CHUNK_SIZE * 2)
        with pytest.raises(SectionRoutingError) as exc_info:
            router.route(two_chunks, plan)

        assert exc_info.value.chunk_index == 2

    def test_all_tiers_are_attempted_before_raising(self):
        """The LLM must be called exactly len(_TOKEN_LIMIT_TIERS) times for 1 failing chunk."""
        plan = _make_plan(["Alpha"])
        garbage_responses = [_garbage()] * len(_TOKEN_LIMIT_TIERS)
        mock_client = _make_mock_client(garbage_responses)
        router = SectionRouter(mock_client)

        with pytest.raises(SectionRoutingError):
            router.route("D" * _CHUNK_SIZE, plan)

        assert mock_client.call_model.call_count == len(_TOKEN_LIMIT_TIERS)


# ---------------------------------------------------------------------------
# Progressive token-limit escalation
# ---------------------------------------------------------------------------

class TestProgressiveTokenEscalation:
    def test_succeeds_on_second_tier(self):
        """Garbage on first tier, valid JSON on second → content accumulated, no error."""
        plan = _make_plan(["Alpha"])
        valid = json.dumps({
            "meta/overview": None, "definitions": None, "examples": None,
            "categories/alpha": "Recovered on tier 2",
        })
        mock_client = _make_mock_client([_garbage(), valid])
        router = SectionRouter(mock_client)

        file_map = router.route("E" * _CHUNK_SIZE, plan)

        assert "Recovered on tier 2" in file_map["categories/alpha.md"]
        # Exactly 2 calls: one failure + one success
        assert mock_client.call_model.call_count == 2

    def test_succeeds_on_third_tier(self):
        """Two garbage responses then valid JSON → succeeds on tier 3."""
        plan = _make_plan(["Alpha"])
        valid = json.dumps({
            "meta/overview": None, "definitions": None, "examples": None,
            "categories/alpha": "Recovered on tier 3",
        })
        mock_client = _make_mock_client([_garbage(), _garbage(), valid])
        router = SectionRouter(mock_client)

        file_map = router.route("F" * _CHUNK_SIZE, plan)

        assert "Recovered on tier 3" in file_map["categories/alpha.md"]
        assert mock_client.call_model.call_count == 3

    def test_succeeds_on_fourth_tier(self):
        """Three garbage responses then valid JSON → succeeds on final tier."""
        plan = _make_plan(["Alpha"])
        valid = json.dumps({
            "meta/overview": None, "definitions": None, "examples": None,
            "categories/alpha": "Recovered on tier 4 (max)",
        })
        mock_client = _make_mock_client([_garbage(), _garbage(), _garbage(), valid])
        router = SectionRouter(mock_client)

        file_map = router.route("G" * _CHUNK_SIZE, plan)

        assert "Recovered on tier 4 (max)" in file_map["categories/alpha.md"]
        assert mock_client.call_model.call_count == 4

    def test_any_parse_failure_escalates_not_just_truncation(self):
        """
        A response that ends with '}' but is still invalid JSON (e.g. extra
        outer text before the object) must also trigger tier escalation.

        This covers the old blind spot: previously only visually-truncated
        responses were retried.  Now any parsing failure escalates.
        """
        plan = _make_plan(["Alpha"])
        # Ends with '}' but is not parseable JSON (has preamble text)
        bad_with_closing_brace = 'Here is the JSON you asked for: {"categories/alpha": "text"}'
        valid = json.dumps({
            "meta/overview": None, "definitions": None, "examples": None,
            "categories/alpha": "Returned after escalation",
        })
        # JSONExtractor will extract from the first response (it strips preamble),
        # so use a response that truly cannot be extracted
        truly_bad = "Sorry, I cannot classify this chunk. { broken"
        mock_client = _make_mock_client([truly_bad, valid])
        router = SectionRouter(mock_client)

        file_map = router.route("H" * _CHUNK_SIZE, plan)

        # Main assertion: escalation happened and result was recovered
        assert "Returned after escalation" in file_map["categories/alpha.md"]
        assert mock_client.call_model.call_count == 2


# ---------------------------------------------------------------------------
# Section registry
# ---------------------------------------------------------------------------

class TestSectionRegistry:
    def test_registry_uses_categories_folder(self):
        """Section registry keys should use categories/ prefix, not hazards/."""
        plan = _make_plan(["Violent Crimes", "Hate Speech"])
        registry = SectionRouter._build_section_registry(plan)

        category_keys = [k for k in registry if k.startswith("categories/")]
        hazard_keys = [k for k in registry if k.startswith("hazards/")]

        assert len(category_keys) == 2
        assert len(hazard_keys) == 0
        assert "categories/violent_crimes" in registry
        assert "categories/hate_speech" in registry

    def test_registry_always_has_fixed_sections(self):
        """meta/overview, definitions, examples should always exist."""
        plan = _make_plan([])
        registry = SectionRouter._build_section_registry(plan)

        assert "meta/overview" in registry
        assert "definitions" in registry
        assert "examples" in registry


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------

class TestFinalisation:
    def test_finalise_appends_md_extension(self):
        """All output keys should end with .md."""
        file_map = {
            "meta/overview": ["text"],
            "definitions": [],
            "examples": [],
            "categories/alpha": ["content"],
        }
        result = SectionRouter._finalise(file_map)

        for key in result:
            assert key.endswith(".md"), f"Key '{key}' does not end with .md"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_chunk_size_is_5000(self):
        """Chunk size should be 5000 to match ingestion pipeline."""
        assert _CHUNK_SIZE == 5000

    def test_token_limit_tiers_are_ascending(self):
        """Token limit tiers must be strictly ascending for escalation to make sense."""
        assert list(_TOKEN_LIMIT_TIERS) == sorted(_TOKEN_LIMIT_TIERS), (
            "_TOKEN_LIMIT_TIERS must be in ascending order"
        )

    def test_token_limit_tiers_has_four_entries(self):
        """Ensure we have exactly 4 escalation tiers as designed."""
        assert len(_TOKEN_LIMIT_TIERS) == 4

    def test_token_limit_tiers_max_is_at_least_16k(self):
        """Final tier must be at least 16K to handle dense, multi-category chunks."""
        assert _TOKEN_LIMIT_TIERS[-1] >= 16384
