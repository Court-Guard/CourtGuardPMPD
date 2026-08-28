"""
Tests for debate/attacker_parser.py

Create as: tests/debate/test_attacker_parser.py

Covers:
  - Well-formed structured block extraction
  - Empty categories -> SAFE
  - Multiple categories
  - PRELIMINARY_LABEL: SAFE -> no accusation
  - PRELIMINARY_LABEL: UNSAFE -> accusation
  - Category validation against valid_categories list
  - Relaxed pass 2 parsing
  - Inference pass 3 from free text
  - Empty/None input -> safe default
  - ParsedAccusation.is_accusation property
  - parse_success flag on fallback
  - raw_output always preserved
"""

import pytest
from debate.attacker_parser import AttackerOutputParser, ParsedAccusation


# ---------------------------------------------------------------------------
# Sample outputs
# ---------------------------------------------------------------------------

WELL_FORMED_UNSAFE = """
PRELIMINARY_LABEL: 1
CANDIDATE_CATEGORIES: S9, S6
EVIDENCE_SPANS: "step-by-step instructions for assault"; "physical harm"
"""

WELL_FORMED_UNSAFE_SELECTOR_ONLY = """
CANDIDATE_CATEGORIES: S9, S6
EVIDENCE_SPANS: "step-by-step instructions for assault"; "physical harm"
"""

WELL_FORMED_UNSAFE_LEGACY = """
ACCUSATION: The response provides step-by-step instructions for assault,
which directly violates the violent crimes policy category.
EVIDENCE: "step-by-step instructions for assault"
CATEGORIES: S9, S6
PRELIMINARY_LABEL: 1
"""

WELL_FORMED_SAFE = """
PRELIMINARY_LABEL: 0
CANDIDATE_CATEGORIES:
EVIDENCE_SPANS:
"""

SINGLE_CATEGORY = """
PRELIMINARY_LABEL: 1
CANDIDATE_CATEGORIES: S2
EVIDENCE_SPANS: "defamatory claims"
"""

MISSING_BLOCK = """
I reviewed this content and it seems to violate the S9 category
because it describes physical violence in detail.
"""

EMPTY_INPUT = ""

LABEL_ONLY = """
After analysis: PRELIMINARY_LABEL: 0
"""

VALID_CATS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]


# ---------------------------------------------------------------------------
# Parser fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return AttackerOutputParser(valid_categories=VALID_CATS)


@pytest.fixture
def unconstrained_parser():
    """Parser with no valid_categories list — accepts any code."""
    return AttackerOutputParser()


# ---------------------------------------------------------------------------
# Well-formed structured block
# ---------------------------------------------------------------------------

class TestWellFormedParsing:
    def test_extracts_label_unsafe(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE)
        assert result.preliminary_label == "UNSAFE"

    def test_extracts_label_safe(self, parser):
        result = parser.parse(WELL_FORMED_SAFE)
        assert result.preliminary_label == "SAFE"

    def test_extracts_categories(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE)
        assert "S9" in result.categories
        assert "S6" in result.categories

    def test_empty_categories_safe(self, parser):
        result = parser.parse(WELL_FORMED_SAFE)
        assert result.categories == []

    def test_single_category(self, parser):
        result = parser.parse(SINGLE_CATEGORY)
        assert result.categories == ["S2"]

    def test_selector_only_block_without_preliminary_label_parses(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE_SELECTOR_ONLY)
        assert result.preliminary_label == "UNSAFE"
        assert result.categories == ["S9", "S6"]
        assert result.parse_success is True

    def test_category_count_is_capped_at_two(self, parser):
        text = """
        CANDIDATE_CATEGORIES: S9, S6, S2
        EVIDENCE_SPANS: "quote"
        """
        result = parser.parse(text)
        assert result.categories == ["S9", "S6"]

    def test_accusation_text_extracted(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE)
        assert "Selected categories" in result.accusation

    def test_evidence_text_extracted(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE)
        assert "step-by-step instructions for assault" in result.evidence

    def test_raw_output_always_preserved(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE)
        assert result.raw_output == WELL_FORMED_UNSAFE

    def test_parse_success_true(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE)
        assert result.parse_success is True

    def test_legacy_accusation_block_still_parses(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE_LEGACY)
        assert result.preliminary_label == "UNSAFE"
        assert "S9" in result.categories


# ---------------------------------------------------------------------------
# is_accusation property
# ---------------------------------------------------------------------------

class TestIsAccusation:
    def test_unsafe_label_is_accusation(self, parser):
        result = parser.parse(WELL_FORMED_UNSAFE)
        assert result.is_accusation is True

    def test_safe_label_not_accusation(self, parser):
        result = parser.parse(WELL_FORMED_SAFE)
        assert result.is_accusation is False

    def test_categories_without_label_is_accusation(self):
        """If categories found but label missing, still treat as accusation."""
        result = ParsedAccusation(
            accusation="test",
            categories=["S9"],
            preliminary_label="SAFE",
            raw_output="",
            default_label="SAFE",
        )
        # categories present → is_accusation True regardless of label
        assert result.is_accusation is True


# ---------------------------------------------------------------------------
# Fallback parsing
# ---------------------------------------------------------------------------

class TestFallbackParsing:
    def test_empty_input_returns_safe_default(self, parser):
        result = parser.parse(EMPTY_INPUT)
        assert result.preliminary_label == "SAFE"
        assert result.categories        == []
        assert result.parse_success     is False

    def test_none_like_input_returns_safe_default(self, parser):
        result = parser.parse("   ")
        assert result.parse_success is False

    def test_missing_block_infers_from_text(self, parser):
        result = parser.parse(MISSING_BLOCK)
        # Should detect S9 from free text and infer violation
        assert "S9" in result.categories or result.preliminary_label == "UNSAFE"

    def test_raw_output_preserved_on_failure(self, parser):
        result = parser.parse(EMPTY_INPUT)
        assert result.raw_output == EMPTY_INPUT

    def test_label_only_parsed(self, parser):
        result = parser.parse(LABEL_ONLY)
        assert result.preliminary_label == "SAFE"


# ---------------------------------------------------------------------------
# Category validation
# ---------------------------------------------------------------------------

class TestCategoryValidation:
    def test_invalid_category_filtered(self, parser):
        text = """
        ACCUSATION: test
        CATEGORIES: S9, FAKECODE, S6
        PRELIMINARY_LABEL: 1
        """
        result = parser.parse(text)
        assert "FAKECODE" not in result.categories
        assert "S9" in result.categories

    def test_structured_output_with_unrecognized_category_marks_parse_unsuccessful(self, parser):
        text = """
        ACCUSATION: test
        EVIDENCE: "bad code"
        CATEGORIES: VCR
        PRELIMINARY_LABEL: 1
        """
        result = parser.parse(text)
        assert result.categories == []
        assert result.preliminary_label == "UNSAFE"
        assert result.parse_success is False
        assert result.had_category_text is True
        assert result.invalid_category_mismatch is True

    def test_unconstrained_accepts_any_code(self, unconstrained_parser):
        text = """
        ACCUSATION: test
        CATEGORIES: CUSTOMCODE, ABC
        PRELIMINARY_LABEL: 1
        """
        result = unconstrained_parser.parse(text)
        # Should accept codes without validation
        assert len(result.categories) > 0

    def test_structured_output_with_top_level_ids_parses_successfully(self, parser):
        text = """
        ACCUSATION: test
        EVIDENCE: "valid code"
        CATEGORIES: S9
        PRELIMINARY_LABEL: 1
        """
        result = parser.parse(text)
        assert result.categories == ["S9"]
        assert result.parse_success is True


# ---------------------------------------------------------------------------
# safe_default classmethod
# ---------------------------------------------------------------------------

class TestSafeDefault:
    def test_safe_default_values(self):
        result = ParsedAccusation.safe_default("some raw text")
        assert result.preliminary_label == "SAFE"
        assert result.categories        == []
        assert result.parse_success     is False
        assert result.raw_output        == "some raw text"
