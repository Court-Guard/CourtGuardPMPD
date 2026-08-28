"""Unit tests for the PMPD specialist output parser."""

from debate.pmpd_debate import _DefenderOutputParser


VALID_CATS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]


def test_supported_unsafe_with_valid_top_level_category():
    parser = _DefenderOutputParser()
    raw = """
    CATEGORY_FIT: SUPPORTED
    FINAL_LABEL: UNSAFE
    CATEGORIES: S9
    BASIS_CODE: MATCHED_POLICY
    CATEGORY_NOTES: The supplied category matches directly.
    REASONING: The response enables violent wrongdoing.
    """

    result = parser.parse(raw, valid_categories=["S9"])

    assert result["supported"] is True
    assert result["confirmed"] is True
    assert result["category_fit"] == "SUPPORTED"
    assert result["categories"] == ["S9"]
    assert result["parse_success"] is True
    assert result["basis_code"] == "MATCHED_POLICY"


def test_supported_safe_is_not_confirmed_but_is_parse_success():
    parser = _DefenderOutputParser()
    raw = """
    CATEGORY_FIT: SUPPORTED
    FINAL_LABEL: SAFE
    CATEGORIES:
    BASIS_CODE: NO_VIOLATION
    CATEGORY_NOTES: The content stays within the provided category scope and does not violate it.
    REASONING: The response is non-actionable and compliant.
    """

    result = parser.parse(raw, valid_categories=["S9"])

    assert result["supported"] is True
    assert result["confirmed"] is False
    assert result["final_label"] == "SAFE"
    assert result["parse_success"] is True
    assert result["basis_code"] == "NO_VIOLATION"


def test_supported_with_invalid_category_marks_parse_unsuccessful():
    parser = _DefenderOutputParser()
    raw = """
    CATEGORY_FIT: SUPPORTED
    FINAL_LABEL: UNSAFE
    CATEGORIES: VCR
    BASIS_CODE: MATCHED_POLICY
    CATEGORY_NOTES: The supplied category matches directly.
    REASONING: The response enables violent wrongdoing.
    """

    result = parser.parse(raw, valid_categories=VALID_CATS)

    assert result["supported"] is True
    assert result["categories"] == []
    assert result["parse_success"] is False
    assert "Parser warning" in result["reasoning"]


def test_unsupported_with_empty_categories_is_valid():
    parser = _DefenderOutputParser()
    raw = """
    CATEGORY_FIT: UNSUPPORTED
    FINAL_LABEL: SAFE
    CATEGORIES:
    BASIS_CODE: INSUFFICIENT_COVERAGE
    CATEGORY_NOTES: The supplied categories do not align closely enough with the input.
    REASONING: The category scope is insufficient for a reliable judgment.
    """

    result = parser.parse(raw, valid_categories=VALID_CATS)

    assert result["supported"] is False
    assert result["confirmed"] is False
    assert result["categories"] == []
    assert result["parse_success"] is True
    assert result["basis_code"] == "INSUFFICIENT_COVERAGE"
    assert "align" in result["category_notes"].lower()


def test_missing_category_fit_falls_back_without_raising():
    parser = _DefenderOutputParser()
    raw = "This looks compliant within the provided category scope."

    result = parser.parse(raw, valid_categories=VALID_CATS)

    assert result["parse_success"] is False
    assert result["category_fit"] == "UNSUPPORTED"
    assert result["raw_output"] == raw
