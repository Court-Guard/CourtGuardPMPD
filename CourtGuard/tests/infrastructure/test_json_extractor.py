"""
Tests for infrastructure/json_extractor.py

Covers all extraction cases the LLMs produce in practice:
  - Clean JSON
  - Fenced JSON (```json ... ```)
  - Prefixed JSON (preamble before {)
  - Trailing commas
  - Nested objects
  - Total failure returns None
"""

import pytest

from infrastructure.json_extractor import JSONExtractor


@pytest.fixture
def extractor() -> JSONExtractor:
    return JSONExtractor()


class TestJSONExtractor:
    def test_clean_json(self, extractor):
        raw = '{"key": "value", "number": 42}'
        result = extractor.extract(raw)
        assert result == {"key": "value", "number": 42}

    def test_fenced_json_with_language_tag(self, extractor):
        raw = '```json\n{"key": "value"}\n```'
        assert extractor.extract(raw) == {"key": "value"}

    def test_fenced_json_without_language_tag(self, extractor):
        raw = '```\n{"key": "value"}\n```'
        assert extractor.extract(raw) == {"key": "value"}

    def test_preamble_before_json(self, extractor):
        raw = 'Here is the JSON you requested:\n{"key": "value"}'
        assert extractor.extract(raw) == {"key": "value"}

    def test_trailing_comma_before_closing_brace(self, extractor):
        raw = '{"key": "value", "other": "thing",}'
        assert extractor.extract(raw) == {"key": "value", "other": "thing"}

    def test_trailing_comma_before_closing_bracket(self, extractor):
        raw = '{"items": ["a", "b", "c",]}'
        assert extractor.extract(raw) == {"items": ["a", "b", "c"]}

    def test_nested_objects(self, extractor):
        raw = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = extractor.extract(raw)
        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_returns_none_on_pure_text(self, extractor):
        assert extractor.extract("This is not JSON at all.") is None

    def test_returns_none_on_empty_string(self, extractor):
        assert extractor.extract("") is None

    def test_returns_none_on_whitespace_only(self, extractor):
        assert extractor.extract("   \n\t  ") is None

    def test_unicode_values(self, extractor):
        raw = '{"title": "AILuminate® v1.1"}'
        result = extractor.extract(raw)
        assert result["title"] == "AILuminate® v1.1"

    def test_multiline_string_values(self, extractor):
        raw = '{"text": "line one\\nline two"}'
        result = extractor.extract(raw)
        assert "line one" in result["text"]

    def test_fenced_with_preamble_and_trailing_comma(self, extractor):
        """Combines all three hard cases in one response."""
        raw = (
            "Sure! Here is the output:\n"
            "```json\n"
            '{"a": "1", "b": "2",}\n'
            "```\n"
            "Hope that helps!"
        )
        result = extractor.extract(raw)
        assert result == {"a": "1", "b": "2"}

    def test_list_values_preserved(self, extractor):
        raw = '{"principles": ["a", "b", "c"]}'
        result = extractor.extract(raw)
        assert result["principles"] == ["a", "b", "c"]
