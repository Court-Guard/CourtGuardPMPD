"""
JSON Extractor Utility

Single implementation of the JSON extraction logic that was previously
duplicated as _extract_json() in four separate modules:
  - pmpd_parser.py
  - prompt_generator.py
  - rag_pipeline.py        (inlined)
  - policy_ingester.py     (inlined twice)

Handles the full range of LLM response formats encountered in practice:
  1. Clean JSON  — direct json.loads()
  2. Fenced JSON — ```json ... ``` or ``` ... ```
  3. Prefixed    — preamble text before the opening brace
  4. Trailing commas — before } or ] (common LLM formatting mistake)
  5. Total failure — returns None so callers can apply their own fallback
"""

from __future__ import annotations

import json
import re


class JSONExtractor:
    """
    Robustly extracts a JSON object from a raw LLM response string.

    Stateless — all methods are static.  Instantiate once and reuse,
    or call the class methods directly.

    Usage
    -----
        extractor = JSONExtractor()
        parsed    = extractor.extract(raw_llm_response)
        if parsed is None:
            # apply fallback
    """

    # Compiled patterns — built once at class load, reused across calls.
    _FENCE_PATTERN = re.compile(r"```(?:json)?|```")
    _TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
    _JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, raw: str) -> dict | None:
        """
        Extract the first valid JSON object from a raw LLM response.

        Applies a three-pass strategy:
          Pass 1 — strip fences and trailing commas, try direct parse.
          Pass 2 — extract outermost {...} block, try parse.
          Pass 3 — return None (caller applies its own fallback).

        Args:
            raw: Raw string output from an LLM call.

        Returns:
            Parsed dict, or None if all passes fail.
        """
        if not raw or not raw.strip():
            return None

        cleaned = self._clean(raw)

        # Pass 1: cleaned string directly
        result = self._try_parse(cleaned)
        if result is not None:
            return result

        # Pass 2: extract outermost {...} block
        match = self._JSON_OBJECT_PATTERN.search(cleaned)
        if match:
            result = self._try_parse(self._fix_trailing_commas(match.group()))
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean(self, raw: str) -> str:
        """Strip markdown fences and fix trailing commas."""
        cleaned = self._FENCE_PATTERN.sub("", raw).strip()
        return self._fix_trailing_commas(cleaned)

    @staticmethod
    def _fix_trailing_commas(text: str) -> str:
        """Remove trailing commas before } or ] — common LLM formatting mistake."""
        return re.sub(r",\s*([}\]])", r"\1", text)

    @staticmethod
    def _try_parse(text: str) -> dict | None:
        """Attempt json.loads(); return None on any failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
