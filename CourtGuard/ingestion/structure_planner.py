"""
Document Structure Planner

Identifies a policy document's title, domain, and top-level categories
via a single LLM call.

Extracted from _llm_plan_structure() in policy_ingester.py.

Design
──────
The planner scans five ordered windows of the raw text (start,
structure anchor, middle, keyword-density, end) and combines them so
the LLM sees the document's natural hierarchy — parent categories
appear before their sub-types, making taxonomy-level extraction
reliable.

A typed StructurePlan dataclass is returned instead of a raw dict,
so downstream consumers (SectionRouter, MarkdownTreeWriter) have
explicit, type-checked access to the plan fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.api_client import APIClient
from infrastructure.config import ModelConfig
from infrastructure.json_extractor import JSONExtractor
from infrastructure.llm_retry_client import LLMRetryClient, RetryConfig

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------

_STRUCTURE_SYSTEM_MSG = (
    "You are a document architect. You analyse policy document text "
    "and identify its structure. You respond only with valid JSON."
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class StructurePlan:
    """
    Typed output of the DocumentStructurePlanner.

    Attributes
    ----------
    title           : Exact document title extracted from the text
    domain          : One-line description of what this policy covers
    categories      : List of top-level category names from the document
    has_definitions : Whether the document contains a definitions section
    has_examples    : Whether the document contains worked examples
    """

    title: str
    domain: str
    categories: list[str]
    has_definitions: bool = True
    has_examples: bool = False

    # Fallback used when LLM parsing fails
    @classmethod
    def fallback(cls) -> StructurePlan:
        """Return a minimal fallback plan when LLM extraction fails."""
        return cls(
            title="Policy Document",
            domain="General policy",
            categories=["general"],
            has_definitions=True,
            has_examples=False,
        )

    def to_dict(self) -> dict:
        """Convert to a plain dict for downstream compatibility."""
        return {
            "title": self.title,
            "domain": self.domain,
            "categories": self.categories,
            "has_definitions": self.has_definitions,
            "has_examples": self.has_examples,
        }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class DocumentStructurePlanner:
    """
    Analyses raw text and returns a StructurePlan via one LLM call.

    Sends the full document text to the LLM so it can see every
    category, heading, and structural element.  Documents up to ~120K
    chars are sent in full; longer documents are truncated with notice.

    Usage
    -----
        planner = DocumentStructurePlanner(api_client)
        plan    = planner.plan(raw_text)
        print(plan.categories)
    """

    # Safety cap — truncate only if the document exceeds safe context
    _MAX_CHARS = 120_000

    def __init__(
        self,
        api_client: APIClient,
        model_config: ModelConfig | None = None,
    ) -> None:
        """
        Args:
            api_client:   Initialized APIClient instance.
            model_config: ModelConfig for bootstrap model selection.
                          Defaults to ModelConfig.default().
        """
        cfg = model_config or ModelConfig.default()
        self._retry = LLMRetryClient(api_client, RetryConfig.bootstrap())
        self._model = cfg.bootstrap_model
        self._json = JSONExtractor()

    def plan(self, raw_text: str) -> StructurePlan:
        """
        Identify document structure from raw extracted text.

        Args:
            raw_text: Full text output from PDFExtractor.extract().

        Returns:
            StructurePlan with title, domain, and categories.
            Falls back to StructurePlan.fallback() on LLM parse failure.
        """
        preview = self._build_preview(raw_text)
        prompt = self._build_prompt(preview)

        raw = self._retry.call(
            prompt, _STRUCTURE_SYSTEM_MSG, self._model, max_tokens=2048, temperature=0.1
        )
        parsed = self._json.extract(raw)

        if not parsed:
            print("  ⚠ Could not parse structure JSON — using fallback.")
            return StructurePlan.fallback()

        return StructurePlan(
            title=parsed.get("title", "Policy Document"),
            domain=parsed.get("domain", "General policy"),
            categories=parsed.get("categories", ["general"]),
            has_definitions=parsed.get("has_definitions", True),
            has_examples=parsed.get("has_examples", False),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_preview(self, raw_text: str) -> str:
        """
        Return the full document text for the LLM.

        Only truncates if the text exceeds _MAX_CHARS (120K) to stay
        within model context limits.  For typical policy documents
        (60-100K chars) the full text is sent.
        """
        if len(raw_text) <= self._MAX_CHARS:
            return raw_text

        # For very large documents, keep as much as possible with
        # a note about truncation
        return (
            raw_text[:self._MAX_CHARS]
            + "\n\n[... document truncated at 120K characters ...]"
        )

    @staticmethod
    def _build_prompt(preview: str) -> str:
        """Build the structure-extraction prompt."""
        return f"""Analyse the policy document below and identify its structure.

Return ONLY a valid JSON object — no markdown fences, no explanation:
{{
  "title": "exact document title from the text",
  "domain": "one-line description of what this policy covers",
  "categories": ["Category1", "Category2", "..."],
  "has_definitions": true,
  "has_examples": true
}}

CRITICAL RULES for `categories`:
1. Extract ONLY the top-level categories from the document's primary taxonomy.
   - Look for numbered lists, section headings, or taxonomy tables that define
     the main category structure of the document.
   - If a category contains sub-types or sub-categories beneath it,
     include ONLY the parent category, NOT the sub-types.
   - Example: if you see "Sex-Related Crimes" with sub-items "Sex trafficking"
     and "Sexual assault" underneath, include ONLY "Sex-Related Crimes".
2. Do NOT include umbrella/grouping terms that merely organize categories
   into clusters (e.g., "Physical Hazards" grouping several crime categories).
3. Do NOT include metadata fields, document version info, or formatting labels.
4. Two categories with similar names but different scopes ARE separate categories.
   (e.g., "Sex-Related Crimes" and "Sexual Content" are distinct).
5. If the document does not define sub-categories, treat each listed category
   as a top-level category directly — this is valid and expected for some documents.
6. Most policy documents define 8-15 top-level categories. If you find fewer
   than 5 or more than 25, re-examine whether you are at the correct taxonomy level.

FULL DOCUMENT:
{preview}"""
