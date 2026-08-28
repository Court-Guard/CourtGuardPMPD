"""
Section Router

Routes raw policy text chunks into named Markdown sections via LLM calls.

Extracted from _llm_assign_sections() in policy_ingester.py.

Design
──────
One LLM call per chunk classifies ALL sections simultaneously.
Total LLM calls = num_chunks (not num_chunks × num_sections).
This is preserved exactly from the original optimisation.

The chr(10) f-string workaround from the original is eliminated by
building prompt parts as a list and joining — cleaner and testable.

Robustness
──────────
Each chunk is processed through an escalating token-limit retry loop.
If every tier is exhausted and the response is still unparseable, a
SectionRoutingError is raised, aborting the entire ingestion run.

Silently skipping a chunk is NEVER acceptable — a partial Markdown tree
is worse than no tree, because the corruption is invisible downstream.
"""

from __future__ import annotations

import re

from core.exceptions import SectionRoutingError
from infrastructure.api_client import APIClient
from infrastructure.config import ModelConfig
from infrastructure.json_extractor import JSONExtractor
from infrastructure.llm_retry_client import LLMRetryClient, RetryConfig
from ingestion.structure_planner import StructurePlan

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------

_ASSIGNMENT_SYSTEM_MSG = (
    "You are a document classifier. Given a policy text chunk and a list "
    "of target sections, you extract the verbatim text that belongs to each "
    "section. You respond only with valid JSON."
)

_CHUNK_SIZE = 5000

# Escalating token limits tried in order for each chunk.
# If the chunk is too dense for 4096 tokens the model truncates the JSON;
# the next tier doubles the budget.  gpt-oss-120b has a 128K context window
# so 32768 output tokens is well within reach even for the densest chunks.
_TOKEN_LIMIT_TIERS: tuple[int, ...] = (4096, 8192, 16384, 32768)


# ---------------------------------------------------------------------------
# Section Router
# ---------------------------------------------------------------------------


class SectionRouter:
    """
    Classifies raw text chunks into policy Markdown sections.

    For each chunk the LLM assigns verbatim text to every section key
    simultaneously — one call per chunk, regardless of section count.

    All chunks MUST be routed successfully.  If a chunk cannot be parsed
    after exhausting all token-limit tiers, SectionRoutingError is raised
    and the entire ingestion run aborts.

    Usage
    -----
        router   = SectionRouter(api_client)
        file_map = router.route(raw_text, plan)
        # file_map: {"categories/defamation.md": "...", "meta/overview.md": "..."}
    """

    def __init__(
        self,
        api_client: APIClient,
        model_config: ModelConfig | None = None,
    ) -> None:
        """
        Args:
            api_client:   Initialized APIClient instance.
            model_config: ModelConfig for bootstrap model selection.
        """
        cfg = model_config or ModelConfig.default()
        self._retry = LLMRetryClient(api_client, RetryConfig.bootstrap())
        self._model = cfg.bootstrap_model
        self._json = JSONExtractor()

    def route(self, raw_text: str, plan: StructurePlan) -> dict[str, str]:
        """
        Classify all chunks of raw_text into Markdown section files.

        Args:
            raw_text: Full text from PDFExtractor.
            plan:     StructurePlan from DocumentStructurePlanner.

        Returns:
            Dict mapping relative .md file paths to their content strings.
            e.g. {"categories/defamation.md": "...", "meta/overview.md": "..."}

        Raises:
            SectionRoutingError: If any chunk permanently fails to parse.
        """
        sections = self._build_section_registry(plan)
        section_keys = list(sections.keys())
        section_desc = self._format_section_descriptions(sections)
        file_map = {k: [] for k in section_keys}

        chunks = [raw_text[i : i + _CHUNK_SIZE] for i in range(0, len(raw_text), _CHUNK_SIZE)]

        print(f"  📂 {len(chunks)} chunks × 1 call each = {len(chunks)} total LLM calls")

        for idx, chunk in enumerate(chunks, start=1):
            self._process_chunk(
                chunk,
                idx,
                len(chunks),
                section_keys,
                section_desc,
                file_map,
            )

        return self._finalise(file_map)

    # ------------------------------------------------------------------
    # Section registry
    # ------------------------------------------------------------------

    @staticmethod
    def _build_section_registry(plan: StructurePlan) -> dict[str, str]:
        """
        Build the ordered dict of {section_key: description} from a plan.

        Fixed sections (meta/overview, definitions, examples) are always
        present.  One categories/<slug>.md entry is added per category.
        """
        sections: dict[str, str] = {
            "meta/overview": ("document title, scope, version, purpose, introduction"),
            "definitions": ("defined terms, glossary entries, and their definitions"),
            "examples": ("examples, case studies, illustrative scenarios, sample violations"),
        }

        for cat in plan.categories:
            slug = re.sub(r"[^\w\-]", "_", cat.lower().strip())
            sections[f"categories/{slug}"] = f"content about the '{cat}' category"

        return sections

    @staticmethod
    def _format_section_descriptions(sections: dict[str, str]) -> str:
        """Format section registry as a prompt-ready string."""
        return "\n".join(f'  "{k}": "{v}"' for k, v in sections.items())

    # ------------------------------------------------------------------
    # Chunk processing
    # ------------------------------------------------------------------

    def _process_chunk(
        self,
        chunk: str,
        idx: int,
        total: int,
        section_keys: list[str],
        section_desc: str,
        file_map: dict[str, list[str]],
    ) -> None:
        """
        Send one chunk to the LLM and accumulate results into file_map.

        Uses an escalating token-limit retry strategy.  Any JSON parsing
        failure (not just visual truncation) triggers the next tier.
        Raises SectionRoutingError if all tiers are exhausted.

        Args:
            chunk:        The text chunk to classify.
            idx:          1-based chunk index (for progress display).
            total:        Total number of chunks.
            section_keys: Ordered list of section keys.
            section_desc: Pre-formatted section description string.
            file_map:     Mutable accumulator — updated in place.

        Raises:
            SectionRoutingError: If the chunk cannot be parsed after
                                 exhausting all _TOKEN_LIMIT_TIERS.
        """
        prompt = self._build_chunk_prompt(chunk, idx, total, section_keys, section_desc)
        last_response: str = ""
        parsed = None

        for attempt, max_tokens in enumerate(_TOKEN_LIMIT_TIERS, start=1):
            if attempt > 1:
                print(
                    f"  ⚠ Chunk {idx} parse failed — "
                    f"retrying with {max_tokens:,} token limit "
                    f"(attempt {attempt}/{len(_TOKEN_LIMIT_TIERS)})..."
                )

            response = self._retry.call(
                prompt,
                _ASSIGNMENT_SYSTEM_MSG,
                self._model,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            last_response = response or ""
            parsed = self._json.extract(last_response)

            if parsed is not None:
                break  # success — stop escalating

        if parsed is None:
            # All tiers exhausted — crash loudly instead of silently skipping.
            raise SectionRoutingError(
                f"Chunk {idx}/{total} could not be parsed into valid JSON after "
                f"{len(_TOKEN_LIMIT_TIERS)} attempts "
                f"(max token limit reached: {_TOKEN_LIMIT_TIERS[-1]:,}).\n"
                f"Last response snippet: {last_response[:300]!r}\n"
                f"Aborting ingestion — partial Markdown trees are not acceptable.",
                chunk_index=idx,
                attempts=len(_TOKEN_LIMIT_TIERS),
                last_response=last_response,
            )

        for key in section_keys:
            val = parsed.get(key)
            if (
                val
                and isinstance(val, str)
                and val.strip().lower() not in ("null", "none", "")
            ):
                file_map[key].append(val.strip())

        if idx % 5 == 0 or idx == total:
            print(f"  ⏳ Processed {idx}/{total} chunks...")

    @staticmethod
    def _build_chunk_prompt(
        chunk: str,
        idx: int,
        total: int,
        section_keys: list[str],
        section_desc: str,
    ) -> str:
        """Build the classification prompt for a single chunk."""
        # Build hazard key lines as a list to avoid chr(10) f-string hack
        hazard_lines = "\n".join(
            f'  "{k}": "verbatim text or null",' for k in section_keys if k.startswith("categories/")
        )

        return (
            f"You are classifying a policy document chunk into sections.\n\n"
            f"SECTIONS (key: description):\n{section_desc}\n\n"
            f"For each section key, extract the verbatim text from the chunk "
            f"that belongs to that section.\n"
            f"If a section has no relevant content in this chunk, use null.\n\n"
            f"Respond ONLY with a valid JSON object — no markdown fences:\n"
            f"{{\n"
            f'  "meta/overview": "verbatim text or null",\n'
            f'  "definitions": "verbatim text or null",\n'
            f'  "examples": "verbatim text or null",\n'
            f"{hazard_lines}\n"
            f"}}\n\n"
            f"CHUNK {idx}/{total}:\n{chunk}"
        )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _finalise(file_map: dict[str, list[str]]) -> dict[str, str]:
        """
        Convert accumulated chunk lists to final content strings
        and append .md extension to all keys.

        Args:
            file_map: Dict mapping section keys to lists of content parts.

        Returns:
            Dict mapping .md file paths to their final content strings.
        """
        return {
            f"{key}.md": "\n\n".join(parts) if parts else "" for key, parts in file_map.items()
        }

