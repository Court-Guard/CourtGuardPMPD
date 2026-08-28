"""
Policy Ingester — Orchestrator

Converts a raw policy PDF into a RAG-ready Markdown directory tree.

This class is a thin orchestrator — it wires the pipelines together:

  Stage 1 — DocumentExtractor       : doc → raw text (MD, DOCX, LlamaParse)
  Stage 2 — DocumentStructurePlanner : raw text → StructurePlan (1 LLM call)
  Stage 3 — SectionRouter           : raw text + plan → file map (N LLM calls)
  Stage 4 — MarkdownTreeWriter      : file map → Markdown files on disk

Total LLM calls = num_chunks + 1  (same as original — preserved exactly)

Usage
─────
    client   = APIClient(api_key="sk-or-v1-...")
    ingester = PolicyIngester(client)
    result   = ingester.ingest("policy.pdf", "policy/md_tree")
"""

from __future__ import annotations

import os

from infrastructure.api_client import APIClient
from infrastructure.config import ModelConfig
from ingestion.markdown_tree_writer import MarkdownTreeWriter
from ingestion.extractors.extractor_factory import ExtractorFactory
from ingestion.section_router import SectionRouter
from ingestion.structure_planner import DocumentStructurePlanner


class PolicyIngester:
    """
    Converts a raw policy PDF into a RAG-ready Markdown directory tree.

    Dependencies are injected at construction time so each stage can be
    tested independently or swapped for a different implementation.

    Usage
    -----
        client   = APIClient(api_key="sk-or-v1-...")
        ingester = PolicyIngester(client)
        result   = ingester.ingest("policy.pdf", "policy/md_tree")
    """

    def __init__(
        self,
        api_client: APIClient,
        model_config: ModelConfig | None = None,
    ) -> None:
        """
        Args:
            api_client:   Initialized APIClient for LLM calls.
            model_config: ModelConfig controlling which model is used.
                          Defaults to ModelConfig.default().
        """
        cfg = model_config or ModelConfig.default()

        self._planner = DocumentStructurePlanner(api_client, cfg)
        self._router = SectionRouter(api_client, cfg)
        self._writer = MarkdownTreeWriter()

    def ingest(self, file_path: str, output_dir: str) -> dict:
        """
        Full ingestion pipeline: Document → Markdown tree.

        Steps:
            1. DocumentExtractor (Markdown, Word, LlamaParse) extracts text/tables.
            2. StructurePlanner identifies document structure (1 LLM call)
            3. SectionRouter   routes each chunk to sections (N LLM calls)
            4. MarkdownTreeWriter writes all section files to disk

        Args:
            file_path:  Path to the input document (.pdf, .docx, .md).
            output_dir: Root directory for the Markdown tree output.

        Returns:
            Dict with tree_path, categories, section_count, title, domain.
        """
        print(f"\n{'='*60}")
        print(f"PolicyIngester: {os.path.basename(file_path)}")
        print(f"{'='*60}")

        # Stage 1: Extract
        print(f"\n[1/4] Extracting raw textual geometry from {os.path.basename(file_path)}...")
        extractor = ExtractorFactory.get_extractor(file_path)
        raw_text = extractor.extract_text(file_path)
        print(f"  ✅ Extracted {len(raw_text):,} characters")

        # Stage 2: Plan structure
        print("\n[2/4] Identifying document structure...")
        plan = self._planner.plan(raw_text)
        print(f"  ✅ Title     : {plan.title}")
        print(f"  ✅ Domain    : {plan.domain}")
        print(f"  ✅ Categories: {plan.categories}")

        # Stage 3: Route sections
        print("\n[3/4] Routing text to Markdown files...")
        file_map = self._router.route(raw_text, plan)
        non_empty = sum(1 for v in file_map.values() if v.strip())
        print(f"  ✅ {non_empty}/{len(file_map)} file(s) have content")

        # Stage 4: Write tree
        print("\n[4/4] Writing Markdown tree...")
        self._writer.write(file_map, plan, output_dir)

        print(f"\n{'='*60}")
        print("PolicyIngester: Complete")
        print(f"  Tree      : {output_dir}")
        print(f"  Files     : {len(file_map)}")
        print(f"  Categories: {plan.categories}")
        print(f"{'='*60}\n")

        return {
            "tree_path": output_dir,
            "categories": plan.categories,
            "section_count": len(file_map),
            "title": plan.title,
            "domain": plan.domain,
        }
