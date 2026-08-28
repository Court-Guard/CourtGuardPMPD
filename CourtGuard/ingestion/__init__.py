"""
Ingestion package.

Converts a raw policy PDF into a RAG-ready Markdown directory tree.

Pipeline
────────
  PDFExtractor            — PDF → raw text (pdfplumber, no LLM)
  DocumentStructurePlanner — raw text → structure plan (1 LLM call)
  SectionRouter           — raw text + plan → file map (N LLM calls)
  MarkdownTreeWriter      — file map → Markdown files on disk

Orchestrated by PolicyIngester (the only public entry point).

Public API
──────────
    from ingestion import PolicyIngester

    client   = APIClient(api_key="...")
    ingester = PolicyIngester(client)
    result   = ingester.ingest("policy.pdf", "policy/md_tree")
"""

from ingestion.policy_ingester import PolicyIngester

__all__ = ["PolicyIngester"]
