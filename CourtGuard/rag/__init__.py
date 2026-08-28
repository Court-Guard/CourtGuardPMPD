"""
RAG package.

Handles document indexing and similarity retrieval for policy-based
red-teaming.

Pipeline
────────
  RAGTuner       — analyses Markdown tree, recommends chunking parameters
  FAISSIndexStore — builds, caches, and loads the FAISS vector index
  DocumentRetriever — queries the index for similar documents
  RAGPipeline    — thin facade wiring all three together

Public API
──────────
    from rag import RAGPipeline, RAGTuner
    from rag.config import RAGConfig, ParameterGrid
"""

from __future__ import annotations

from typing import Any

__all__ = ["RAGPipeline", "RAGTuner"]


def __getattr__(name: str) -> Any:
    """Lazily import heavy RAG runtime modules on first attribute access."""
    if name == "RAGPipeline":
        from rag.pipeline import RAGPipeline

        return RAGPipeline
    if name == "RAGTuner":
        from rag.tuner import RAGTuner

        return RAGTuner
    raise AttributeError(f"module 'rag' has no attribute {name!r}")
