"""
Document Retriever

Queries a built FAISS index for similar documents.

Extracted from RAGPipeline.search_similar_documents() and
RAGPipeline.get_category_documents() in rag_pipeline.py.

Fixes
─────
  get_category_documents used an empty-string similarity search:

      all_docs = self.db.similarity_search("", k=1000)  # undefined behaviour

  The FAISS docstore exposes all stored documents directly via its
  internal _dict — no query needed for category-based filtering:

      all_docs = list(self.db.docstore._dict.values())

  This returns actual Document objects without a meaningless similarity
  score distorting which documents are returned.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.config import RAGConfig


class DocumentRetriever:
    """
    Queries a built FAISS index for similar or category-filtered documents.

    Requires a built FAISSIndexStore — inject the .db property.

    Usage
    -----
        store     = FAISSIndexStore(config)
        db, meta  = store.build_or_load(tree_path, policy_name)

        retriever = DocumentRetriever(db, config)
        docs      = retriever.search("query about defamation", k=5)
        cat_docs  = retriever.get_by_category("hazards")
    """

    def __init__(self, db: FAISS, config: RAGConfig) -> None:
        """
        Args:
            db:     Built FAISS database from FAISSIndexStore.
            config: RAGConfig supplying the default_k value.
        """
        self._db = db
        self._config = config

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int | None = None,
        filter_category: str | None = None,
    ) -> list[Document]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query:           Similarity search query string.
            k:               Number of documents to retrieve.
                             Uses config.default_k if not specified.
            filter_category: Optional metadata category filter,
                             e.g. "hazards", "definitions", "meta".

        Returns:
            List of most relevant Document objects.
        """
        k_to_use = k if k is not None else self._config.default_k
        docs = self._db.similarity_search(query, k=k_to_use)

        if filter_category:
            docs = [d for d in docs if d.metadata.get("category") == filter_category]

        return docs

    # ------------------------------------------------------------------
    # Category retrieval (fixed)
    # ------------------------------------------------------------------

    def get_by_category(self, category: str, limit: int = 10) -> list[Document]:
        """
        Retrieve documents belonging to a specific category.

        Uses FAISS docstore direct access instead of an empty-string
        similarity search (which had undefined behaviour in the original).

        Args:
            category: Category name to filter on, e.g. "hazards".
            limit:    Maximum number of documents to return.

        Returns:
            List of Document objects from the specified category.
        """
        all_docs = list(self._db.docstore._dict.values())
        filtered = [
            d
            for d in all_docs
            if isinstance(d, Document) and d.metadata.get("category") == category
        ]
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Index statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Return statistics about the current FAISS index.

        Returns:
            Dict with status, vector count, dimension, and config values.
        """
        return {
            "status": "ready",
            "total_vectors": self._db.index.ntotal,
            "embedding_dimension": self._db.index.d,
            "chunk_size": self._config.chunk_size,
            "chunk_overlap": self._config.chunk_overlap,
            "default_k": self._config.default_k,
            "embeddings_model": self._config.embeddings_model,
        }
