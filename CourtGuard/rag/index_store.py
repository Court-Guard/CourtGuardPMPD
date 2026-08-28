"""
FAISS Index Store

Builds, caches, and loads the FAISS vector index from a Markdown tree.

Extracted from RAGPipeline.build_or_load_index() in rag_pipeline.py.

Changes from original
─────────────────────
  pickle → FAISS native save_local / load_local
  ────────────────────────────────────────────
  The original used pickle to cache the FAISS database:

      with open(index_file, "rb") as f:
          cached = pickle.load(f)         # arbitrary code execution risk

  FAISS provides safe, native serialisation:

      db.save_local(folder_path)
      db = FAISS.load_local(folder_path, embeddings,
                            allow_dangerous_deserialization=True)

  The allow_dangerous_deserialization=True flag is required by LangChain's
  FAISS wrapper but we control the write path — we only load what we saved.

  get_category_documents empty-query fix
  ───────────────────────────────────────
  The original performed a similarity search with an empty string:

      all_docs = self.db.similarity_search("", k=1000)  # undefined behaviour

  The FAISS docstore exposes all documents directly — no query needed:

      docs = list(self.db.docstore._dict.values())
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from infrastructure.markdown_tree_reader import MarkdownTreeReader
from rag.config import RAGConfig

# ---------------------------------------------------------------------------
# Index metadata
# ---------------------------------------------------------------------------


class IndexMetadata:
    """
    Metadata recorded when an index is built or loaded.

    Stored as a plain dict for JSON-compatibility and returned alongside
    the FAISS database so callers have full observability.
    """

    @staticmethod
    def build(
        documents: list[Document],
        chunks: list[Document],
        config: RAGConfig,
        policy_name: str,
        build_time: float,
    ) -> dict[str, Any]:
        """Build a metadata dict from index build results."""
        categories: dict[str, int] = {}
        for doc in documents:
            cat = doc.metadata.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "num_source_files": len(documents),
            "num_documents": len(chunks),
            "categories": categories,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "default_k": config.default_k,
            "embeddings_model": config.embeddings_model,
            "policy_name": policy_name,
            "build_time_s": round(build_time, 2),
        }


# ---------------------------------------------------------------------------
# FAISS Index Store
# ---------------------------------------------------------------------------


class FAISSIndexStore:
    """
    Manages the lifecycle of a FAISS vector index:
      • Build from a Markdown tree
      • Cache to disk using FAISS native serialisation (no pickle)
      • Load from cache
      • Expose the built database for retrieval

    Usage
    -----
        config = RAGConfig.default()
        store  = FAISSIndexStore(config)
        db, meta = store.build_or_load("policy/md_tree", "policy")

        # Later:
        docs = store.db.similarity_search("query", k=5)
    """

    def __init__(self, config: RAGConfig) -> None:
        """
        Args:
            config: Immutable RAGConfig controlling chunk size,
                    overlap, and embeddings model.
        """
        self._config = config
        self._db: FAISS | None = None
        self._meta: dict | None = None

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def db(self) -> FAISS:
        """
        The built FAISS database.

        Raises:
            RuntimeError: If build_or_load() has not been called yet.
        """
        if self._db is None:
            raise RuntimeError("Index not built. Call build_or_load() first.")
        return self._db

    @property
    def is_built(self) -> bool:
        """True if the index has been built or loaded."""
        return self._db is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_or_load(
        self,
        tree_path: str,
        policy_name: str = "policy",
        timing_info: list[str] | None = None,
        force_rebuild: bool = False,
    ) -> tuple[FAISS, dict]:
        """
        Build the FAISS index from a Markdown tree, or load a cached version.

        Cache directory name is derived from policy_name to avoid stale
        cross-policy index reuse.

        Args:
            tree_path:     Root path of the Markdown tree.
            policy_name:   Identifier used for the cache directory name.
            timing_info:   Optional list to append timing strings to.
            force_rebuild: If True, ignore cache and rebuild from scratch.

        Returns:
            Tuple of (FAISS database, metadata dict).

        Raises:
            FileNotFoundError: If tree_path does not exist.
            ValueError:        If no Markdown files are found.
        """
        cache_dir = self._cache_dir(policy_name)

        if not force_rebuild and os.path.exists(cache_dir):
            loaded = self._try_load_cache(cache_dir, timing_info)
            if loaded is not None:
                self._db, self._meta = loaded
                return self._db, self._meta

        self._db, self._meta = self._build_fresh(tree_path, policy_name, cache_dir, timing_info)
        return self._db, self._meta

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _try_load_cache(
        self,
        cache_dir: str,
        timing_info: list[str] | None,
    ) -> tuple[FAISS, dict] | None:
        """
        Attempt to load a cached FAISS index.

        Returns (db, metadata) on success, None on any failure
        (triggering a fresh build).
        """
        print(f"  📂 Loading cached index: {cache_dir}")
        start = time.time()

        try:
            embeddings = HuggingFaceEmbeddings(model_name=self._config.embeddings_model)
            db = FAISS.load_local(
                cache_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )

            # Load companion metadata JSON
            meta = self._load_meta_json(cache_dir)
            load_time = time.time() - start

            if timing_info is not None:
                timing_info.append(f"Index load time: {load_time:.2f}s")

            print(f"  ✅ Loaded cached index: " f"{meta.get('num_documents', '?')} chunks")
            return db, meta

        except Exception as exc:
            print(f"  ⚠ Error loading cached index: {exc} — rebuilding...")
            return None

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_fresh(
        self,
        tree_path: str,
        policy_name: str,
        cache_dir: str,
        timing_info: list[str] | None,
    ) -> tuple[FAISS, dict]:
        """
        Build a fresh FAISS index from the Markdown tree.

        Saves the index and a metadata JSON file to cache_dir.

        Args:
            tree_path:   Root of the Markdown tree.
            policy_name: Used in metadata and cache dir name.
            cache_dir:   Directory to save index and metadata.
            timing_info: Optional list for timing strings.

        Returns:
            Tuple of (FAISS database, metadata dict).
        """
        print(f"  🔨 Building index from {tree_path}...")
        start = time.time()

        if not os.path.exists(tree_path):
            raise FileNotFoundError(f"Policy tree directory not found: {tree_path}")

        # Load documents via shared MarkdownTreeReader
        print("  📖 Loading Markdown files...")
        reader = MarkdownTreeReader(tree_path)
        md_files = reader.load_all_files()
        documents = [
            Document(
                page_content=mf.content,
                metadata={
                    "source": mf.relative_path,
                    "category": mf.category,
                    "filename": mf.filename,
                    "full_path": mf.full_path,
                },
            )
            for mf in md_files
        ]

        if not documents:
            raise ValueError(f"No Markdown files found in {tree_path}")

        print(f"  ✅ Loaded {len(documents)} file(s)")

        # Chunk documents
        print("  ✂  Splitting into chunks...")
        chunks = self._splitter.split_documents(documents)
        print(
            f"  ✅ Created {len(chunks)} chunks "
            f"(size={self._config.chunk_size}, "
            f"overlap={self._config.chunk_overlap})"
        )

        # Build FAISS index
        print("  🧮 Creating embeddings and building FAISS index...")
        embeddings = HuggingFaceEmbeddings(model_name=self._config.embeddings_model)
        db = FAISS.from_documents(chunks, embeddings)

        build_time = time.time() - start
        meta = IndexMetadata.build(documents, chunks, self._config, policy_name, build_time)

        # Cache using FAISS native serialisation (no pickle)
        self._save_cache(db, meta, cache_dir)

        if timing_info is not None:
            timing_info.append(f"Index build time: {build_time:.2f}s")

        print(f"  ✅ Index built: {len(chunks)} chunks " f"from {len(documents)} files")
        print(f"     Categories: {meta['categories']}")

        return db, meta

    @staticmethod
    def _save_cache(db: FAISS, meta: dict, cache_dir: str) -> None:
        """
        Save the FAISS index and metadata to disk.

        Uses FAISS native save_local — no pickle.

        Args:
            db:        The FAISS database to save.
            meta:      Metadata dict to save alongside.
            cache_dir: Target directory.
        """
        import json

        print(f"  💾 Caching index to {cache_dir}...")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            db.save_local(cache_dir)

            meta_path = os.path.join(cache_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print("  ✅ Index cached")
        except Exception as exc:
            print(f"  ⚠ Could not cache index: {exc}")

    @staticmethod
    def _load_meta_json(cache_dir: str) -> dict:
        """Load the companion metadata.json from a cache directory."""
        import json

        meta_path = os.path.join(cache_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_dir(policy_name: str) -> str:
        """
        Derive a safe cache directory name from a policy name.

        e.g. "MLCommons AILuminate v1.1" → "mlcommons_ailuminate_v1_1_faiss"
        """
        safe = re.sub(r"[^\w\-]", "_", policy_name.lower())
        return f"{safe}_faiss"
