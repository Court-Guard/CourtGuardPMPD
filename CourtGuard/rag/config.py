"""
RAG Configuration

Typed configuration dataclasses and parameter grids for the RAG pipeline.

Extracted from module-level constants in rag_pipeline.py:

    DEFAULT_CHUNK_SIZE    = 1024
    DEFAULT_CHUNK_OVERLAP = 256
    DEFAULT_K             = 5
    ALLOWED_CHUNK_SIZES   = [512, 768, 1024, 1536]
    ALLOWED_OVERLAP_RATIOS= [0.10, 0.20, 0.30]
    ALLOWED_K_VALUES      = [3, 4, 5, 6, 8]

Classes
───────
  RAGConfig      — immutable runtime config for one pipeline instance
  ParameterGrid  — the allowed value grids the LLM must choose from
                   (injectable into RAGTuner so tests can use narrow grids)
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# RAG runtime configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RAGConfig:
    """
    Immutable configuration for a RAGPipeline instance.

    Attributes
    ----------
    chunk_size       : Maximum size of text chunks in characters.
    chunk_overlap    : Overlap between consecutive chunks in characters.
    default_k        : Default number of documents to retrieve per query.
    embeddings_model : HuggingFace model name for text embeddings.
    rationale        : Human-readable explanation of why these values
                       were chosen (populated by RAGTuner).
    """

    chunk_size: int = 1024
    chunk_overlap: int = 256
    default_k: int = 5
    embeddings_model: str = "sentence-transformers/all-mpnet-base-v2"
    rationale: str = ""

    @classmethod
    def default(cls) -> RAGConfig:
        """Return the standard default configuration."""
        return cls()

    @classmethod
    def from_tuner_dict(cls, d: dict) -> RAGConfig:
        """
        Build a RAGConfig from a RAGTuner result dict.

        Args:
            d: Dict with keys chunk_size, chunk_overlap, k, rationale.

        Returns:
            RAGConfig instance.
        """
        return cls(
            chunk_size=int(d.get("chunk_size", 1024)),
            chunk_overlap=int(d.get("chunk_overlap", 256)),
            default_k=int(d.get("k", 5)),
            rationale=d.get("rationale", ""),
        )


# ---------------------------------------------------------------------------
# Parameter grid (injectable into RAGTuner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterGrid:
    """
    The allowed value grids the LLM tuner must choose from.

    Kept as an injectable dataclass so:
      • Production uses the full grids defined below.
      • Tests can inject a narrow grid (e.g. only one value each)
        to make assertions deterministic without real LLM calls.
      • Adding a new allowed value requires no changes to RAGTuner.

    Attributes
    ----------
    chunk_sizes    : Allowed chunk_size values (characters).
    overlap_ratios : Allowed overlap fractions of chunk_size.
    k_values       : Allowed retrieval counts.
    """

    chunk_sizes: tuple[int, ...] = (512, 768, 1024, 1536)
    overlap_ratios: tuple[float, ...] = (0.10, 0.20, 0.30)
    k_values: tuple[int, ...] = (3, 4, 5, 6, 8)

    @classmethod
    def default(cls) -> ParameterGrid:
        """Return the standard production parameter grid."""
        return cls()

    def snap(self, value: float, allowed: tuple) -> float:
        """
        Snap a value to the nearest element in an allowed tuple.

        Args:
            value:   The value to snap.
            allowed: The allowed values to snap to.

        Returns:
            The nearest allowed value.
        """
        return min(allowed, key=lambda x: abs(x - value))

    def validate_and_snap(self, config: dict) -> RAGConfig:
        """
        Validate an LLM-returned config dict and snap any out-of-grid
        values to the nearest allowed value.

        Args:
            config: Dict with chunk_size, overlap_ratio, k, rationale.

        Returns:
            Valid RAGConfig with all values within the grid.
        """
        chunk_size = int(config.get("chunk_size", 1024))
        overlap_ratio = float(config.get("overlap_ratio", 0.25))
        k = int(config.get("k", 5))
        rationale = config.get("rationale", "No rationale provided.")

        chunk_size = int(self.snap(chunk_size, self.chunk_sizes))
        overlap_ratio = float(self.snap(overlap_ratio, self.overlap_ratios))
        k = int(self.snap(k, self.k_values))
        chunk_overlap = round(chunk_size * overlap_ratio)

        return RAGConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            default_k=k,
            rationale=rationale,
        )
