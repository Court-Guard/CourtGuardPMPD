"""
RAG Tuner

Analyses a policy Markdown tree and recommends optimal RAG pipeline
parameters via a single LLM call.

Extracted from the RAGTuner class in rag_pipeline.py.

Changes from original
─────────────────────
  • ParameterGrid is injected — no longer hardcoded module constants.
    Tests can pass a narrow grid; production uses ParameterGrid.default().
  • LLMRetryClient replaces the inline API call — consistent retry behaviour.
  • MarkdownTreeReader replaces the duplicated _sample_tree() method.
  • JSONExtractor replaces the inline JSON parsing block.
  • Returns a typed RAGConfig instead of a raw dict.
"""

from __future__ import annotations

from infrastructure.api_client import APIClient
from infrastructure.config import ModelConfig
from infrastructure.json_extractor import JSONExtractor
from infrastructure.llm_retry_client import LLMRetryClient, RetryConfig
from infrastructure.markdown_tree_reader import MarkdownTreeReader
from rag.config import ParameterGrid, RAGConfig

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------

_TUNING_SYSTEM_MSG = (
    "You are an expert in Retrieval-Augmented Generation systems. "
    "You analyse document characteristics and recommend optimal chunking "
    "parameters. You always respond with valid JSON and nothing else."
)


# ---------------------------------------------------------------------------
# RAG Tuner
# ---------------------------------------------------------------------------


class RAGTuner:
    """
    Analyses a Markdown policy tree and recommends RAG pipeline parameters.

    Reads a sample of each category file, reasons about document density
    and structure, then returns a RAGConfig with chunk_size, chunk_overlap,
    and k values chosen from the injected ParameterGrid.

    Usage
    -----
        tuner  = RAGTuner(api_client)
        config = tuner.analyze("policy/md_tree")

        pipeline = RAGPipeline(config)
    """

    def __init__(
        self,
        api_client: APIClient,
        model_config: ModelConfig | None = None,
        parameter_grid: ParameterGrid | None = None,
    ) -> None:
        """
        Args:
            api_client:     Initialized APIClient instance.
            model_config:   ModelConfig for model selection.
                            Defaults to ModelConfig.default().
            parameter_grid: Allowed parameter grids for snapping LLM output.
                            Defaults to ParameterGrid.default().
        """
        cfg = model_config or ModelConfig.default()
        self._grid = parameter_grid or ParameterGrid.default()
        self._retry = LLMRetryClient(api_client, RetryConfig.bootstrap())
        self._model = cfg.bootstrap_model
        self._json = JSONExtractor()

    def analyze(self, tree_path: str) -> RAGConfig:
        """
        Analyse the Markdown tree and return recommended RAG parameters.

        The LLM must choose values from the injected ParameterGrid.
        Any out-of-grid values are automatically snapped to the nearest
        allowed value.

        Args:
            tree_path: Root path of the generated Markdown tree.

        Returns:
            RAGConfig with optimal chunk_size, chunk_overlap, k, rationale.
            Falls back to RAGConfig.default() on any failure.
        """
        print("  📊 Sampling Markdown tree for RAG tuning analysis...")

        reader = MarkdownTreeReader(tree_path)
        sample = reader.sample_tree(max_chars_per_file=1500)

        if not sample.strip():
            print("  ⚠ No Markdown content found — using defaults.")
            return RAGConfig.default()

        prompt = self._build_prompt(sample)

        try:
            raw = self._retry.call(
                prompt,
                _TUNING_SYSTEM_MSG,
                self._model,
                max_tokens=256,
                temperature=0.1,
            )
            parsed = self._json.extract(raw)

            if not parsed:
                print("  ⚠ Could not parse tuning JSON — using defaults.")
                return RAGConfig.default()

            return self._grid.validate_and_snap(parsed)

        except Exception as exc:
            print(f"  ⚠ RAG tuning error: {exc} — using defaults.")
            return RAGConfig.default()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, sample: str) -> str:
        """Build the RAG tuning analysis prompt."""
        grid = self._grid
        return (
            f"Analyse the policy document sample below and recommend "
            f"RAG chunking parameters.\n\n"
            f"ALLOWED VALUES (you MUST choose from these only):\n"
            f"- chunk_size: {list(grid.chunk_sizes)}\n"
            f"- overlap_ratio: {list(grid.overlap_ratios)}  "
            f"(chunk_overlap = round(chunk_size * overlap_ratio))\n"
            f"- k (retrieval count): {list(grid.k_values)}\n\n"
            f"REASONING GUIDE:\n"
            f"- Dense legal prose with long paragraphs → larger chunk_size "
            f"(1024-1536), overlap_ratio 0.20-0.30\n"
            f"- Short enumeration-style paragraphs or checklists → smaller "
            f"chunk_size (512-768), overlap_ratio 0.10-0.20\n"
            f"- Many distinct hazard categories → higher k (6-8) to ensure "
            f"cross-category coverage\n"
            f"- Few broad categories → lower k (3-5) is sufficient\n\n"
            f"DOCUMENT SAMPLE:\n{sample[:6000]}\n\n"
            f"Respond ONLY with this JSON object and nothing else:\n"
            f"{{\n"
            f'  "chunk_size": <chosen from allowed list>,\n'
            f'  "overlap_ratio": <chosen from allowed list>,\n'
            f'  "k": <chosen from allowed list>,\n'
            f'  "rationale": "<one or two sentences explaining your choices>"\n'
            f"}}"
        )
