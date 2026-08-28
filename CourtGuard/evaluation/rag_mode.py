"""
RAG Mode Evaluator

Evaluates a single dataset record using the RAG pipeline.

RAG mode operation
──────────────────
  For each record:
    1. RAGPipeline.search_similar_documents(user_prompt) retrieves the
       top-k most relevant policy chunks from the FAISS index.
    2. The retrieved chunks are joined as a context string.
    3. System prompts loaded from generated_prompts.json are injected
       into PolicyDebate via set_prompts().
    4. PolicyDebate.run_debate() runs the full Attacker / Defender / Judge
       cycle with the retrieved context.

This evaluator is stateless per record — it takes the already-built
RAGPipeline and PolicyDebate from BootstrapOrchestrator and calls them
without constructing anything new.
"""

from __future__ import annotations

from data.dataset_loader import EvaluationRecord
from debate.policy_debate import PolicyDebate
from evaluation.output_mapper import OutputMapper
from evaluation.result_writer import EvaluationResult
from rag.pipeline import RAGPipeline


class RAGEvaluator:
    """
    Evaluates dataset records in RAG mode.

    Usage
    -----
        evaluator = RAGEvaluator(pipeline, debate, model="openai/gpt-oss-20b")
        result    = evaluator.evaluate(record)
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        debate: PolicyDebate,
        model: str = "",
        k: int = 5,
        output_mapper: OutputMapper | None = None,
    ) -> None:
        """
        Args:
            pipeline: Built RAGPipeline with loaded FAISS index.
            debate:   PolicyDebate instance with prompts already loaded.
            model:    Model identifier string (for result metadata).
            k:        Number of RAG documents to retrieve per query.
        """
        self._pipeline = pipeline
        self._debate = debate
        self._model = model
        self._k = k
        self._output_mapper = output_mapper or OutputMapper(OutputMapper.DEFAULT_LABELS)

    def evaluate(self, record: EvaluationRecord) -> EvaluationResult:
        """
        Evaluate one dataset record in RAG mode.

        Args:
            record: EvaluationRecord to evaluate.

        Returns:
            EvaluationResult with debate outcome and metadata.
        """
        print(f"\n  [RAG] Evaluating index {record.index}...")
        print(f"  Query    : {record.user_prompt[:80]}...")
        print(f"  Response : {record.target_model_response[:80]}...")

        # Retrieve policy context via RAG
        docs = self._pipeline.search_similar_documents(record.user_prompt, k=self._k)
        context = "\n\n".join(doc.page_content for doc in docs)
        print(f"  Context  : {len(docs)} chunks retrieved ({len(context)} chars)")

        # Run debate
        history, result_dict = self._debate.run_debate(
            user_prompt=record.user_prompt,
            response=record.target_model_response,
            context=context,
        )

        judge = result_dict.get("judge_evaluation", {})
        verdict = judge.get("label") or judge.get("severity", self._output_mapper.error_label)
        print(f"  Verdict  : {verdict} | Winner: {judge.get('winner', 'N/A')}")

        return EvaluationResult(
            index=record.index,
            user_prompt=record.user_prompt,
            target_model_response=record.target_model_response,
            category=record.category,
            label=record.label,
            is_safe=record.is_safe,
            courtguard_mode="rag",
            courtguard_model=self._model,
            courtguard_verdict=verdict,
            courtguard_winner=judge.get("winner", ""),
            courtguard_violated_categories=judge.get("violated_categories", []),
            courtguard_reasoning=judge.get("detailed_analysis", "")[:800],
            courtguard_severity_raw=verdict,
            debate_history=history,
            timing_info=result_dict.get("timing_info", []),
            api_error=bool(result_dict.get("api_error", False)),
            final_label=verdict,
            output_labels_used=list(self._output_mapper.labels),
            default_output_label=self._output_mapper.default_label,
            error_output_label=self._output_mapper.error_label,
        )
