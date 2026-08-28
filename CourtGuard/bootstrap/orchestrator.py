"""
Bootstrap Orchestrator

Merges the duplicated run_bootstrap() functions from main.py and
pmpd_main.py into a single, unified class.

All five bootstrap stages are run in order, skipping any that are
already complete according to BootstrapStateManager.

Stage overview
──────────────
  Stage 0 : PolicyIngester  — PDF → Markdown tree
  Stage 1 : RAGTuner        — Markdown tree → RAG parameters
  Stage 2 : RAGPipeline     — build FAISS index
  Stage 3 : PromptGenerator — Markdown tree → role prompts (JSON)
  Stage 4 : PMPDParser      — Markdown tree → PMPD database

Design
──────
  The orchestrator owns the APIClient internally and rotates it when
  stages encounter APIError.  Callers receive the final client via
  BootstrapResult so the debate loop can reuse it.

  Each stage is a private method that returns True on success or False
  on unrecoverable failure.  A False from any halt-stage (0, 2) causes
  the orchestrator to return BootstrapResult(success=False).
  Soft-failure stages (1, 3, 4) fall back to defaults and continue.
"""

from __future__ import annotations

import os
import re
import glob
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bootstrap.artifact_archiver import ArchiveSnapshot, BootstrapArtifactArchiver
from bootstrap.state_manager import BootstrapState, BootstrapStateManager
from core.exceptions import PMPDScalabilityError
from core.pmpd import PMPD
from infrastructure.api_key_manager import APIKeyManager
from infrastructure.bootstrap_tracker import BootstrapTracker
from infrastructure.config import ModelConfig, PathConfig, RAGDefaults
from infrastructure.pmpd_repository import PMPDRepository
from evaluation.output_mapper import OutputMapper

if TYPE_CHECKING:
    from infrastructure.api_client import APIClient
    from rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """
    Output of a completed bootstrap run.

    Attributes
    ----------
    success  : True if all halt-stages completed successfully.
    pipeline : Built RAGPipeline, or None on failure.
    pmpd_db  : Populated PMPD database, or None on failure.
    client   : The APIClient active at the end of bootstrap
               (may have been rotated from the starting key).
    """

    success: bool
    pipeline: RAGPipeline | None
    pmpd_db: PMPD | None
    client: APIClient | None


# ---------------------------------------------------------------------------
# Bootstrap Orchestrator
# ---------------------------------------------------------------------------


class BootstrapOrchestrator:
    """
    Runs the CourtGuard bootstrap pipeline.

    Manages stage sequencing, caching, and API key rotation internally.
    Callers only need to call run() and inspect BootstrapResult.

    Usage
    -----
        key_manager  = APIKeyManager("api_keys.txt")
        orchestrator = BootstrapOrchestrator(key_manager)
        result       = orchestrator.run(force=False, force_pmpd=False)

        if result.success:
            run_debate(result.pipeline, result.pmpd_db, result.client)
    """

    def __init__(
        self,
        key_manager: APIKeyManager,
        paths: PathConfig | None = None,
        model_config: ModelConfig | None = None,
        output_mapper: OutputMapper | None = None,
    ) -> None:
        """
        Args:
            key_manager:  Initialized APIKeyManager with a key already set.
            paths:        PathConfig. Defaults to PathConfig.default().
            model_config: ModelConfig. Defaults to ModelConfig.default().
        """
        self._key_manager = key_manager
        self._paths = paths or PathConfig.default()
        self._model_config = model_config
        self._output_mapper = output_mapper or OutputMapper(OutputMapper.DEFAULT_LABELS)
        self._state_manager = BootstrapStateManager(self._paths)
        self._archiver = BootstrapArtifactArchiver(self._paths)
        self._client: Any | None = None
        self._rag_pipeline: Any | None = None
        self._pmpd_db: PMPD | None = None
        self._bootstrap_tracker: BootstrapTracker | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        eval_mode: str = "pmpd",
        force: bool = False,
        force_pmpd: bool = False,
        pmpd_k: int = 5,
        parse_only: bool = False,
        skip_pmpd_scalability_check: bool = False,
        load_rag_for_pmpd: bool = False,
    ) -> BootstrapResult:
        """
        Execute all bootstrap stages, skipping completed ones.

        Args:
            eval_mode:  "rag" or "pmpd". Determines which pipelines are built.
            force:      Re-run all stages (--force-bootstrap).
            force_pmpd: Re-run Stage 4 only (--force-pmpd).
            pmpd_k:     Runtime PMPD retrieval count. Legacy callers also
                        reused this as the native PMPD scalability limit.
            skip_pmpd_scalability_check:
                        If True, bypass the native PMPD category-count guard.
            load_rag_for_pmpd:
                        If True, PMPD mode will try to load existing cached
                        RAG assets so PMPD retrieval can use FAISS when available.

        Returns:
            BootstrapResult with success flag, pipeline, pmpd_db, client.
        """
        if self._model_config is None:
            self._model_config = ModelConfig.default()

        from infrastructure.api_client import APIClient

        self._client = APIClient(api_key=self._key_manager.current_key)

        # Create a fresh tracker for this run
        self._bootstrap_tracker = BootstrapTracker()

        state = self._state_manager.load_state()

        if force:
            print("\nForce mode: resetting all bootstrap stages.")
            state.reset()
            self._state_manager.save_state(state)

        doc_path = self._state_manager.detect_policy_document()
        state = self._handle_new_document(state, doc_path)

        needs = self._compute_needs(state, doc_path, force, force_pmpd)

        # -- Stage 0: Ingestion -------------------------------------------
        if needs["ingestion"]:
            if not self._run_ingestion(doc_path, state):
                return self._failure()
        else:
            print("\nStage 0 (Ingestion): complete - skipping")

        # -- Stage 1: RAG Tuning ------------------------------------------
        if eval_mode == "pmpd":
            if load_rag_for_pmpd:
                rag_config = self._load_existing_rag_config_for_pmpd(state)
            else:
                print("\nStage 1 (RAG Tuning): skipping (decoupled for PMPD)")
                rag_config = None
        else:
            rag_config = self._run_rag_tuning_or_load(needs["rag_tuning"], state)

        # -- Stage 2: FAISS Index Build ------------------------------------
        if eval_mode == "pmpd":
            if load_rag_for_pmpd and rag_config is not None:
                policy_name = self._policy_name(doc_path)
                if not self._run_index_build(rag_config, policy_name):
                    print("  Warning: existing RAG assets could not be loaded for PMPD - using heuristic selector.")
            else:
                print("\nStage 2 (FAISS Build): skipping (decoupled for PMPD)")
        else:
            policy_name = self._policy_name(doc_path)
            if rag_config and not self._run_index_build(rag_config, policy_name):
                return self._failure()

        # -- Stage 3: Prompt Generation -----------------------------------
        if needs["prompts"]:
            self._run_prompt_generation(state)
        else:
            print("\nStage 3 (Prompt Generation): complete - skipping")

        # -- Stage 4: PMPD Parsing ----------------------------------------
        if eval_mode == "pmpd" or needs["pmpd"]:
            if needs["pmpd"]:
                self._run_pmpd_parsing(state)
            else:
                print("\nStage 4 (PMPD Parsing): complete - loading database")
                self._pmpd_db = self._load_pmpd_or_empty()

            if self._pmpd_db and eval_mode == "pmpd" and not parse_only:
                cat_count = len(self._pmpd_db.list_categories())
                if not skip_pmpd_scalability_check and cat_count > pmpd_k:
                    raise PMPDScalabilityError(
                        f"Policy contains {cat_count} categories (max configured limit is {pmpd_k}).\n"
                        f"PMPD cannot load this natively without breaking model context windows. \n"
                        f"You must initialize FAISS RAG to support this policy: "
                        f"Run with '--mode rag --parse-only' first.",
                        requires_faiss=True
                    )

        result = BootstrapResult(
            success=True,
            pipeline=self._rag_pipeline,
            pmpd_db=self._pmpd_db,
            client=self._client,
        )

        # Save bootstrap usage stats after all stages complete
        if self._bootstrap_tracker:
            self._bootstrap_tracker.print_summary()
            self._bootstrap_tracker.save(self._paths.bootstrap_stats_path)

        return result

    # ------------------------------------------------------------------
    # Cache Teardown
    # ------------------------------------------------------------------

    def archive_bootstrap_artifacts(
        self,
        label: str = "manual",
        include_pmpd: bool = True,
        include_rag: bool = True,
    ) -> ArchiveSnapshot | None:
        """Archive the current bootstrap artifacts without deleting them."""
        return self._archiver.archive(
            label=label,
            include_pmpd=include_pmpd,
            include_rag=include_rag,
        )

    def clean_all(self, archive_label: str = "clean_all") -> ArchiveSnapshot | None:
        """Archive all current bootstrap artifacts, then remove PMPD and RAG outputs."""
        snapshot = self.archive_bootstrap_artifacts(
            label=archive_label,
            include_pmpd=True,
            include_rag=True,
        )
        self.clean_pmpd(archive=False)
        self.clean_rag(archive=False)
        return snapshot

    def clean_pmpd(
        self,
        archive: bool = True,
        archive_label: str = "clean_pmpd",
    ) -> ArchiveSnapshot | None:
        """
        Wipe all PMPD-dependent artifacts and reset related state flags.

        Deletes:
          - pmpd_store.json      (Stage 4 output)
          - generated_prompts.json (Stage 3 output)
          - policy/md_tree/       (Stage 0 output — re-ingestion is required
                                   so Stage 4 does not re-parse stale markdown)

        The bootstrap state flags for ingestion, pmpd, and prompts are all
        reset so the next orchestrator run re-executes all three stages.
        """
        snapshot = None
        if archive:
            snapshot = self.archive_bootstrap_artifacts(
                label=archive_label,
                include_pmpd=True,
                include_rag=False,
            )

        state = self._state_manager.load_state()
        if os.path.exists(self._paths.pmpd_db_path):
            os.remove(self._paths.pmpd_db_path)
            print(f"  Deleted: {self._paths.pmpd_db_path}")
        if os.path.exists(self._paths.generated_prompts):
            os.remove(self._paths.generated_prompts)
            print(f"  Deleted: {self._paths.generated_prompts}")
        if os.path.exists(self._paths.markdown_tree_dir):
            shutil.rmtree(self._paths.markdown_tree_dir)
            print(f"  Deleted: {self._paths.markdown_tree_dir}/")
        state.pmpd_parsed = False
        state.prompts_generated = False
        state.ingested = False
        self._state_manager.save_state(state)
        return snapshot

    def clean_rag(
        self,
        archive: bool = True,
        archive_label: str = "clean_rag",
    ) -> ArchiveSnapshot | None:
        """
        Wipe all RAG-dependent artifacts and reset related state flags.

        Deletes:
          - Any directory matching [safe_name]_faiss (Stage 2 output)
          - policy/md_tree/ (Stage 0 output — re-ingestion is required
                             so Stage 2 does not index stale markdown)

        The bootstrap state flags for ingestion and rag_tuned are both
        reset so the next orchestrator run re-executes stages 0, 1, and 2.
        """
        snapshot = None
        if archive:
            snapshot = self.archive_bootstrap_artifacts(
                label=archive_label,
                include_pmpd=False,
                include_rag=True,
            )

        state = self._state_manager.load_state()

        # 1. Targeted cleanup: Detect current document to find its specific index
        doc_path = self._state_manager.detect_policy_document()
        if doc_path:
            p_name = self._policy_name(doc_path)
            safe_name = re.sub(r"[^\w\-]", "_", p_name.lower())
            specific_faiss_dir = f"{safe_name}_faiss"
            if os.path.exists(specific_faiss_dir):
                shutil.rmtree(specific_faiss_dir)
                print(f"  Deleted specific index: {specific_faiss_dir}/")

        # 2. Thorough cleanup: Search for any other FAISS indices in the root
        # This ensures that even if the policy filename changed, old indices are nuked.
        for faiss_dir in glob.glob("*_faiss"):
            if os.path.isdir(faiss_dir):
                shutil.rmtree(faiss_dir)
                print(f"  Deleted legacy index: {faiss_dir}/")

        if os.path.exists(self._paths.markdown_tree_dir):
            shutil.rmtree(self._paths.markdown_tree_dir)
            print(f"  Deleted: {self._paths.markdown_tree_dir}/")

        state.rag_tuned = False
        state.ingested = False
        self._state_manager.save_state(state)
        return snapshot

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_ingestion(
        self,
        doc_path: str | None,
        state: BootstrapState,
    ) -> bool:
        """Stage 0 — PDF → Markdown tree. Returns False on unrecoverable failure."""
        from infrastructure.api_client import APIError
        from ingestion.policy_ingester import PolicyIngester

        print(f"\n{'='*60}")
        print("STAGE 0: Policy Ingestion  (PDF → Markdown tree)")
        print(f"{'='*60}")

        while True:
            try:
                ingester = PolicyIngester(self._client, self._model_config)
                result = ingester.ingest(doc_path, self._paths.markdown_tree_dir)
                state.ingested = True
                state.policy_file = doc_path
                self._state_manager.save_state(state)
                print(
                    f"  ✅ Ingestion complete: {result['section_count']} files, "
                    f"{len(result['categories'])} categories"
                )
                return True

            except APIError as exc:
                print(f"  ⚠ API error: {exc}")
                if not self._rotate_client():
                    print("  ✖ All keys exhausted — cannot continue.")
                    return False

            except Exception as exc:
                print(f"  ✖ Ingestion failed: {exc}")
                return False

    def _run_rag_tuning_or_load(
        self,
        needs_tuning: bool,
        state: BootstrapState,
    ) -> dict:
        """Stage 1 — RAG parameter tuning. Always returns a valid config dict."""
        from infrastructure.api_client import APIError
        from rag.tuner import RAGTuner

        if not needs_tuning:
            print("\n⏭  Stage 1 (RAG Tuning): complete — loading config")
            config = self._state_manager.load_rag_config()
            if config is not None:
                return config
            print("  ⚠ Config file missing — re-running tuning.")

        print(f"\n{'='*60}")
        print("STAGE 1: RAG Parameter Tuning")
        print(f"{'='*60}")

        while True:
            try:
                tuner = RAGTuner(self._client, self._model_config)
                config = tuner.analyze(self._paths.markdown_tree_dir)
                self._state_manager.save_rag_config(
                    config.to_dict()
                    if hasattr(config, "to_dict")
                    else {
                        "chunk_size": config.chunk_size,
                        "chunk_overlap": config.chunk_overlap,
                        "k": config.default_k,
                        "rationale": config.rationale,
                    }
                )
                state.rag_tuned = True
                self._state_manager.save_state(state)
                print(
                    f"  ✅ chunk={config.chunk_size}, "
                    f"overlap={config.chunk_overlap}, k={config.default_k}"
                )
                return {
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "k": config.default_k,
                    "rationale": config.rationale,
                }

            except APIError as exc:
                print(f"  ⚠ API error: {exc}")
                if not self._rotate_client():
                    print("  ⚠ All keys exhausted — using default RAG config.")
                    return self._rag_fallback("API keys exhausted during tuning.")

            except Exception as exc:
                print(f"  ⚠ RAG tuning error: {exc} — using defaults.")
                return self._rag_fallback(str(exc))

    def _run_index_build(self, rag_config: dict, policy_name: str) -> bool:
        """Stage 2 — FAISS index build. Returns False on failure."""
        from rag.pipeline import RAGPipeline

        print(f"\n{'='*60}")
        print("STAGE 2: FAISS Index Build")
        print(f"{'='*60}")

        try:
            pipeline = RAGPipeline.from_policy_config(rag_config)
            _, meta = pipeline.build_or_load_index(
                tree_path=self._paths.markdown_tree_dir,
                policy_name=policy_name,
            )
            self._rag_pipeline = pipeline
            print(f"  ✅ Index ready: {meta['num_documents']} chunks")
            return True

        except Exception as exc:
            print(f"  ✖ Index build failed: {exc}")
            return False

    def _run_prompt_generation(self, state: BootstrapState) -> None:
        """Stage 3 — role prompt generation. Falls back to defaults on failure."""
        from infrastructure.api_client import APIError
        from prompts.generator import PromptGenerator

        print(f"\n{'='*60}")
        print("STAGE 3: Prompt Generation  (Markdown tree → role prompts)")
        print(f"{'='*60}")

        while True:
            try:
                generator = PromptGenerator(
                    self._client,
                    model_config=self._model_config,
                    output_mapper=self._output_mapper,
                    output_path=self._paths.generated_prompts,
                    bootstrap_tracker=self._bootstrap_tracker,
                )
                result = generator.generate(self._paths.markdown_tree_dir)
                state.prompts_generated = True
                self._state_manager.save_state(state)
                print(f"  ✅ Prompts generated for: {result['policy_title']}")
                return

            except APIError as exc:
                print(f"  ⚠ API error: {exc}")
                if not self._rotate_client():
                    print("  ⚠ All keys exhausted — debate will use fallback prompts.")
                    return

            except Exception as exc:
                print(f"  ⚠ Prompt generation error: {exc} — using fallbacks.")
                return

    def _load_existing_rag_config_for_pmpd(
        self,
        state: BootstrapState,
    ) -> dict | None:
        """
        Load cached RAG settings for PMPD runtimes if they already exist.

        This is intentionally non-destructive: PMPD mode should not kick off a
        fresh tuning job here. It only reuses previously built assets.
        """
        if not state.rag_tuned or not os.path.exists(self._paths.rag_config_file):
            print("\n⏭  Stage 1 (RAG Tuning): no cached RAG config for PMPD — using heuristic selector")
            return None

        print("\n⏭  Stage 1 (RAG Tuning): loading existing RAG config for PMPD runtime")
        config = self._state_manager.load_rag_config()
        if config is None:
            print("  ⚠ Cached RAG config missing or unreadable — using heuristic selector")
            return None
        return config

    def _run_pmpd_parsing(self, state: BootstrapState) -> None:
        """Stage 4 — PMPD database build. Falls back to existing/empty on failure."""
        from infrastructure.api_client import APIError
        from pmpd_pipeline.parser import PMPDParser

        print(f"\n{'='*60}")
        print("STAGE 4: PMPD Parsing  (Markdown tree → PMPD database)")
        print(f"{'='*60}")

        while True:
            try:
                parser = PMPDParser(
                    self._client,
                    self._model_config,
                    output_mapper=self._output_mapper,
                    bootstrap_tracker=self._bootstrap_tracker,
                )
                db = parser.parse(
                    tree_path=self._paths.markdown_tree_dir,
                    db_path=self._paths.pmpd_db_path,
                )
                state.pmpd_parsed = True
                self._state_manager.save_state(state)
                self._pmpd_db = db
                print(
                    f"  ✅ PMPD built: {len(db.categories)} categories, "
                    f"saved to {self._paths.pmpd_db_path}"
                )
                return

            except APIError as exc:
                print(f"  ⚠ API error during PMPD parsing: {exc}")
                if not self._rotate_client():
                    print(
                        "  ⚠ All keys exhausted — loading existing PMPD or " "running without it."
                    )
                    self._pmpd_db = self._load_pmpd_or_empty()
                    return

            except Exception as exc:
                print(f"  ⚠ PMPD parsing error: {exc}")
                self._pmpd_db = self._load_pmpd_or_empty()
                return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotate_client(self) -> bool:
        """
        Rotate to the next API key and rebuild the client.

        Returns True if rotation succeeded, False if all keys exhausted.
        """
        result = self._key_manager.rotate()
        if not result.success:
            return False
        print(
            f"\n{'='*60}\n"
            f"Rate limit hit — switching to API key #{result.key_number}\n"
            f"{'='*60}\n"
        )
        from infrastructure.api_client import APIClient

        self._client = APIClient(api_key=result.key_value)
        return True

    def _load_pmpd_or_empty(self) -> PMPD:
        """Load existing PMPD from disk, or return a blank one."""
        repo = PMPDRepository(self._paths.pmpd_db_path)
        db = repo.load()
        if db.categories:
            print(f"  Loaded existing PMPD: {len(db.categories)} categories")
        else:
            print("  ℹ No existing PMPD found — creating empty database.")
        return db

    def _failure(self) -> BootstrapResult:
        """Return a failed BootstrapResult."""
        return BootstrapResult(
            success=False,
            pipeline=None,
            pmpd_db=None,
            client=self._client,
        )

    def _handle_new_document(
        self,
        state: BootstrapState,
        doc_path: str | None,
    ) -> BootstrapState:
        """Detect a newly dropped PDF and reset all stages if found."""
        if doc_path and state.policy_file and doc_path != state.policy_file:
            import os as _os

            print(
                f"\n🆕 New policy PDF detected: "
                f"{_os.path.basename(doc_path)} — resetting bootstrap."
            )
            state.reset()
            self._state_manager.save_state(state)
        return state

    def _compute_needs(
        self,
        state: BootstrapState,
        doc_path: str | None,
        force: bool,
        force_pmpd: bool,
    ) -> dict[str, bool]:
        """Compute which stages need to run based on state and flags."""
        p = self._paths
        return {
            "ingestion": (force or not state.ingested or not os.path.exists(p.markdown_tree_dir)),
            "rag_tuning": (force or not state.rag_tuned or not os.path.exists(p.rag_config_file)),
            "prompts": (
                force or not state.prompts_generated or not os.path.exists(p.generated_prompts)
            ),
            "pmpd": (
                force or force_pmpd or not state.pmpd_parsed or not os.path.exists(p.pmpd_db_path)
            ),
        }

    @staticmethod
    def _rag_fallback(reason: str) -> dict:
        """Return default RAG config dict."""
        d = RAGDefaults.default()
        return {
            "chunk_size": d.chunk_size,
            "chunk_overlap": d.chunk_overlap,
            "k": d.k,
            "rationale": f"Fallback defaults ({reason}).",
        }

    @staticmethod
    def _policy_name(doc_path: str | None) -> str:
        """Derive policy name from PDF path, or use 'policy'."""
        if doc_path:
            return os.path.splitext(os.path.basename(doc_path))[0]
        return "policy"
