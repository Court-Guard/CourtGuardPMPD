"""
CourtGuard Evaluation Entry Point

Evaluates a JSON dataset in RAG or PMPD mode.

Usage
-----
    # RAG mode, range
    python evaluate.py --mode rag \
        --data_file "data/datasets/PKU_SafeRLHF_180.json" \
        --start_index 0 --end_index 9

    # PMPD mode, specific indexes
    python evaluate.py --mode pmpd \
        --data_file "data/datasets/PKU_SafeRLHF_180.json" \
        --indexes "0,3,7,12"

    # Mixed range + individual indexes
    python evaluate.py --mode pmpd \
        --data_file "data/datasets/PKU_SafeRLHF_180.json" \
        --indexes "0-5,9,11"

Bootstrap is run automatically on first use (same as main.py).
Subsequent runs reuse the cached FAISS index and PMPD database.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import sys
import time
from typing import Any

from evaluation.cli import parse_eval_args
from infrastructure.api_key_manager import APIKeyManager
from infrastructure.config import EvaluationConfig, ModelConfig, PathConfig


# Lazily populated so tests can patch evaluate.BootstrapOrchestrator directly.
BootstrapOrchestrator: Any | None = None


def _resolve_output_path(output_dir: str, output_file: str) -> str:
    """Resolve an explicit output path the same way ResultWriter does."""
    if os.path.isabs(output_file):
        return output_file
    if os.path.dirname(output_file):
        return output_file
    return os.path.join(output_dir, output_file)


def _load_completed_indexes(output_path: str) -> set[int]:
    """Load already-written dataset indexes from an existing JSON results file."""
    if not os.path.exists(output_path):
        return set()

    try:
        with open(output_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()

    completed: set[int] = set()
    if not isinstance(payload, list):
        return completed

    for row in payload:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        if isinstance(idx, int):
            completed.add(idx)
    return completed


def _get_bootstrap_orchestrator_class() -> Any:
    """Resolve the orchestrator class lazily while preserving test patch points."""
    global BootstrapOrchestrator
    if BootstrapOrchestrator is None:
        from bootstrap.orchestrator import BootstrapOrchestrator as _BootstrapOrchestrator

        BootstrapOrchestrator = _BootstrapOrchestrator
    return BootstrapOrchestrator


def main() -> None:
    args = parse_eval_args()
    paths = PathConfig.default()

    # Cache management first so teardown does not prompt for API keys.
    if args.clean_all or args.clean_pmpd or args.clean_rag or args.archive_bootstrap:
        bootstrap_orchestrator_cls = _get_bootstrap_orchestrator_class()

        print(f"\n{'=' * 60}")
        print(f"CourtGuard Cleanup - mode: {args.mode.upper()}")
        print(f"{'=' * 60}\n")
        print("Executing bootstrap archive / cache management...")

        dummy_mgr = APIKeyManager(keys_file=args.keys_file)
        orchestrator = bootstrap_orchestrator_cls(
            key_manager=dummy_mgr,
            paths=paths,
            model_config=None,
        )

        if args.archive_bootstrap:
            snapshot = orchestrator.archive_bootstrap_artifacts(label=args.archive_label)
            if snapshot:
                print(f"\nArchive complete: {snapshot.snapshot_dir}")
            else:
                print("\nArchive complete: no bootstrap artifacts were present.")

        if args.clean_all:
            snapshot = orchestrator.clean_all(archive_label=args.archive_label or "clean_all")
            if snapshot:
                print(f"\nArchived before cleanup: {snapshot.snapshot_dir}")
        else:
            if args.clean_pmpd:
                snapshot = orchestrator.clean_pmpd(archive_label=args.archive_label or "clean_pmpd")
                if snapshot:
                    print(f"\nArchived PMPD artifacts: {snapshot.snapshot_dir}")
            if args.clean_rag:
                snapshot = orchestrator.clean_rag(archive_label=args.archive_label or "clean_rag")
                if snapshot:
                    print(f"\nArchived RAG artifacts: {snapshot.snapshot_dir}")

        print("\nCleanup/archive complete. Exiting.")
        sys.exit(0)

    eval_config = EvaluationConfig.from_env()
    model_config = ModelConfig.default()
    bootstrap_orchestrator_cls = _get_bootstrap_orchestrator_class()
    from infrastructure.api_client import APIError

    from evaluation.output_mapper import OutputMapper

    output_mapper = OutputMapper.from_config(
        eval_config,
        cli_labels=getattr(args, "output_labels", None),
        cli_default_label=getattr(args, "default_output_label", None),
        cli_error_label=getattr(args, "error_output_label", None),
    )

    print(f"\n{'=' * 60}")
    print(f"CourtGuard Evaluation - mode: {args.mode.upper()}")
    print(f"{'=' * 60}\n")

    try:
        key_manager = APIKeyManager(keys_file=args.keys_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    available = key_manager.available_key_numbers
    print("Available API keys:")
    for num in available:
        print(f"  {num}")

    while True:
        try:
            choice = input("\nEnter the API key number to start with: ").strip()
            key_manager.set_key(int(choice))
            print(f"Starting with API key #{choice}")
            break
        except (KeyError, ValueError):
            print(f"Invalid choice. Please select from: {available}")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    start = time.time()
    orchestrator = bootstrap_orchestrator_cls(
        key_manager=key_manager,
        paths=paths,
        model_config=model_config,
        output_mapper=output_mapper,
    )

    bootstrap_result = orchestrator.run(
        eval_mode=args.mode,
        force=args.force_bootstrap,
        force_pmpd=args.force_pmpd,
        pmpd_k=args.pmpd_k,
        parse_only=args.parse_only,
        skip_pmpd_scalability_check=(args.mode == "pmpd" and args.pmpd_runtime == "multivote"),
        load_rag_for_pmpd=False,
    )

    if not bootstrap_result.success:
        print("\nBootstrap failed. Check errors above and retry.")
        sys.exit(1)

    print(f"\nBootstrap completed in {time.time() - start:.1f}s")

    if args.parse_only:
        print("\n--parse-only specified. Pipelines built. Halting before dataset evaluation.")
        sys.exit(0)

    from data.dataset_loader import DatasetLoader
    from debate.policy_debate import PolicyDebate
    from debate.pmpd_debate import PMPDDebate
    from debate.pmpd_multivote import PMPDMultiVote
    from evaluation.input_mapper import InputMapper
    from evaluation.pmpd_mode import PMPDEvaluator
    from evaluation.result_writer import ResultWriter
    from evaluation.runner import EvaluationRunner
    from pmpd_pipeline.assembler import PMPDAssembler
    from prompts.loader import PromptLoader

    try:
        loader = DatasetLoader(args.data_file)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    try:
        if args.indexes:
            records = loader.load_indexes(args.indexes)
        else:
            records = loader.load_range(args.start_index, args.end_index)
    except ValueError as exc:
        print(f"Index error: {exc}")
        sys.exit(1)

    if args.resume_output:
        output_path = _resolve_output_path(args.output_dir, args.output_file)
        completed_indexes = _load_completed_indexes(output_path)
        if completed_indexes:
            original_count = len(records)
            records = [record for record in records if record.index not in completed_indexes]
            skipped = original_count - len(records)
            print(f"\nResume mode: loaded {len(completed_indexes)} completed indexes from {output_path}")
            print(f"Resume mode: skipping {skipped} already-written records")
        else:
            print(f"\nResume mode: no completed indexes found in {output_path}")

    print(f"\nRecords to evaluate: {len(records)}")

    loader_prompts = PromptLoader(paths.generated_prompts).load()
    debate = PolicyDebate(
        api_client=bootstrap_result.client,
        prompts=loader_prompts.to_prompts_dict(),
        output_mapper=output_mapper,
    )

    if args.mode == "rag":
        from evaluation.rag_mode import RAGEvaluator

        if bootstrap_result.pipeline is None:
            print("RAG pipeline not available. Bootstrap may have failed.")
            sys.exit(1)
        evaluator = RAGEvaluator(
            pipeline=bootstrap_result.pipeline,
            debate=debate,
            model=model_config.debate_model,
            k=args.rag_k,
            output_mapper=output_mapper,
        )
    else:
        if bootstrap_result.pmpd_db is None:
            print("PMPD database not available. Bootstrap may have failed.")
            sys.exit(1)

        if bootstrap_result.pipeline:
            assembler = PMPDAssembler.with_rag(
                bootstrap_result.pmpd_db,
                bootstrap_result.pipeline,
            )
        else:
            assembler = PMPDAssembler.with_heuristic(bootstrap_result.pmpd_db)

        overrides: dict[str, object] = {}
        if getattr(args, "max_rounds", None) is not None:
            overrides["max_rounds"] = args.max_rounds
        if getattr(args, "tie_winner", None) is not None:
            overrides["tie_winner"] = args.tie_winner
        if getattr(args, "prompt_style", None) is not None:
            overrides["prompt_style"] = args.prompt_style
        if getattr(args, "use_judge", False):
            overrides["use_judge"] = True

        if overrides:
            eval_config = dataclasses.replace(eval_config, **overrides)

        input_mapper = InputMapper.from_config(
            eval_config,
            cli_fields=getattr(args, "input_fields", None),
        )

        if args.pmpd_runtime == "multivote":
            pmpd_debate = PMPDMultiVote(
                api_client=bootstrap_result.client,
                db=bootstrap_result.pmpd_db,
                judge_model=model_config.debate_model,
                judge_count=args.multivote_judges,
                output_mapper=output_mapper,
            )
        else:
            pmpd_debate = PMPDDebate(
                api_client=bootstrap_result.client,
                db=bootstrap_result.pmpd_db,
                max_rounds=eval_config.max_rounds,
                attacker_model=model_config.debate_model,
                defender_model=model_config.debate_model,
                output_mapper=output_mapper,
            )

        evaluator = PMPDEvaluator(
            assembler=assembler,
            debate=pmpd_debate,
            model=model_config.debate_model,
            k=args.pmpd_k,
            inspect=args.inspect_pmpd,
            input_mapper=input_mapper,
            output_mapper=output_mapper,
        )

    writer_mode = (
        f"{args.mode}_{args.pmpd_runtime}"
        if args.mode == "pmpd" and args.pmpd_runtime != "debate"
        else args.mode
    )
    writer = ResultWriter(
        mode=writer_mode,
        model=model_config.debate_model,
        dataset_name=loader.dataset_name,
        output_dir=args.output_dir,
        output_file=args.output_file,
    )

    try:
        runner = EvaluationRunner(evaluator, writer)
        runner.run(records)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print(f"  Partial results saved to: {writer.output_path}")
        sys.exit(0)
    except APIError as exc:
        print(f"\nCritical API error: {exc}")
        print(f"  Partial results saved to: {writer.output_path}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        print(f"  Partial results saved to: {writer.output_path}")
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    main()
