"""
Evaluation CLI

Argument parsing for evaluate.py.

Mirrors the LlamaGuard CLI convention:
    python evaluate.py --mode rag --data_file "data/datasets/file.json"
                       --start_index 0 --end_index 10

    python evaluate.py --mode pmpd --data_file "data/datasets/file.json"
                       --indexes "0,3,7,12"

If both --indexes and --start_index/--end_index are provided,
--indexes takes priority.
"""

from __future__ import annotations

import argparse


def parse_eval_args() -> argparse.Namespace:
    """Parse and return command-line arguments for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description=(
            "CourtGuard Evaluation Runner — " "evaluate a JSON dataset in RAG or PMPD mode"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # RAG mode, range
  python evaluate.py --mode rag \\
      --data_file "data/datasets/PKU_SafeRLHF_180.json" \\
      --start_index 0 --end_index 9

  # PMPD mode, specific indexes
  python evaluate.py --mode pmpd \\
      --data_file "data/datasets/PKU_SafeRLHF_180.json" \\
      --indexes "0,3,7,12"

  # PMPD mode, mixed range + individual
  python evaluate.py --mode pmpd \\
      --data_file "data/datasets/PKU_SafeRLHF_180.json" \\
      --indexes "0-5,9,11"

  # Force re-run bootstrap stages
  python evaluate.py --mode rag --data_file "..." \\
      --start_index 0 --end_index 5 --force-bootstrap
        """,
    )

    # ── Mode ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        required=True,
        choices=["rag", "pmpd"],
        help="Evaluation mode: 'rag' uses FAISS retrieval; "
        "'pmpd' uses structured policy modules",
    )

    # ── Dataset ───────────────────────────────────────────────────────
    parser.add_argument(
        "--data_file",
        required=False,
        help="Path to the JSON dataset file (required unless running --clean-*)",
    )

    # ── Index selection ───────────────────────────────────────────────
    index_group = parser.add_mutually_exclusive_group(required=False)

    index_group.add_argument(
        "--indexes",
        default=None,
        help="Index specification string. "
        "Comma-separated: '0,3,7' or range: '0-10' or mixed: '0-5,9,11'",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start of index range (inclusive). Used with --end_index.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End of index range (inclusive). Used with --start_index.",
    )

    # ── PMPD debate configuration ─────────────────────────────────────────
    parser.add_argument(
        "--input-fields",
        default=None,
        dest="input_fields",
        help=(
            "Comma-separated JSON field names to extract from each record. "
            "e.g. 'user_prompt,target_model_response' or 'oldtext,newtext,diff'. "
            "Overrides COURTGUARD_INPUT_FIELDS env var unless "
            "COURTGUARD_USE_ENV_FIELDS=true."
        ),
    )
    parser.add_argument(
        "--output-labels",
        default=None,
        dest="output_labels",
        help=(
            "Comma-separated output labels used by debate agents and result mapping. "
            "e.g. 'SAFE,UNSAFE' or 'ALLOW,REVIEW,BLOCK'. "
            "Overrides COURTGUARD_OUTPUT_LABELS unless "
            "COURTGUARD_USE_ENV_OUTPUT_LABELS=true."
        ),
    )
    parser.add_argument(
        "--default-output-label",
        default=None,
        dest="default_output_label",
        help=(
            "Compliant/default output label. "
            "Overrides COURTGUARD_DEFAULT_OUTPUT_LABEL."
        ),
    )
    parser.add_argument(
        "--error-output-label",
        default=None,
        dest="error_output_label",
        help=(
            "Fail-closed output label used on hard errors. "
            "Overrides COURTGUARD_ERROR_OUTPUT_LABEL."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        dest="max_rounds",
        help=(
            "Maximum debate rounds for PMPD mode. "
            "Overrides COURTGUARD_MAX_ROUNDS env var. (default: 2)"
        ),
    )
    parser.add_argument(
        "--tie-winner",
        default=None,
        dest="tie_winner",
        choices=["defender", "attacker", "judge"],
        help=(
            "Who wins when all rounds exhausted without confirmation. "
            "Overrides COURTGUARD_TIE_WINNER env var. (default: defender)"
        ),
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        dest="use_judge",
        help="Call Judge agent on tie instead of tie_winner policy.",
    )
    parser.add_argument(
        "--prompt-style",
        default=None,
        dest="prompt_style",
        choices=["standard", "harmony"],
        help=(
            "Prompt construction style. 'harmony' for gpt-oss models, "
            "'standard' for all others. "
            "Overrides COURTGUARD_PROMPT_STYLE env var."
        ),
    )
    parser.add_argument(
        "--pmpd-runtime",
        default="debate",
        dest="pmpd_runtime",
        choices=["debate", "multivote"],
        help=(
            "PMPD runtime strategy. 'debate' runs the selector/specialist flow. "
            "'multivote' runs a category gate first, then several scoped PMPD judges "
            "and aggregates them."
        ),
    )
    parser.add_argument(
        "--multivote-judges",
        type=int,
        default=3,
        dest="multivote_judges",
        help="Number of independent PMPD judges to run in multivote mode (default: 3).",
    )

    # ── Bootstrap flags ───────────────────────────────────────────────
    parser.add_argument(
        "--force-bootstrap",
        action="store_true",
        help="Re-run all bootstrap stages even if already complete",
    )
    parser.add_argument(
        "--force-pmpd",
        action="store_true",
        help="Re-run Stage 4 (PMPD parsing) only",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Execute pipelines and halt before attempting evaluation",
    )

    # ── Cache Management ──────────────────────────────────────────────
    parser.add_argument(
        "--clean-all",
        action="store_true",
        help="Purge all bootstrap states, FAISS indices, and Markdown trees",
    )
    parser.add_argument(
        "--clean-pmpd",
        action="store_true",
        help="Purge parsed PMPD database and clear PMPD state",
    )
    parser.add_argument(
        "--clean-rag",
        action="store_true",
        help="Purge FAISS indices and clear RAG state",
    )
    parser.add_argument(
        "--archive-bootstrap",
        action="store_true",
        help="Archive current bootstrap artifacts without deleting anything",
    )
    parser.add_argument(
        "--archive-label",
        default="manual",
        help="Human-readable label for archived bootstrap snapshot names",
    )

    # ── Miscellaneous ──────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Directory to write result JSON files (default: data/results)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Exact target JSON filepath to aggressively append results to. Bypasses timestamping.",
    )
    parser.add_argument(
        "--resume-output",
        action="store_true",
        dest="resume_output",
        help=(
            "Resume into an existing --output-file by skipping indexes that are already "
            "present in that JSON file."
        ),
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=5,
        help="Number of RAG documents to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--pmpd-k",
        type=int,
        default=20,
        help="Number of PMPD categories to select per query (default: 20)",
    )

    parser.add_argument(
        "--inspect-pmpd",
        action="store_true",
        help="Print assembled PMPD prompts and module contents before each evaluation",
    )
    parser.add_argument(
        "--keys-file",
        default="api_keys.txt",
        dest="keys_file",
        help="Path to the file containing API keys (default: api_keys.txt)",
    )

    args = parser.parse_args()

    # Validate logical routing combinations
    has_indexes = args.indexes is not None
    has_range = args.start_index is not None and args.end_index is not None
    has_start_only = args.start_index is not None and args.end_index is None
    has_end_only = args.end_index is not None and args.start_index is None

    if has_start_only or has_end_only:
        parser.error("--start_index and --end_index must be used together.")

    is_cleaner = args.clean_all or args.clean_pmpd or args.clean_rag or args.archive_bootstrap
    is_admin_bootstrap = args.parse_only or args.force_bootstrap or args.force_pmpd

    if not is_cleaner and not is_admin_bootstrap and not args.data_file:
        parser.error("--data_file is strictly required unless executing a cache teardown or administrative bootstrap (--parse-only, --force-bootstrap, --force-pmpd).")

    if not is_cleaner and not is_admin_bootstrap:
        if not has_indexes and not has_range:
            parser.error("Specify evaluation indexes using --indexes or range flags!")

    if args.resume_output and not args.output_file:
        parser.error("--resume-output requires --output-file.")

    return args
