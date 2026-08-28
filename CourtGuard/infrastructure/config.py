"""
Application Configuration

Centralises all path constants and model identifiers that were previously
duplicated across main.py and pmpd_main.py, and scattered as module-level
constants in policy_ingester.py, prompt_generator.py, and pmpd_parser.py.

Classes
-------
  PathConfig      -- all filesystem paths used by the bootstrap pipeline
  ModelConfig     -- canonical LLM model identifiers
  RAGDefaults     -- default RAG parameters (mirrors rag_pipeline constants)
  EvaluationConfig -- runtime evaluation settings (taxonomy, debate params)

Design notes
------------
  Both dataclasses use frozen=True so instances are immutable after creation.
  PathConfig.default() returns the standard layout; callers that need
  non-standard paths (e.g. tests) can construct their own instance.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some shells
    find_dotenv = None
    load_dotenv = None

def _load_env_fallback() -> None:
    """Load a nearby .env file even when python-dotenv is unavailable."""
    current = Path(__file__).resolve()
    candidate_dirs = [
        Path.cwd(),
        current.parent,
        current.parent.parent,
        current.parent.parent.parent,
    ]

    seen: set[Path] = set()
    for directory in candidate_dirs:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        env_path = resolved / ".env"
        if not env_path.exists():
            continue

        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                os.environ.setdefault(key, value)
        except OSError:
            pass
        return


if find_dotenv and load_dotenv:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:  # pragma: no cover - depends on local environment layout
        _load_env_fallback()
else:  # pragma: no cover - depends on local environment packages
    _load_env_fallback()

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathConfig:
    """
    Filesystem paths used throughout the CourtGuard bootstrap pipeline.

    Previously these were copy-pasted module-level constants in both
    main.py and pmpd_main.py:

        POLICY_INPUT_DIR  = "policy"
        MARKDOWN_TREE_DIR = "policy/md_tree"
        RAG_CONFIG_FILE   = "rag_config.json"
        GENERATED_PROMPTS = "generated_prompts.py"  (now .json)
        BOOTSTRAP_STATE   = ".bootstrap_state.json"
        PMPD_DB_PATH      = "pmpd_store.json"

    Attributes
    ----------
    policy_input_dir     : Directory where the source PDF is placed
    markdown_tree_dir    : Output directory for PolicyIngester Markdown tree
    rag_config_file      : JSON file storing RAGTuner output parameters
    generated_prompts    : JSON file storing PromptGenerator output
                           (was generated_prompts.py -- changed to .json
                            to eliminate the code-generation smell)
    bootstrap_state      : JSON file tracking which bootstrap stages are done
    pmpd_db_path         : JSON file storing the PMPD database
    api_keys_file        : Default path to the API keys file
    bootstrap_stats_path : JSON file storing per-run bootstrap usage stats
                           (api calls, token counts, latency per stage)
    bootstrap_archive_dir: Directory where old bootstrap artifacts are snapshotted
                           before cleanup / re-bootstrap
    """

    policy_input_dir:     str = "policy"
    markdown_tree_dir:    str = "policy/md_tree"
    rag_config_file:      str = "rag_config.json"
    generated_prompts:    str = "generated_prompts.json"
    bootstrap_state:      str = ".bootstrap_state.json"
    pmpd_db_path:         str = "pmpd_store.json"
    api_keys_file:        str = "api_keys.txt"
    bootstrap_stats_path: str = "bootstrap_stats.json"
    bootstrap_archive_dir: str = "../archive/pmpd_artifacts"

    @classmethod
    def default(cls) -> "PathConfig":
        """Return the standard path configuration for production use."""
        return cls()


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """
    Canonical LLM model identifiers used across all pipeline modules.

    Both models MUST be set via environment variables — there are no
    hardcoded defaults.  If a variable is missing the code raises
    a ValueError at startup so the operator notices immediately.

    Environment Variables
    ---------------------
    COURTGUARD_BOOTSTRAP_MODEL : Model for PDF structure planning, section
                                  routing, prompt generation, and PMPD extraction.
    COURTGUARD_DEBATE_MODEL    : Model for all three debate roles
                                  (Attacker, Defender, Judge).

    Attributes
    ----------
    bootstrap_model : str — set via COURTGUARD_BOOTSTRAP_MODEL
    debate_model    : str — set via COURTGUARD_DEBATE_MODEL
    """

    bootstrap_model: str
    debate_model: str

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Build ModelConfig from environment variables. Raises ValueError if missing."""
        bootstrap = os.getenv("COURTGUARD_BOOTSTRAP_MODEL")
        debate = os.getenv("COURTGUARD_DEBATE_MODEL")

        missing = []
        if not bootstrap:
            missing.append("COURTGUARD_BOOTSTRAP_MODEL")
        if not debate:
            missing.append("COURTGUARD_DEBATE_MODEL")

        if missing:
            raise ValueError(
                f"Required environment variable(s) not set: {', '.join(missing)}. "
                f"Set them in your .env file or shell environment."
            )

        return cls(bootstrap_model=bootstrap, debate_model=debate)

    @classmethod
    def default(cls) -> "ModelConfig":
        """Return model configuration from environment. Raises if env vars missing."""
        return cls.from_env()


# ---------------------------------------------------------------------------
# RAG defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RAGDefaults:
    """
    Default RAG pipeline parameters.

    Mirrors the DEFAULT_* constants in rag_pipeline.py but provides them
    as a typed, importable config object so other modules (e.g. bootstrap
    orchestrator fallback) do not need to import from rag_pipeline directly.

    Attributes
    ----------
    chunk_size    : Default maximum chunk size in characters
    chunk_overlap : Default overlap between consecutive chunks
    k             : Default number of documents to retrieve
    """

    chunk_size: int = 1024
    chunk_overlap: int = 256
    k: int = 5

    @classmethod
    def default(cls) -> "RAGDefaults":
        """Return the standard RAG defaults."""
        return cls()


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlamaCloudConfig:
    """
    Configuration for LlamaCloud (LlamaParse) services.

    Attributes
    ----------
    api_key  : API key from XXXX
    base_url : Regional API endpoint (US or EU)
    """

    api_key: str | None = None
    base_url: str = "XXXX"

    @classmethod
    def from_env(cls) -> "LlamaCloudConfig":
        """Build LlamaCloudConfig from environment variables."""
        return cls(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            base_url=os.getenv(
                "LLAMA_CLOUD_BASE_URL", "XXXX"
            ),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Runtime configuration for an evaluation run.

    Controls input field mapping, prompt style, debate parameters,
    and PMPD-specific behaviour.

    All values are read from environment variables with CLI override
    support implemented in evaluation/cli.py and evaluation/input_mapper.py.

    Attributes
    ----------
    input_fields          : Ordered list of JSON field names to extract from
                            each dataset record and present to agents.
                            e.g. ["user_prompt", "target model response"]
                                 ["oldtext", "newtext", "diff"]
    use_env_fields        : If True, input_fields are read from
                            COURTGUARD_INPUT_FIELDS env var, ignoring CLI.
                            If False, CLI --input-fields takes priority.
                            Controlled by COURTGUARD_USE_ENV_FIELDS=true|false
    output_labels         : Configured output taxonomy labels (e.g. SAFE,UNSAFE)
    use_env_output_labels : If True, output_labels are read from env var.
    default_output_label  : The first/compliant label.
    error_output_label    : The last/flagged label.
    prompt_style          : "harmony" or "standard".
                            Controlled by COURTGUARD_PROMPT_STYLE env var.
    max_rounds            : Maximum debate rounds before tie resolution.
                            Controlled by COURTGUARD_MAX_ROUNDS env var.
    tie_winner            : Who wins when all rounds exhausted.
                            Controlled by COURTGUARD_TIE_WINNER env var.
    use_judge             : If True, Judge agent is called on tie instead
                            of tie_winner policy.
                            Controlled by COURTGUARD_USE_JUDGE=true|false
    use_harmony_roles     : If True, sends developer+system message split.
                            If False, collapses both into system only.
                            Controlled by COURTGUARD_USE_HARMONY_ROLES=true|false
    """

    input_fields:          tuple[str, ...] = ("user_prompt", "target model response")
    use_env_fields:        bool             = False
    output_labels:         tuple[str, ...] = ("SAFE", "UNSAFE")
    use_env_output_labels: bool             = False
    default_output_label:  str              = "SAFE"
    error_output_label:    str              = "UNSAFE"
    prompt_style:          str              = "standard"
    max_rounds:            int              = 2
    tie_winner:            str              = "defender"
    use_judge:             bool             = False
    use_harmony_roles:     bool             = False

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """
        Build EvaluationConfig from environment variables.

        Called by evaluate.py before CLI args are applied.
        CLI args can override individual fields on top of this.
        """
        # Input fields -- parse comma-separated env var
        env_fields_raw = os.getenv("COURTGUARD_INPUT_FIELDS", "")
        if env_fields_raw.strip():
            env_fields = tuple(
                f.strip() for f in env_fields_raw.split(",") if f.strip()
            )
        else:
            env_fields = ("user_prompt", "target model response")

        use_env_fields = os.getenv(
            "COURTGUARD_USE_ENV_FIELDS", "false"
        ).lower() == "true"

        env_output_labels_raw = os.getenv("COURTGUARD_OUTPUT_LABELS", "")
        if env_output_labels_raw.strip():
            output_labels = tuple(
                label.strip().upper()
                for label in env_output_labels_raw.split(",")
                if label.strip()
            )
        else:
            output_labels = ("SAFE", "UNSAFE")

        use_env_output_labels = os.getenv(
            "COURTGUARD_USE_ENV_OUTPUT_LABELS", "false"
        ).lower() == "true"

        default_output_label = os.getenv(
            "COURTGUARD_DEFAULT_OUTPUT_LABEL",
            output_labels[0] if output_labels else "SAFE",
        ).strip().upper()
        error_output_label = os.getenv(
            "COURTGUARD_ERROR_OUTPUT_LABEL",
            output_labels[-1] if output_labels else "UNSAFE",
        ).strip().upper()

        prompt_style = os.getenv("COURTGUARD_PROMPT_STYLE", "standard").lower()
        max_rounds   = int(os.getenv("COURTGUARD_MAX_ROUNDS", "2"))
        tie_winner   = os.getenv("COURTGUARD_TIE_WINNER", "defender").lower()
        use_judge    = os.getenv("COURTGUARD_USE_JUDGE", "false").lower() == "true"
        use_harmony_roles = (
            os.getenv("COURTGUARD_USE_HARMONY_ROLES", "false").lower() == "true"
        )

        return cls(
            input_fields=          env_fields,
            use_env_fields=        use_env_fields,
            output_labels=         output_labels,
            use_env_output_labels= use_env_output_labels,
            default_output_label=  default_output_label,
            error_output_label=    error_output_label,
            prompt_style=          prompt_style,
            max_rounds=            max_rounds,
            tie_winner=            tie_winner,
            use_judge=             use_judge,
            use_harmony_roles=     use_harmony_roles,
        )

    @classmethod
    def default(cls) -> "EvaluationConfig":
        """Return default evaluation config (reads from env)."""
        return cls.from_env()
