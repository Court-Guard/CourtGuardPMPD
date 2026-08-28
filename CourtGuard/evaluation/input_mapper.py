"""
Input Mapper

Resolves which JSON fields to extract from each dataset record and
presents them to debate agents in a consistent format.

This decouples CourtGuard from any specific dataset schema.
The same system evaluates:
  - Safety/jailbreak:  user_prompt + target_model_response
  - Wikipedia vandalism: oldtext + newtext + diff
  - Toxicity:          message
  - Any future task:   whatever fields you configure

Field resolution priority
──────────────────────────
  1. If COURTGUARD_USE_ENV_FIELDS=true → use COURTGUARD_INPUT_FIELDS
  2. If --input-fields CLI flag provided → use CLI value
  3. Default → ("user_prompt", "target_model_response")

This is controlled by EvaluationConfig and resolved here at runtime
so the rest of the system never needs to know where the config came from.

Output format to agents
────────────────────────
Agents receive a single formatted string combining all fields:

  FIELD_NAME_1:
  <value>

  FIELD_NAME_2:
  <value>

This is consistent regardless of how many fields are configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Resolved input
# ---------------------------------------------------------------------------

@dataclass
class ResolvedInput:
    """
    The evaluation input resolved from a dataset record.

    Attributes
    ----------
    fields      : Ordered dict of {field_name: field_value} as extracted
                  from the dataset record.
    formatted   : Single string combining all fields, ready for injection
                  into agent prompts.
    field_names : The field names that were requested (for metadata).
    missing     : Field names that were requested but not found in record.
    """

    fields:      dict[str, str]
    formatted:   str
    field_names: tuple[str, ...]
    missing:     list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True if all requested fields were found in the record."""
        return len(self.missing) == 0

    def get(self, field_name: str, default: str = "") -> str:
        """Retrieve a specific field value by name."""
        return self.fields.get(field_name, default)


# ---------------------------------------------------------------------------
# Input Mapper
# ---------------------------------------------------------------------------

class InputMapper:
    """
    Maps dataset record dicts to ResolvedInput for agent consumption.

    Instantiated once per evaluation run with the resolved field names.
    Call map(record_dict) for each record.

    Usage
    -----
        # From EvaluationConfig:
        mapper = InputMapper.from_config(eval_config, cli_fields)

        # Per record:
        resolved = mapper.map(record.raw)
        prompt_text = resolved.formatted
    """

    # Default fields used when no configuration is provided
    DEFAULT_FIELDS: tuple[str, ...] = ("user_prompt", "target_model_response")

    def __init__(self, field_names: tuple[str, ...]) -> None:
        """
        Args:
            field_names: Ordered tuple of JSON field names to extract.
        """
        self._field_names = field_names

    @classmethod
    def from_config(
        cls,
        eval_config,                    # EvaluationConfig
        cli_fields: str | None = None,  # Raw CLI --input-fields string
    ) -> "InputMapper":
        """
        Resolve which fields to use based on config priority rules.

        Priority:
          1. use_env_fields=True  → use eval_config.input_fields (from env)
          2. cli_fields provided  → parse and use CLI value
          3. Neither              → use DEFAULT_FIELDS

        Args:
            eval_config: EvaluationConfig instance (from infrastructure/config.py).
            cli_fields:  Raw --input-fields CLI string, e.g. "user_prompt,response".
                         None if not provided on CLI.

        Returns:
            InputMapper configured with the resolved field names.
        """
        if eval_config.use_env_fields:
            # Env takes priority — use whatever is in EvaluationConfig
            return cls(eval_config.input_fields)

        if cli_fields is not None:
            # CLI provided — parse it
            parsed = cls._parse_fields_string(cli_fields)
            if parsed:
                return cls(parsed)

        # Check if env has non-default fields even when use_env_fields=False
        # This handles the case where user set COURTGUARD_INPUT_FIELDS
        # but forgot to set COURTGUARD_USE_ENV_FIELDS=true
        if eval_config.input_fields != cls.DEFAULT_FIELDS:
            return cls(eval_config.input_fields)

        return cls(cls.DEFAULT_FIELDS)

    @classmethod
    def from_fields_string(cls, fields_string: str) -> "InputMapper":
        """
        Create an InputMapper directly from a comma-separated field string.

        Args:
            fields_string: e.g. "user_prompt,target_model_response"

        Returns:
            InputMapper with parsed fields.
        """
        parsed = cls._parse_fields_string(fields_string)
        return cls(parsed or cls.DEFAULT_FIELDS)

    # ------------------------------------------------------------------
    # Core mapping
    # ------------------------------------------------------------------

    def map(self, record: dict) -> ResolvedInput:
        """
        Extract configured fields from a raw dataset record dict.

        Handles both key variants for backwards compatibility:
          "target model response" (space) and "target_model_response" (underscore)

        Args:
            record: Raw dict from the dataset JSON.

        Returns:
            ResolvedInput with extracted values and formatted prompt text.
        """
        extracted: dict[str, str] = {}
        missing:   list[str]      = []

        for name in self._field_names:
            value = self._extract_field(record, name)
            if value is not None:
                extracted[name] = value
            else:
                extracted[name] = ""
                missing.append(name)

        formatted = self._format(extracted)

        return ResolvedInput(
            fields=      extracted,
            formatted=   formatted,
            field_names= self._field_names,
            missing=     missing,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """The field names this mapper is configured for."""
        return self._field_names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_field(record: dict, name: str) -> str | None:
        """
        Extract a field value, trying both underscore and space variants.

        Handles the "target model response" vs "target_model_response"
        inconsistency common in safety benchmark datasets.
        """
        # Direct match
        if name in record:
            val = record[name]
            return str(val) if val is not None else None

        # Try space variant (underscore → space)
        space_name = name.replace("_", " ")
        if space_name in record and space_name != name:
            val = record[space_name]
            return str(val) if val is not None else None

        # Try underscore variant (space → underscore)
        underscore_name = name.replace(" ", "_")
        if underscore_name in record and underscore_name != name:
            val = record[underscore_name]
            return str(val) if val is not None else None

        return None

    @staticmethod
    def _format(fields: dict[str, str]) -> str:
        """
        Format extracted fields into a single prompt-ready string.

        Format:
            FIELD_NAME_1:
            <value>

            FIELD_NAME_2:
            <value>

        Field names are uppercased and underscores replaced with spaces
        for readability in prompts.
        """
        parts: list[str] = []
        for name, value in fields.items():
            display_name = name.upper().replace("_", " ")
            parts.append(f"{display_name}:\n{value}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_fields_string(s: str) -> tuple[str, ...] | None:
        """
        Parse a comma-separated field names string.

        Returns None if the string is empty or produces no valid fields.
        """
        if not s or not s.strip():
            return None
        fields = tuple(f.strip() for f in s.split(",") if f.strip())
        return fields if fields else None