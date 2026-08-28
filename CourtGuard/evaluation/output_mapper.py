"""
Output Mapper

Resolves the runtime output label taxonomy from environment variables and
CLI overrides, then provides shared helpers for prompts, parsers, and
result serialization.

The first configured label is treated as the default compliant label.
The configured error label is used for fail-closed fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputTaxonomy:
    """Resolved output taxonomy shared across runtime components."""

    labels: tuple[str, ...]
    default_label: str
    error_label: str

    @property
    def review_label(self) -> str | None:
        """Optional middle/review label if the taxonomy has 3+ labels."""
        if len(self.labels) >= 3:
            return self.labels[1]
        return None

    @property
    def allowed_labels_text(self) -> str:
        """Human-readable label list for prompt instructions."""
        return " | ".join(self.labels)


class OutputMapper:
    """
    Resolve and normalize runtime output labels.

    Priority mirrors InputMapper:
      1. If use_env_output_labels=True -> use env-configured labels
      2. If CLI labels provided         -> use CLI labels
      3. Otherwise                      -> use defaults

    The first configured label is treated as the default compliant label.
    The error label defaults to the final configured label.
    """

    DEFAULT_LABELS: tuple[str, ...] = ("SAFE", "UNSAFE")

    _DEFAULT_ALIASES = {
        "0": lambda self: self.default_label,
        "1": lambda self: self.error_label,
        "SAFE": lambda self: self.default_label,
        "COMPLIANT": lambda self: self.default_label,
        "ALLOW": lambda self: self.default_label,
        "ALLOWED": lambda self: self.default_label,
        "BENIGN": lambda self: self.default_label,
        "CLEAN": lambda self: self.default_label,
        "UNSAFE": lambda self: self.error_label,
        "VIOLATION": lambda self: self.error_label,
        "BLOCK": lambda self: self.error_label,
        "BLOCKED": lambda self: self.error_label,
        "HARMFUL": lambda self: self.error_label,
        "BORDERLINE": lambda self: self.review_label or self.error_label,
        "REVIEW": lambda self: self.review_label or self.error_label,
        "UNSURE": lambda self: self.review_label or self.error_label,
        "UNKNOWN": lambda self: self.error_label,
    }

    def __init__(
        self,
        labels: tuple[str, ...],
        default_label: str | None = None,
        error_label: str | None = None,
    ) -> None:
        normalized = self._normalize_labels(labels)
        if len(normalized) < 2:
            raise ValueError("Output taxonomy must contain at least 2 labels.")

        self._taxonomy = OutputTaxonomy(
            labels=normalized,
            default_label=(default_label or normalized[0]).strip().upper(),
            error_label=(error_label or normalized[-1]).strip().upper(),
        )

        if self._taxonomy.default_label not in normalized:
            raise ValueError(
                f"Default output label '{self._taxonomy.default_label}' must be in labels."
            )
        if self._taxonomy.error_label not in normalized:
            raise ValueError(
                f"Error output label '{self._taxonomy.error_label}' must be in labels."
            )

    @classmethod
    def from_config(
        cls,
        eval_config,
        cli_labels: str | None = None,
        cli_default_label: str | None = None,
        cli_error_label: str | None = None,
    ) -> "OutputMapper":
        """Resolve the output taxonomy from config plus optional CLI overrides."""
        if getattr(eval_config, "use_env_output_labels", False):
            labels = tuple(eval_config.output_labels)
            default_label = eval_config.default_output_label
            error_label = eval_config.error_output_label
            return cls(labels, default_label=default_label, error_label=error_label)

        if cli_labels is not None:
            parsed = cls._parse_labels_string(cli_labels)
            if parsed:
                return cls(
                    parsed,
                    default_label=cli_default_label or parsed[0],
                    error_label=cli_error_label or parsed[-1],
                )

        if tuple(getattr(eval_config, "output_labels", cls.DEFAULT_LABELS)) != cls.DEFAULT_LABELS:
            return cls(
                tuple(eval_config.output_labels),
                default_label=cli_default_label or eval_config.default_output_label,
                error_label=cli_error_label or eval_config.error_output_label,
            )

        return cls(
            cls.DEFAULT_LABELS,
            default_label=cli_default_label or eval_config.default_output_label,
            error_label=cli_error_label or eval_config.error_output_label,
        )

    @classmethod
    def from_labels_string(
        cls,
        labels_string: str,
        default_label: str | None = None,
        error_label: str | None = None,
    ) -> "OutputMapper":
        """Create directly from a comma-separated label string."""
        parsed = cls._parse_labels_string(labels_string) or cls.DEFAULT_LABELS
        return cls(parsed, default_label=default_label, error_label=error_label)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._taxonomy.labels

    @property
    def default_label(self) -> str:
        return self._taxonomy.default_label

    @property
    def error_label(self) -> str:
        return self._taxonomy.error_label

    @property
    def review_label(self) -> str | None:
        return self._taxonomy.review_label

    @property
    def allowed_labels_text(self) -> str:
        return self._taxonomy.allowed_labels_text

    def is_valid(self, label: str) -> bool:
        """True if the given label is part of the configured taxonomy."""
        return label.strip().upper() in self.labels

    def is_default(self, label: str) -> bool:
        """True if the given label is the default compliant label."""
        return self.normalize_label(label, fallback="") == self.default_label

    def normalize_label(self, raw_label: str, fallback: str | None = None) -> str:
        """
        Normalize a raw label into the configured taxonomy.

        Supports:
          - exact configured labels
          - legacy numeric PMPD labels 0/1
          - a small set of common safety aliases
        """
        candidate = (raw_label or "").strip().upper()
        if not candidate:
            return fallback or self.default_label

        if candidate in self.labels:
            return candidate

        alias = self._DEFAULT_ALIASES.get(candidate)
        if alias is not None:
            return alias(self)

        return fallback or self.error_label

    def compliant_label(self) -> str:
        """Return the configured default compliant label."""
        return self.default_label

    def flagged_label(self, preferred: str | None = None) -> str:
        """Return a non-default label for fail-closed or legacy binary cases."""
        if preferred:
            return self.normalize_label(preferred, fallback=self.error_label)
        return self.error_label

    @staticmethod
    def _normalize_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        normalized: list[str] = []
        for label in labels:
            value = label.strip().upper()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return tuple(normalized)

    @staticmethod
    def _parse_labels_string(s: str) -> tuple[str, ...] | None:
        """Parse a comma-separated output label string."""
        if not s or not s.strip():
            return None
        parts = tuple(label.strip().upper() for label in s.split(",") if label.strip())
        return parts if parts else None
