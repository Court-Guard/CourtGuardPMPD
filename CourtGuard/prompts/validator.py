"""
Prompt Validator

Rule-based validation of generated debate prompts.

Extracted from _validate_prompts() in prompt_generator.py, where
validation rules were a hardcoded if-chain вЂ” adding a new rule required
modifying the function directly (OCP violation).

Each rule is a ValidationRule dataclass. Rules are injected into
PromptValidator at construction time, making the validator open for
extension and closed for modification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationRule:
    """
    A single prompt validation check.

    Attributes
    ----------
    role        : Which role this rule applies to: "attacker", "defender",
                  "judge", or "any" for all roles.
    description : Human-readable description of what is being checked
                  (used in issue reporting).
    check       : Callable(text: str, meta: dict) в†’ bool.
                  Returns True if the prompt PASSES the check.
                  meta contains "policy_title" and "category_names" (list).
    """

    role: str
    description: str
    check: Callable[[str, dict], bool]


def _mentions_policy_title(text: str, meta: dict) -> bool:
    title = meta.get("policy_title", "")
    return len(title) <= 5 or title.lower() in text.lower()


def _mentions_any_category(text: str, meta: dict) -> bool:
    cats = meta.get("category_names", [])
    return not cats or any(c.lower() in text.lower() for c in cats)


def _attacker_has_adversarial_framing(text: str, _meta: dict) -> bool:
    return any(
        w in text.lower() for w in ["violat", "non-compli", "breach", "prohibit", "against"]
    )


def _judge_has_verdict_instruction(text: str, _meta: dict) -> bool:
    return any(w in text.lower() for w in ["verdict", "label", "severity", "winner"])


DEFAULT_RULES: tuple[ValidationRule, ...] = (
    ValidationRule(
        role="any",
        description="mentions policy title",
        check=_mentions_policy_title,
    ),
    ValidationRule(
        role="any",
        description="references at least one hazard category",
        check=_mentions_any_category,
    ),
    ValidationRule(
        role="attacker",
        description="has adversarial framing",
        check=_attacker_has_adversarial_framing,
    ),
    ValidationRule(
        role="judge",
        description="has verdict/label instruction",
        check=_judge_has_verdict_instruction,
    ),
)


class PromptValidator:
    """
    Validates generated debate prompts against a set of injectable rules.

    Usage
    -----
        validator = PromptValidator()                    # default rules
        validator = PromptValidator(rules=my_rules)     # custom rules

        issues = validator.validate(prompts, meta)
        if issues:
            print(issues)
    """

    def __init__(
        self,
        rules: tuple[ValidationRule, ...] | None = None,
    ) -> None:
        """
        Args:
            rules: Tuple of ValidationRule objects to apply.
                   Defaults to DEFAULT_RULES if not provided.
        """
        self._rules = rules if rules is not None else DEFAULT_RULES

    def validate(
        self,
        prompts: dict[str, str],
        meta: dict,
    ) -> list[str]:
        """
        Run all rules against the given prompts.

        Args:
            prompts: Dict with keys "attacker", "defender", "judge".
            meta:    Dict with keys "policy_title" and "category_names".

        Returns:
            List of issue description strings. Empty list = all passed.
        """
        issues: list[str] = []

        for role in ("attacker", "defender", "judge"):
            text = prompts.get(role, "")
            for rule in self._rules:
                if rule.role not in (role, "any"):
                    continue
                if not rule.check(text, meta):
                    issues.append(f"{role}: {rule.description}")

        return issues
