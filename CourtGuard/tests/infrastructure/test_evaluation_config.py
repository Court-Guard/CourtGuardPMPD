"""
Tests for infrastructure/config.py — EvaluationConfig

Create as: tests/infrastructure/test_evaluation_config.py

Covers:
  - from_env() reads all env vars correctly
  - Default values when env vars not set
  - use_env_fields flag
  - input_fields parsing from comma-separated string
  - max_rounds int parsing
  - prompt_style and tie_winner string values
  - use_harmony_roles bool parsing
"""

import os
import pytest
from unittest.mock import patch

from infrastructure.config import EvaluationConfig


class TestEvaluationConfigDefaults:
    def test_default_input_fields(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove relevant keys if present
            for key in ["COURTGUARD_INPUT_FIELDS", "COURTGUARD_USE_ENV_FIELDS"]:
                os.environ.pop(key, None)
            config = EvaluationConfig.from_env()
        assert config.input_fields == ("user_prompt", "target model response")

    def test_default_prompt_style(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COURTGUARD_PROMPT_STYLE", None)
            config = EvaluationConfig.from_env()
        assert config.prompt_style == "standard"

    def test_default_max_rounds(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COURTGUARD_MAX_ROUNDS", None)
            config = EvaluationConfig.from_env()
        assert config.max_rounds == 2

    def test_default_tie_winner(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COURTGUARD_TIE_WINNER", None)
            config = EvaluationConfig.from_env()
        assert config.tie_winner == "defender"

    def test_default_use_judge_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COURTGUARD_USE_JUDGE", None)
            config = EvaluationConfig.from_env()
        assert config.use_judge is False

    def test_default_use_harmony_roles_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COURTGUARD_USE_HARMONY_ROLES", None)
            config = EvaluationConfig.from_env()
        assert config.use_harmony_roles is False

    def test_default_output_labels(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in [
                "COURTGUARD_OUTPUT_LABELS",
                "COURTGUARD_USE_ENV_OUTPUT_LABELS",
                "COURTGUARD_DEFAULT_OUTPUT_LABEL",
                "COURTGUARD_ERROR_OUTPUT_LABEL",
            ]:
                os.environ.pop(key, None)
            config = EvaluationConfig.from_env()
        assert config.output_labels == ("SAFE", "UNSAFE")
        assert config.default_output_label == "SAFE"
        assert config.error_output_label == "UNSAFE"


class TestEvaluationConfigFromEnv:
    def test_custom_input_fields(self):
        with patch.dict(os.environ, {
            "COURTGUARD_INPUT_FIELDS": "oldtext,newtext,diff"
        }):
            config = EvaluationConfig.from_env()
        assert config.input_fields == ("oldtext", "newtext", "diff")

    def test_single_input_field(self):
        with patch.dict(os.environ, {
            "COURTGUARD_INPUT_FIELDS": "message"
        }):
            config = EvaluationConfig.from_env()
        assert config.input_fields == ("message",)

    def test_input_fields_strips_whitespace(self):
        with patch.dict(os.environ, {
            "COURTGUARD_INPUT_FIELDS": " user_prompt , target_model_response "
        }):
            config = EvaluationConfig.from_env()
        assert config.input_fields == ("user_prompt", "target_model_response")

    def test_use_env_fields_true(self):
        with patch.dict(os.environ, {
            "COURTGUARD_USE_ENV_FIELDS": "true"
        }):
            config = EvaluationConfig.from_env()
        assert config.use_env_fields is True

    def test_use_env_fields_false(self):
        with patch.dict(os.environ, {
            "COURTGUARD_USE_ENV_FIELDS": "false"
        }):
            config = EvaluationConfig.from_env()
        assert config.use_env_fields is False

    def test_prompt_style_harmony(self):
        with patch.dict(os.environ, {
            "COURTGUARD_PROMPT_STYLE": "harmony"
        }):
            config = EvaluationConfig.from_env()
        assert config.prompt_style == "harmony"

    def test_max_rounds_custom(self):
        with patch.dict(os.environ, {
            "COURTGUARD_MAX_ROUNDS": "4"
        }):
            config = EvaluationConfig.from_env()
        assert config.max_rounds == 4

    def test_tie_winner_attacker(self):
        with patch.dict(os.environ, {
            "COURTGUARD_TIE_WINNER": "attacker"
        }):
            config = EvaluationConfig.from_env()
        assert config.tie_winner == "attacker"

    def test_use_judge_true(self):
        with patch.dict(os.environ, {
            "COURTGUARD_USE_JUDGE": "true"
        }):
            config = EvaluationConfig.from_env()
        assert config.use_judge is True

    def test_use_harmony_roles_true(self):
        with patch.dict(os.environ, {
            "COURTGUARD_USE_HARMONY_ROLES": "true"
        }):
            config = EvaluationConfig.from_env()
        assert config.use_harmony_roles is True

    def test_output_labels_parsed(self):
        with patch.dict(os.environ, {
            "COURTGUARD_OUTPUT_LABELS": "allow,review,block"
        }):
            config = EvaluationConfig.from_env()
        assert config.output_labels == ("ALLOW", "REVIEW", "BLOCK")

    def test_use_env_output_labels_true(self):
        with patch.dict(os.environ, {
            "COURTGUARD_USE_ENV_OUTPUT_LABELS": "true"
        }):
            config = EvaluationConfig.from_env()
        assert config.use_env_output_labels is True

    def test_custom_default_and_error_output_labels(self):
        with patch.dict(os.environ, {
            "COURTGUARD_OUTPUT_LABELS": "allow,review,block",
            "COURTGUARD_DEFAULT_OUTPUT_LABEL": "allow",
            "COURTGUARD_ERROR_OUTPUT_LABEL": "block",
        }):
            config = EvaluationConfig.from_env()
        assert config.default_output_label == "ALLOW"
        assert config.error_output_label == "BLOCK"

    def test_frozen_dataclass(self):
        config = EvaluationConfig()
        with pytest.raises(Exception):
            config.max_rounds = 99  # type: ignore[misc]


class TestEvaluationConfigAllEnvVars:
    def test_full_env_config(self):
        with patch.dict(os.environ, {
            "COURTGUARD_INPUT_FIELDS":      "oldtext,newtext,diff",
            "COURTGUARD_USE_ENV_FIELDS":    "true",
            "COURTGUARD_OUTPUT_LABELS":     "allow,review,block",
            "COURTGUARD_USE_ENV_OUTPUT_LABELS": "true",
            "COURTGUARD_DEFAULT_OUTPUT_LABEL": "allow",
            "COURTGUARD_ERROR_OUTPUT_LABEL": "block",
            "COURTGUARD_PROMPT_STYLE":      "harmony",
            "COURTGUARD_MAX_ROUNDS":        "3",
            "COURTGUARD_TIE_WINNER":        "judge",
            "COURTGUARD_USE_JUDGE":         "true",
            "COURTGUARD_USE_HARMONY_ROLES": "true",
        }):
            config = EvaluationConfig.from_env()

        assert config.input_fields      == ("oldtext", "newtext", "diff")
        assert config.use_env_fields    is True
        assert config.output_labels     == ("ALLOW", "REVIEW", "BLOCK")
        assert config.use_env_output_labels is True
        assert config.default_output_label == "ALLOW"
        assert config.error_output_label == "BLOCK"
        assert config.prompt_style      == "harmony"
        assert config.max_rounds        == 3
        assert config.tie_winner        == "judge"
        assert config.use_judge         is True
        assert config.use_harmony_roles is True
