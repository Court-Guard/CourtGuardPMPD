"""Tests for evaluation/output_mapper.py."""

from evaluation.output_mapper import OutputMapper
from infrastructure.config import EvaluationConfig


def test_defaults_are_binary_safe_unsafe():
    mapper = OutputMapper(OutputMapper.DEFAULT_LABELS)
    assert mapper.labels == ("SAFE", "UNSAFE")
    assert mapper.default_label == "SAFE"
    assert mapper.error_label == "UNSAFE"


def test_three_label_taxonomy_exposes_review_label():
    mapper = OutputMapper(("ALLOW", "REVIEW", "BLOCK"))
    assert mapper.review_label == "REVIEW"
    assert mapper.allowed_labels_text == "ALLOW | REVIEW | BLOCK"


def test_legacy_numeric_labels_normalize_into_configured_taxonomy():
    mapper = OutputMapper(("SAFE", "UNSAFE"))
    assert mapper.normalize_label("0") == "SAFE"
    assert mapper.normalize_label("1") == "UNSAFE"


def test_borderline_maps_to_review_label_when_present():
    mapper = OutputMapper(("SAFE", "BORDERLINE", "UNSAFE"))
    assert mapper.normalize_label("BORDERLINE") == "BORDERLINE"


def test_borderline_falls_closed_to_error_label_in_binary_taxonomy():
    mapper = OutputMapper(("SAFE", "UNSAFE"))
    assert mapper.normalize_label("BORDERLINE") == "UNSAFE"


def test_cli_labels_override_default_config_when_env_flag_off():
    config = EvaluationConfig()
    mapper = OutputMapper.from_config(
        config,
        cli_labels="allow,review,block",
        cli_default_label="allow",
        cli_error_label="block",
    )
    assert mapper.labels == ("ALLOW", "REVIEW", "BLOCK")
    assert mapper.default_label == "ALLOW"
    assert mapper.error_label == "BLOCK"


def test_env_labels_win_when_use_env_output_labels_is_enabled():
    config = EvaluationConfig(
        output_labels=("ALLOW", "BLOCK"),
        use_env_output_labels=True,
        default_output_label="ALLOW",
        error_output_label="BLOCK",
    )
    mapper = OutputMapper.from_config(
        config,
        cli_labels="safe,unsafe",
        cli_default_label="safe",
        cli_error_label="unsafe",
    )
    assert mapper.labels == ("ALLOW", "BLOCK")
    assert mapper.default_label == "ALLOW"
    assert mapper.error_label == "BLOCK"
