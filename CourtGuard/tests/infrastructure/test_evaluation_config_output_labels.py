"""
Optional tests for EvaluationConfig output label environment variables.

Verifies that the Phase 1/2 output taxonomy env vars are correctly parsed
by EvaluationConfig.from_env().

All tests are marked with pytest.mark.optional and are skipped in default
CI runs. They use monkeypatch -- no file I/O, no API calls.

Run with:
    pytest tests/infrastructure/test_evaluation_config_output_labels.py -v
or include optional tests with:
    pytest -m optional
"""

import pytest


@pytest.mark.optional
def test_custom_output_labels_from_env(monkeypatch):
    """COURTGUARD_OUTPUT_LABELS sets the output taxonomy."""
    monkeypatch.setenv("COURTGUARD_OUTPUT_LABELS", "ALLOW,BLOCK")
    monkeypatch.setenv("COURTGUARD_USE_ENV_OUTPUT_LABELS", "true")
    monkeypatch.delenv("COURTGUARD_DEFAULT_OUTPUT_LABEL", raising=False)
    monkeypatch.delenv("COURTGUARD_ERROR_OUTPUT_LABEL",   raising=False)

    from infrastructure.config import EvaluationConfig
    config = EvaluationConfig.from_env()

    assert config.output_labels == ("ALLOW", "BLOCK")
    assert config.use_env_output_labels is True


@pytest.mark.optional
def test_default_output_labels_when_env_not_set(monkeypatch):
    """Without COURTGUARD_OUTPUT_LABELS, default taxonomy is SAFE/UNSAFE."""
    monkeypatch.delenv("COURTGUARD_OUTPUT_LABELS",         raising=False)
    monkeypatch.delenv("COURTGUARD_USE_ENV_OUTPUT_LABELS", raising=False)
    monkeypatch.delenv("COURTGUARD_DEFAULT_OUTPUT_LABEL",  raising=False)
    monkeypatch.delenv("COURTGUARD_ERROR_OUTPUT_LABEL",    raising=False)

    from infrastructure.config import EvaluationConfig
    config = EvaluationConfig.from_env()

    assert config.output_labels == ("SAFE", "UNSAFE")
    assert config.default_output_label == "SAFE"
    assert config.error_output_label == "UNSAFE"


@pytest.mark.optional
def test_custom_default_and_error_labels(monkeypatch):
    """Individual default/error label overrides are respected."""
    monkeypatch.setenv("COURTGUARD_OUTPUT_LABELS",        "PASS,REVIEW,FAIL")
    monkeypatch.setenv("COURTGUARD_DEFAULT_OUTPUT_LABEL", "PASS")
    monkeypatch.setenv("COURTGUARD_ERROR_OUTPUT_LABEL",   "FAIL")

    from infrastructure.config import EvaluationConfig
    config = EvaluationConfig.from_env()

    assert config.output_labels == ("PASS", "REVIEW", "FAIL")
    assert config.default_output_label == "PASS"
    assert config.error_output_label == "FAIL"


@pytest.mark.optional
def test_output_labels_are_uppercased(monkeypatch):
    """Labels from env are normalised to uppercase regardless of input case."""
    monkeypatch.setenv("COURTGUARD_OUTPUT_LABELS", "allow,block")

    from infrastructure.config import EvaluationConfig
    config = EvaluationConfig.from_env()

    assert config.output_labels == ("ALLOW", "BLOCK")


@pytest.mark.optional
def test_use_env_output_labels_false_by_default(monkeypatch):
    """COURTGUARD_USE_ENV_OUTPUT_LABELS defaults to False."""
    monkeypatch.delenv("COURTGUARD_USE_ENV_OUTPUT_LABELS", raising=False)

    from infrastructure.config import EvaluationConfig
    config = EvaluationConfig.from_env()

    assert config.use_env_output_labels is False


@pytest.mark.optional
def test_binary_taxonomy_default_is_first_label(monkeypatch):
    """Without explicit default label env var, first label becomes the default."""
    monkeypatch.setenv("COURTGUARD_OUTPUT_LABELS",        "GOOD,BAD")
    monkeypatch.delenv("COURTGUARD_DEFAULT_OUTPUT_LABEL", raising=False)
    monkeypatch.delenv("COURTGUARD_ERROR_OUTPUT_LABEL",   raising=False)

    from infrastructure.config import EvaluationConfig
    config = EvaluationConfig.from_env()

    assert config.default_output_label == "GOOD"
    assert config.error_output_label == "BAD"
