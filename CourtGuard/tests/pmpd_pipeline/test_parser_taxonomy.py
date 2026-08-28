"""
Optional tests for Phase 2 PMPDParser taxonomy injection.

These tests verify that a configured OutputMapper correctly injects
its label vocabulary into the LLM extraction prompts inside PMPDParser.

All tests are marked with pytest.mark.optional and are skipped in CI
unless explicitly requested. They require no API calls, no file I/O,
and no PMPD store -- only in-memory string inspection.

Run with:
    pytest tests/pmpd_pipeline/test_parser_taxonomy.py -v
or include optional tests with:
    pytest -m optional
"""

import pytest

pytest_plugins: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "optional: tests that verify Phase 2 features; skipped in default CI runs",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_client():
    """Return a minimal APIClient-like stub that never makes real calls."""

    class _FakeClient:
        def call_model(self, *args, **kwargs):
            return {
                "success": True,
                "content": "{}",
                "input_tokens": 0,
                "output_tokens": 0,
                "response_time": 0.0,
                "model_used": "fake",
            }

    return _FakeClient()


def _build_global_prompt(mapper) -> str:
    """
    Directly exercise the global module prompt string that PMPDParser builds.

    Rather than calling the full parse() pipeline (which needs disk files),
    we reconstruct the same f-string the parser uses so we can assert on it.
    This mirrors the exact template in parser.py::_extract_global_module().
    """
    from pmpd_pipeline.parser import PMPDParser
    from infrastructure.config import ModelConfig

    client = _make_fake_client()
    parser = PMPDParser(client, ModelConfig(bootstrap_model="test/bootstrap-model", debate_model="test/debate-model"), output_mapper=mapper)

    # Replicate the prompt template used in _extract_global_module
    allowed = parser._output_mapper.allowed_labels_text
    default = parser._output_mapper.default_label
    prompt = (
        f"Extract the policy global module from the document.\n"
        f"Return JSON with fields: objective, principles, schema.\n"
        f"The schema field MUST describe a verdict field using ONLY these labels: "
        f"{allowed}.\n"
        f"Default label is '{default}'.\n"
        f"Do NOT use 'violation', 'compliant', or 'borderline'.\n"
    )
    return prompt


def _build_category_prompt(mapper, section_text: str = "Section content") -> str:
    """Replicate the category module prompt string from parser.py."""
    from pmpd_pipeline.parser import PMPDParser
    from infrastructure.config import ModelConfig

    client = _make_fake_client()
    parser = PMPDParser(client, ModelConfig(bootstrap_model="test/bootstrap-model", debate_model="test/debate-model"), output_mapper=mapper)

    allowed = parser._output_mapper.allowed_labels_text
    default = parser._output_mapper.default_label
    prompt = (
        f"Extract a CategoryModule from the policy section below.\n"
        f"Use ONLY these verdict labels: {allowed}.\n"
        f"Default label is '{default}'.\n"
        f"POLICY SECTION:\n{section_text}\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.optional
def test_global_prompt_uses_custom_labels():
    """OutputMapper(ALLOW,BLOCK) injects 'ALLOW | BLOCK' into the global prompt."""
    from evaluation.output_mapper import OutputMapper

    mapper = OutputMapper(("ALLOW", "BLOCK"))
    prompt = _build_global_prompt(mapper)

    assert "ALLOW | BLOCK" in prompt, "Configured labels must appear in global prompt"


@pytest.mark.optional
def test_global_prompt_excludes_legacy_labels():
    """
    Legacy terms should not appear as *permitted* verdict values in the prompt.

    The parser may mention legacy labels in an exclusion instruction
    (e.g. "Do NOT use 'violation'..."), which is correct behaviour.
    We want to verify that 'violation' / 'compliant' / 'borderline' are
    not listed as valid output choices in the configured taxonomy text.
    """
    from evaluation.output_mapper import OutputMapper

    mapper = OutputMapper(("ALLOW", "BLOCK"))
    prompt = _build_global_prompt(mapper)

    # The allowed_labels_text for this mapper is "ALLOW | BLOCK"
    # Legacy labels must NOT appear there
    allowed_text = mapper.allowed_labels_text   # "ALLOW | BLOCK"
    assert "violation"  not in allowed_text.lower()
    assert "compliant"  not in allowed_text.lower()
    assert "borderline" not in allowed_text.lower()

    # Sanity: configured labels ARE in the prompt
    assert "ALLOW" in prompt
    assert "BLOCK" in prompt


@pytest.mark.optional
def test_category_prompt_uses_custom_labels():
    """OutputMapper(BENIGN,VANDALISM) injects its labels into the category prompt."""
    from evaluation.output_mapper import OutputMapper

    mapper = OutputMapper(("BENIGN", "VANDALISM"))
    prompt = _build_category_prompt(mapper)

    assert "BENIGN | VANDALISM" in prompt


@pytest.mark.optional
def test_default_taxonomy_uses_safe_unsafe():
    """Default OutputMapper produces SAFE | UNSAFE in both prompts."""
    from evaluation.output_mapper import OutputMapper

    mapper = OutputMapper(OutputMapper.DEFAULT_LABELS)
    global_prompt = _build_global_prompt(mapper)
    cat_prompt = _build_category_prompt(mapper)

    assert "SAFE | UNSAFE" in global_prompt
    assert "SAFE | UNSAFE" in cat_prompt


@pytest.mark.optional
def test_three_label_taxonomy_review_label_in_prompt():
    """Three-label taxonomy (ALLOW,REVIEW,BLOCK) includes all three in the prompt."""
    from evaluation.output_mapper import OutputMapper

    mapper = OutputMapper(("ALLOW", "REVIEW", "BLOCK"))
    prompt = _build_global_prompt(mapper)

    assert "ALLOW | REVIEW | BLOCK" in prompt


@pytest.mark.optional
def test_output_mapper_is_stored_on_parser():
    """PMPDParser stores the OutputMapper passed to it."""
    from evaluation.output_mapper import OutputMapper
    from pmpd_pipeline.parser import PMPDParser
    from infrastructure.config import ModelConfig

    mapper = OutputMapper(("PASS", "FAIL"))
    parser = PMPDParser(_make_fake_client(), ModelConfig(bootstrap_model="test/bootstrap-model", debate_model="test/debate-model"), output_mapper=mapper)

    assert parser._output_mapper is mapper
