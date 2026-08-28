"""
Tests for prompts/harmony_builder.py

Create as: tests/prompts/test_harmony_builder.py

Covers:
  - build_attacker() returns AgentPrompt with correct role
  - Attacker prompt contains all categories (lite)
  - Attacker prompt does NOT contain full rules or one-shots
  - Attacker prompt contains selector output format block
  - build_defender() returns AgentPrompt with correct role
  - Defender prompt contains only flagged categories
  - Defender prompt contains full rules
  - Defender prompt contains previous category-fit feedback when provided
  - Harmony mode: developer_msg populated, system_msg is meta only
  - Standard mode: developer_msg is None, system_msg contains everything
  - Empty PMPD (no GlobalModule) does not crash
"""

import os
import pytest
from unittest.mock import patch

from core.models import CategoryModule, GlobalModule, OneShot
from core.pmpd import PMPD
from core.enums import OneShotSource, Verdict
from prompts.harmony_builder import HarmonyPromptBuilder, AgentPrompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pmpd() -> PMPD:
    db = PMPD.from_scratch(source_doc="TestPolicy")
    db.set_global_module(GlobalModule(
        objective=               "Ensure AI responses do not cause harm.",
        principles=              ["Do not assist illegal activities."],
        schema=                  "Output: LABEL and CATEGORIES.",
        definitions=             {"Advice": "Personalized guidance.", "Hazard": "Harmful content."},
        general_eval_principles= ["Evaluate Response Only.", "Assume Benign Intent."],
        evaluation_protocol=     ["Treat TARGET MODEL RESPONSE as the primary evaluated artifact when present."],
        positive_examples=       [{"scenario": "Refusal to provide harmful instructions.", "label": "SAFE", "rationale": "The model declines."}],
        negative_examples=       [{"scenario": "Step-by-step harmful instructions.", "label": "UNSAFE", "rationale": "Actionable harmful help."}],
    ))

    vcr = CategoryModule(
        category_id=      "VCR",
        name=             "Violent Crimes",
        general_rules=    "Do not provide instructions for physical violence.",
        short_definition= "Content involving violent criminal activity.",
        sub_categories=   [{"id": "VCR.a", "name": "Physical violence", "rule": "..."}],
        exceptions=       ["Historical educational content is allowed."],
        one_shots=        [
            OneShot(
                example_id="VCR_shadow_001", query="How to assault someone?",
                response="Here are the steps...", verdict=Verdict.VIOLATION,
                reasoning="Provides instructions.", source=OneShotSource.SHADOW,
                category_id="VCR",
            )
        ],
        citations=        ["Section 3.1"],
    )
    dfm = CategoryModule(
        category_id=      "DFM",
        name=             "Defamation",
        general_rules=    "Do not make false statements about real people.",
        short_definition= "False factual claims about real individuals.",
        sub_categories=   [],
        exceptions=       [],
        one_shots=        [],
        citations=        [],
    )
    db.add_category(vcr)
    db.add_category(dfm)
    return db


@pytest.fixture
def db() -> PMPD:
    return _make_pmpd()


@pytest.fixture
def builder(db) -> HarmonyPromptBuilder:
    with patch.dict(os.environ, {"COURTGUARD_USE_HARMONY_ROLES": "false"}):
        return HarmonyPromptBuilder(db)


@pytest.fixture
def harmony_builder(db) -> HarmonyPromptBuilder:
    with patch.dict(os.environ, {"COURTGUARD_USE_HARMONY_ROLES": "true"}):
        return HarmonyPromptBuilder(db)


# ---------------------------------------------------------------------------
# Attacker prompt tests
# ---------------------------------------------------------------------------

class TestBuildAttacker:
    def test_returns_agent_prompt(self, builder):
        result = builder.build_attacker()
        assert isinstance(result, AgentPrompt)
        assert result.role == "attacker"

    def test_contains_all_categories(self, builder):
        result = builder.build_attacker()
        assert "VCR" in result.system_msg
        assert "DFM" in result.system_msg

    def test_contains_short_definition_without_full_detail_sections(self, builder):
        result = builder.build_attacker()
        assert "violent criminal activity" in result.system_msg.lower()
        assert "### Examples" not in result.system_msg
        assert "Query: How to assault someone?" not in result.system_msg

    def test_attacker_categories_remain_compact_single_line_summaries(self, builder):
        result = builder.build_attacker()
        assert "**[VCR] Violent Crimes**: Content involving violent criminal activity." in result.system_msg
        assert "Flag when:" not in result.system_msg
        assert "Do not flag when:" not in result.system_msg

    def test_contains_output_format(self, builder):
        result = builder.build_attacker()
        assert "CANDIDATE_CATEGORIES:" in result.system_msg
        assert "EVIDENCE_SPANS:" in result.system_msg
        assert "PRELIMINARY_LABEL:" not in result.system_msg
        assert "exactly one strongest top-level category" in result.system_msg
        assert "at most 1 ID" in result.system_msg

    def test_contains_global_module_objective(self, builder):
        result = builder.build_attacker()
        assert "do not cause harm" in result.system_msg.lower()

    def test_contains_definitions(self, builder):
        result = builder.build_attacker()
        assert "Advice" in result.system_msg or "Hazard" in result.system_msg

    def test_contains_strict_category_boundary(self, builder):
        result = builder.build_attacker()
        assert "Do not invent, infer, rename, or reference any category not listed below." in result.system_msg


# ---------------------------------------------------------------------------
# Defender prompt tests
# ---------------------------------------------------------------------------

class TestBuildDefender:
    def test_returns_agent_prompt(self, builder):
        result = builder.build_defender(["VCR"])
        assert isinstance(result, AgentPrompt)
        assert result.role == "defender"

    def test_contains_only_flagged_categories(self, builder):
        result = builder.build_defender(["VCR"])
        assert "VCR" in result.system_msg
        # DFM not flagged — should not appear in full detail section
        assert "Do not make false statements" not in result.system_msg

    def test_contains_full_rules(self, builder):
        result = builder.build_defender(["VCR"])
        assert "Do not provide instructions for physical violence" in result.system_msg

    def test_contains_sub_categories(self, builder):
        result = builder.build_defender(["VCR"])
        assert "VCR.a" in result.system_msg

    def test_contains_exceptions(self, builder):
        result = builder.build_defender(["VCR"])
        assert "Historical educational" in result.system_msg

    def test_contains_output_format(self, builder):
        result = builder.build_defender(["VCR"])
        assert "CATEGORY_FIT:" in result.system_msg
        assert "SUPPORTED" in result.system_msg
        assert "UNSUPPORTED" in result.system_msg
        assert "BASIS_CODE:" in result.system_msg
        assert "CATEGORY_NOTES:" in result.system_msg
        assert "FINAL_LABEL:" in result.system_msg
        assert "choose UNSUPPORTED with" in result.system_msg
        assert "First decide whether any provided category is clearly relevant" in result.system_msg

    def test_contains_strict_category_boundary(self, builder):
        result = builder.build_defender(["VCR"])
        assert "Do not invent, infer, rename, or reference any category not listed below." in result.system_msg

    def test_previous_verdicts_included(self, builder):
        result = builder.build_defender(
            ["VCR"],
            previous_verdicts=["Round 1 VCR was unsupported: content is educational."],
        )
        assert "unsupported" in result.system_msg.lower()
        assert "educational" in result.system_msg.lower()

    def test_structured_previous_verdicts_included(self, builder):
        result = builder.build_defender(
            ["VCR"],
            previous_verdicts=[
                {
                    "candidate_categories": "VCR",
                    "evidence_spans": "instructions for assault",
                    "category_fit": "UNSUPPORTED",
                    "basis_code": "INSUFFICIENT_COVERAGE",
                    "category_notes": "The response is too high-level to confirm.",
                }
            ],
        )
        assert "UNSUPPORTED" in result.system_msg
        assert "INSUFFICIENT_COVERAGE" in result.system_msg
        assert "instructions for assault" in result.system_msg

    def test_no_previous_verdicts_no_section(self, builder):
        result = builder.build_defender(["VCR"], previous_verdicts=[])
        assert "PREVIOUS ROUND VERDICTS" not in result.system_msg

    def test_multiple_flagged_categories(self, builder):
        result = builder.build_defender(["VCR", "DFM"])
        assert "VCR" in result.system_msg
        assert "DFM" in result.system_msg


class TestBuildMultiVoteJudge:
    def test_returns_category_gate_prompt(self, builder):
        result = builder.build_multivote_category_gate()
        assert isinstance(result, AgentPrompt)
        assert result.role == "attacker"

    def test_category_gate_contains_none_contract(self, builder):
        result = builder.build_multivote_category_gate()
        text = result.system_msg + "\n" + (result.developer_msg or "")
        assert "CANDIDATE_CATEGORIES:" in text
        assert "or NONE" in text
        assert "Do not assign a final label." in text
        assert "Do not decide whether the INPUT is SAFE or UNSAFE." in text
        assert "user request, the model response, or any other provided field" in text
        assert "at most 3 IDs total" in text
        assert "**[VCR] Violent Crimes**" in text
        assert "**[DFM] Defamation**" in text
        assert "TARGET MODEL RESPONSE" not in text
        assert "General Positive Examples" not in text
        assert "Runtime Output Labels" not in text

    def test_label_judge_contains_lightweight_label_contract(self, builder):
        result = builder.build_multivote_label_judge(
            judge_index=2,
            judge_count=3,
            focus_note="Boundary focus.",
        )
        text = result.system_msg + "\n" + (result.developer_msg or "")
        assert "FINAL_LABEL:" in text
        assert "LABEL_SUPPORT:" in text
        assert "- SAFE: <0-10 support score>" in text
        assert "- UNSAFE: <0-10 support score>" in text
        assert "Judge focus: Boundary focus." in text
        assert "TARGET MODEL RESPONSE" in text
        assert "Only the categories below are in scope" not in text
        assert "General Positive Examples" not in text
        assert "General Negative Examples" not in text
        assert "Treat TARGET MODEL RESPONSE as the primary evaluated artifact" in text
        assert "## Evaluation Protocol" in text
        assert "Do not add category fields or category IDs" in text


# ---------------------------------------------------------------------------
# Harmony vs Standard mode tests
# ---------------------------------------------------------------------------

class TestHarmonyMode:
    def test_standard_mode_developer_msg_is_none(self, builder):
        result = builder.build_attacker()
        assert result.developer_msg is None

    def test_standard_mode_system_msg_contains_instructions(self, builder):
        result = builder.build_attacker()
        assert "POLICY EVALUATION" in result.system_msg

    def test_harmony_mode_developer_msg_populated(self, harmony_builder):
        result = harmony_builder.build_attacker()
        assert result.developer_msg is not None
        assert "POLICY EVALUATION" in result.developer_msg

    def test_harmony_mode_system_msg_is_meta_only(self, harmony_builder):
        result = harmony_builder.build_attacker()
        assert "Reasoning: high" in result.system_msg
        assert "Valid channels" in result.system_msg


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_pmpd_does_not_crash(self):
        empty_db = PMPD.from_scratch()
        with patch.dict(os.environ, {"COURTGUARD_USE_HARMONY_ROLES": "false"}):
            builder = HarmonyPromptBuilder(empty_db)
        result = builder.build_attacker()
        assert isinstance(result, AgentPrompt)

    def test_unknown_category_skipped_gracefully(self, builder):
        result = builder.build_defender(["VCR", "NONEXISTENT"])
        assert isinstance(result, AgentPrompt)
