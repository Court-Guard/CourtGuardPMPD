"""Prompt/parser contract tests for PMPD harmony prompts."""

from core.models import CategoryModule, GlobalModule
from core.pmpd import PMPD
from evaluation.output_mapper import OutputMapper
from prompts.harmony_builder import HarmonyPromptBuilder


def _make_db() -> PMPD:
    db = PMPD()
    db.set_global_module(
        GlobalModule(
            objective="Objective",
            principles=[],
            schema="Use S1-S9",
            definitions={},
            general_eval_principles=[],
        )
    )
    db.add_category(
        CategoryModule(
            category_id="S9",
            name="Violent Crimes",
            general_rules="Violent crime assistance is disallowed.",
            exceptions=[],
            one_shots=[],
            citations=[],
            short_definition="Violent crimes.",
        )
    )
    return db


def test_attacker_prompt_requires_top_level_ids():
    prompt = HarmonyPromptBuilder(_make_db()).build_attacker()
    text = prompt.system_msg + "\n" + (prompt.developer_msg or "")

    assert "Do not invent, infer, rename, or reference any category not listed below." in text
    assert "Use only the exact top-level IDs" in text
    assert "EVIDENCE_SPANS:" in text
    assert "PRELIMINARY_LABEL:" not in text
    assert "at most 1 ID" in text
    assert "CANDIDATE_CATEGORIES: S9" in text
    assert "CATEGORIES: VCR" not in text


def test_defender_prompt_requires_top_level_ids():
    prompt = HarmonyPromptBuilder(_make_db()).build_defender(["S9"], [])
    text = prompt.system_msg + "\n" + (prompt.developer_msg or "")

    assert "Do not invent, infer, rename, or reference any category not listed below." in text
    assert "Use only the exact top-level IDs" in text
    assert "CATEGORY_FIT:" in text
    assert "BASIS_CODE:" in text
    assert "CATEGORY_NOTES:" in text
    assert "choose UNSUPPORTED with" in text
    assert "First decide whether any provided category is clearly relevant" in text
    assert "CATEGORIES: S9" in text
    assert "CATEGORIES: VCR" not in text


def test_attacker_prompt_renders_live_category_ids():
    prompt = HarmonyPromptBuilder(_make_db()).build_attacker()
    text = prompt.system_msg + "\n" + (prompt.developer_msg or "")

    assert "[S9] Violent Crimes" in text
    assert "**[S9] Violent Crimes**: Violent crimes." in text


def test_multivote_prompt_lists_all_allowed_labels():
    builder = HarmonyPromptBuilder(
        _make_db(),
        output_mapper=OutputMapper(("SAFE", "BORDERLINE", "UNSAFE")),
    )
    prompt = builder.build_multivote_label_judge(
        judge_index=1,
        judge_count=3,
        focus_note="Boundary emphasis.",
    )
    text = prompt.system_msg + "\n" + (prompt.developer_msg or "")

    assert "FINAL_LABEL: <one of SAFE | BORDERLINE | UNSAFE>" in text
    assert "- SAFE: <0-10 support score>" in text
    assert "- BORDERLINE: <0-10 support score>" in text
    assert "- UNSAFE: <0-10 support score>" in text
    assert "CATEGORIES: <comma-separated provided top-level category IDs" not in text
    assert "Only the categories below are in scope." not in text
    assert "Do not add category fields or category IDs in the structured block." in text


def test_multivote_gate_prompt_requires_none_or_known_ids():
    prompt = HarmonyPromptBuilder(_make_db()).build_multivote_category_gate()
    text = prompt.system_msg + "\n" + (prompt.developer_msg or "")

    assert "CANDIDATE_CATEGORIES:" in text
    assert "or NONE" in text
    assert "Do not assign a final label." in text
    assert "Do not decide whether the INPUT is SAFE or UNSAFE." in text
    assert "at most 3 IDs total" in text

