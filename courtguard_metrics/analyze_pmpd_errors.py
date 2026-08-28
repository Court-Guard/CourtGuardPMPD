"""Error analysis for CourtGuard PMPD result ledgers.

This script focuses on the failure buckets that matter most for PMPD tuning:

- attacker miss count
- overturn TP loss / FP save
- overturn basis breakdown
- round-2 rescue / backdown
- per-category false-negative source breakdown

It is intentionally tolerant of older result ledgers that may not contain the
newer structured telemetry fields added during runtime experimentation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import evaluator as metrics_eval


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _load_results_records(results_path: str) -> list[dict]:
    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("records", "results", "entries", "output_records"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported results payload shape in: {os.path.abspath(results_path)}")


def _infer_hazard(item: dict) -> str:
    for key in (
        "hazard_cate_llamaguard3",
        "category",
        "hazard_category",
        "policy_category",
        "category_id",
    ):
        value = str(item.get(key, "")).strip()
        if value:
            return value
    return "UNKNOWN"


def _load_ground_truth_map(ground_truth_path: str) -> dict[int, dict]:
    with open(ground_truth_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    dataset_name = Path(ground_truth_path).stem
    gt_map: dict[int, dict] = {}

    for item in data:
        idx = item.get("index")
        if idx is None:
            continue

        truth = metrics_eval.extract_ground_truth_label(item, dataset_name)
        if truth is None:
            continue

        gt_map[idx] = {
            "truth": truth,
            "hazard": _infer_hazard(item),
            "raw_item": item,
        }

    if not gt_map:
        raise ValueError(
            f"No labeled ground-truth entries could be extracted from: {os.path.abspath(ground_truth_path)}"
        )

    return gt_map


def _attacker_round_is_unsafe(round_data: dict) -> bool:
    preliminary = str(round_data.get("preliminary_label") or "").strip().upper()
    categories = metrics_eval.normalize_category_list(round_data.get("categories"))
    return preliminary == "UNSAFE" or bool(categories)


def _final_prediction(item: dict) -> int | None:
    return metrics_eval.derive_full_debate_prediction(item)


def _one_round_prediction(item: dict) -> int | None:
    return metrics_eval.derive_one_round_prediction(item)


def _attacker_prediction(item: dict) -> int | None:
    return metrics_eval.derive_attacker_only_prediction(item)


def analyze_results(results_path: str, ground_truth_path: str | None = None) -> dict:
    resolved_ground_truth, dataset_name, dataset_suite = metrics_eval.resolve_ground_truth_metadata(
        ground_truth_path,
        results_path,
    )
    records = _load_results_records(results_path)
    gt_map = _load_ground_truth_map(resolved_ground_truth)

    confusion = Counter()
    prediction_volumes = Counter()

    attacker_parse_fail = 0
    attacker_invalid_mismatch = 0
    defender_parse_fail = 0
    defender_invalid_mismatch = 0

    attacker_round1_unsafe_total = 0
    attacker_round1_unsafe_gt_unsafe = 0
    attacker_round1_unsafe_gt_safe = 0

    overturn_total = 0
    overturn_tp_loss = 0
    overturn_fp_saved = 0
    overturn_basis_total = Counter()
    overturn_basis_unsafe = Counter()
    overturn_basis_safe = Counter()

    round2_triggered = 0
    round2_rescues = 0
    round2_extra_false_positives = 0
    round2_persisted_after_overturn = Counter()
    round2_backed_down_after_overturn = Counter()

    final_false_negatives = 0
    fn_attacker_miss = 0
    fn_attacker_parse_fail = 0
    fn_attacker_invalid_mismatch = 0
    fn_round1_overturn_path = 0
    fn_defender_parse_fail = 0
    fn_defender_basis = Counter()
    fn_round2_backdown = 0
    fn_round2_persist_but_still_safe = 0
    per_category_fn_sources: dict[str, Counter] = defaultdict(Counter)

    for item in records:
        idx = item.get("index")
        if idx is None or idx not in gt_map:
            continue

        truth = int(gt_map[idx]["truth"])
        hazard = gt_map[idx]["hazard"]
        gt_unsafe = truth == 1

        full_pred = _final_prediction(item)
        one_round_pred = _one_round_prediction(item)
        attacker_pred = _attacker_prediction(item)

        if attacker_pred == 1:
            attacker_round1_unsafe_total += 1
            if gt_unsafe:
                attacker_round1_unsafe_gt_unsafe += 1
            else:
                attacker_round1_unsafe_gt_safe += 1

        if one_round_pred == 1:
            prediction_volumes["one_round_unsafe"] += 1
        if full_pred == 1:
            prediction_volumes["full_debate_unsafe"] += 1
        if attacker_pred == 1:
            prediction_volumes["attacker_only_unsafe"] += 1

        attacker_rounds = item.get("attacker_raw_rounds") or []
        defender_rounds = item.get("defender_raw_rounds") or []

        if attacker_rounds:
            round_one_attacker = attacker_rounds[0]
            if not bool(round_one_attacker.get("parse_success", True)):
                attacker_parse_fail += 1
            if bool(round_one_attacker.get("invalid_category_mismatch")):
                attacker_invalid_mismatch += 1

        if defender_rounds:
            round_one_defender = defender_rounds[0]
            if not bool(round_one_defender.get("parse_success", True)):
                defender_parse_fail += 1
            if bool(round_one_defender.get("invalid_category_mismatch")):
                defender_invalid_mismatch += 1

        if attacker_pred == 1 and one_round_pred == 0 and defender_rounds:
            overturn_total += 1
            basis = str(defender_rounds[0].get("overturn_basis") or "UNKNOWN").strip().upper()
            overturn_basis_total[basis] += 1
            if gt_unsafe:
                overturn_tp_loss += 1
                overturn_basis_unsafe[basis] += 1
            else:
                overturn_fp_saved += 1
                overturn_basis_safe[basis] += 1

        if len(attacker_rounds) > 1:
            round2_triggered += 1
            if one_round_pred != 1 and full_pred == 1:
                if gt_unsafe:
                    round2_rescues += 1
                else:
                    round2_extra_false_positives += 1

            if attacker_pred == 1 and one_round_pred == 0:
                round_two_attacker = attacker_rounds[1]
                if _attacker_round_is_unsafe(round_two_attacker):
                    round2_persisted_after_overturn["unsafe" if gt_unsafe else "safe"] += 1
                else:
                    round2_backed_down_after_overturn["unsafe" if gt_unsafe else "safe"] += 1

        if full_pred == 1 and truth == 1:
            confusion["tp"] += 1
        elif full_pred == 1 and truth == 0:
            confusion["fp"] += 1
        elif full_pred == 0 and truth == 0:
            confusion["tn"] += 1
        elif full_pred == 0 and truth == 1:
            confusion["fn"] += 1

        if truth == 1 and full_pred != 1:
            final_false_negatives += 1
            attacker_round_one_unsafe = attacker_pred == 1

            if not attacker_round_one_unsafe:
                fn_attacker_miss += 1
                per_category_fn_sources[hazard]["attacker_miss"] += 1
                if attacker_rounds:
                    if not bool(attacker_rounds[0].get("parse_success", True)):
                        fn_attacker_parse_fail += 1
                        per_category_fn_sources[hazard]["attacker_parse_fail"] += 1
                    if bool(attacker_rounds[0].get("invalid_category_mismatch")):
                        fn_attacker_invalid_mismatch += 1
                        per_category_fn_sources[hazard]["attacker_invalid_mismatch"] += 1
                continue

            fn_round1_overturn_path += 1
            if defender_rounds and not bool(defender_rounds[0].get("parse_success", True)):
                fn_defender_parse_fail += 1
                per_category_fn_sources[hazard]["defender_parse_fail"] += 1

            basis = (
                str(defender_rounds[0].get("overturn_basis") or "UNKNOWN").strip().upper()
                if defender_rounds
                else "UNKNOWN"
            )
            fn_defender_basis[basis] += 1
            per_category_fn_sources[hazard][f"overturn_basis::{basis}"] += 1

            if len(attacker_rounds) > 1:
                round_two_attacker = attacker_rounds[1]
                if _attacker_round_is_unsafe(round_two_attacker):
                    fn_round2_persist_but_still_safe += 1
                    per_category_fn_sources[hazard]["persisted_after_overturn_but_still_safe"] += 1
                else:
                    fn_round2_backdown += 1
                    per_category_fn_sources[hazard]["backed_down_after_overturn"] += 1
            else:
                per_category_fn_sources[hazard]["overturned_without_round2"] += 1

    total = confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"]
    accuracy = _safe_div(confusion["tp"] + confusion["tn"], total)
    precision = _safe_div(confusion["tp"], confusion["tp"] + confusion["fp"])
    recall = _safe_div(confusion["tp"], confusion["tp"] + confusion["fn"])
    specificity = _safe_div(confusion["tn"], confusion["tn"] + confusion["fp"])
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    return {
        "configuration": {
            "results_path": os.path.abspath(results_path),
            "ground_truth_path": os.path.abspath(resolved_ground_truth),
            "dataset_name": dataset_name,
            "dataset_suite": dataset_suite,
            "sample_count": total,
        },
        "final_metrics": {
            "accuracy": _round_or_none(accuracy),
            "precision": _round_or_none(precision),
            "recall": _round_or_none(recall),
            "specificity": _round_or_none(specificity),
            "f1_score": _round_or_none(f1),
            "confusion_matrix": {
                "true_positives": confusion["tp"],
                "true_negatives": confusion["tn"],
                "false_positives": confusion["fp"],
                "false_negatives": confusion["fn"],
            },
        },
        "prediction_volumes": {
            "attacker_only_unsafe": prediction_volumes["attacker_only_unsafe"],
            "one_round_unsafe": prediction_volumes["one_round_unsafe"],
            "full_debate_unsafe": prediction_volumes["full_debate_unsafe"],
        },
        "attacker_stage": {
            "round1_unsafe_predictions": attacker_round1_unsafe_total,
            "round1_unsafe_on_true_unsafe": attacker_round1_unsafe_gt_unsafe,
            "round1_unsafe_on_true_safe": attacker_round1_unsafe_gt_safe,
            "parse_fail_count": attacker_parse_fail,
            "invalid_category_mismatch_count": attacker_invalid_mismatch,
        },
        "defender_overturns": {
            "total_overturns": overturn_total,
            "tp_loss_count": overturn_tp_loss,
            "fp_saved_count": overturn_fp_saved,
            "basis_breakdown_total": dict(sorted(overturn_basis_total.items())),
            "basis_breakdown_on_true_unsafe": dict(sorted(overturn_basis_unsafe.items())),
            "basis_breakdown_on_true_safe": dict(sorted(overturn_basis_safe.items())),
            "defender_parse_fail_count": defender_parse_fail,
            "defender_invalid_category_mismatch_count": defender_invalid_mismatch,
        },
        "round2_dynamics": {
            "round2_triggered": round2_triggered,
            "round2_rescue_count": round2_rescues,
            "round2_extra_false_positive_count": round2_extra_false_positives,
            "persisted_after_overturn": dict(sorted(round2_persisted_after_overturn.items())),
            "backed_down_after_overturn": dict(sorted(round2_backed_down_after_overturn.items())),
        },
        "false_negative_breakdown": {
            "total_final_false_negatives": final_false_negatives,
            "attacker_miss_count": fn_attacker_miss,
            "attacker_parse_fail_count": fn_attacker_parse_fail,
            "attacker_invalid_category_mismatch_count": fn_attacker_invalid_mismatch,
            "debate_path_false_negatives": fn_round1_overturn_path,
            "defender_parse_fail_count": fn_defender_parse_fail,
            "defender_overturn_basis_breakdown": dict(sorted(fn_defender_basis.items())),
            "round2_backdown_count": fn_round2_backdown,
            "round2_persisted_but_final_safe_count": fn_round2_persist_but_still_safe,
        },
        "per_category_fn_source_breakdown": {
            hazard: dict(sorted(counter.items()))
            for hazard, counter in sorted(per_category_fn_sources.items())
        },
    }


def _format_counter_lines(mapping: dict[str, int], indent: str = "  ") -> list[str]:
    if not mapping:
        return [f"{indent}(none)"]
    return [f"{indent}{key}: {value}" for key, value in mapping.items()]


def render_text_report(summary: dict) -> str:
    cfg = summary["configuration"]
    metrics = summary["final_metrics"]
    attacker = summary["attacker_stage"]
    overturns = summary["defender_overturns"]
    round2 = summary["round2_dynamics"]
    fn = summary["false_negative_breakdown"]
    per_category = summary["per_category_fn_source_breakdown"]

    lines: list[str] = []
    lines.append("PMPD Error Analysis")
    lines.append(f"Results file: {cfg['results_path']}")
    lines.append(f"Ground truth: {cfg['ground_truth_path']}")
    lines.append(f"Dataset: {cfg['dataset_name']} ({cfg['dataset_suite']})")
    lines.append(f"Samples: {cfg['sample_count']}")
    lines.append("")
    lines.append("Final Metrics")
    lines.append(
        "  "
        f"acc={metrics['accuracy']} "
        f"prec={metrics['precision']} "
        f"recall={metrics['recall']} "
        f"spec={metrics['specificity']} "
        f"f1={metrics['f1_score']}"
    )
    cm = metrics["confusion_matrix"]
    lines.append(
        "  "
        f"TP={cm['true_positives']} TN={cm['true_negatives']} "
        f"FP={cm['false_positives']} FN={cm['false_negatives']}"
    )
    lines.append("")
    lines.append("Attacker Stage")
    lines.append(f"  round1 unsafe predictions: {attacker['round1_unsafe_predictions']}")
    lines.append(
        "  "
        f"round1 unsafe on true-unsafe: {attacker['round1_unsafe_on_true_unsafe']} | "
        f"on true-safe: {attacker['round1_unsafe_on_true_safe']}"
    )
    lines.append(
        "  "
        f"attacker parse fails: {attacker['parse_fail_count']} | "
        f"invalid category mismatches: {attacker['invalid_category_mismatch_count']}"
    )
    lines.append("")
    lines.append("Defender Overturns")
    lines.append(
        "  "
        f"total={overturns['total_overturns']} "
        f"tp_loss={overturns['tp_loss_count']} "
        f"fp_saved={overturns['fp_saved_count']}"
    )
    lines.append(
        "  "
        f"defender parse fails: {overturns['defender_parse_fail_count']} | "
        f"invalid category mismatches: {overturns['defender_invalid_category_mismatch_count']}"
    )
    lines.append("  basis breakdown (all):")
    lines.extend(_format_counter_lines(overturns["basis_breakdown_total"], indent="    "))
    lines.append("  basis breakdown on true-unsafe:")
    lines.extend(_format_counter_lines(overturns["basis_breakdown_on_true_unsafe"], indent="    "))
    lines.append("  basis breakdown on true-safe:")
    lines.extend(_format_counter_lines(overturns["basis_breakdown_on_true_safe"], indent="    "))
    lines.append("")
    lines.append("Round 2 Dynamics")
    lines.append(
        "  "
        f"triggered={round2['round2_triggered']} "
        f"rescues={round2['round2_rescue_count']} "
        f"extra_fp={round2['round2_extra_false_positive_count']}"
    )
    lines.append("  persisted after overturn:")
    lines.extend(_format_counter_lines(round2["persisted_after_overturn"], indent="    "))
    lines.append("  backed down after overturn:")
    lines.extend(_format_counter_lines(round2["backed_down_after_overturn"], indent="    "))
    lines.append("")
    lines.append("False Negative Breakdown")
    lines.append(
        "  "
        f"total={fn['total_final_false_negatives']} "
        f"attacker_miss={fn['attacker_miss_count']} "
        f"debate_path={fn['debate_path_false_negatives']}"
    )
    lines.append(
        "  "
        f"attacker parse fail={fn['attacker_parse_fail_count']} "
        f"attacker invalid mismatch={fn['attacker_invalid_category_mismatch_count']} "
        f"defender parse fail={fn['defender_parse_fail_count']}"
    )
    lines.append(
        "  "
        f"round2 backdown={fn['round2_backdown_count']} "
        f"round2 persisted but final safe={fn['round2_persisted_but_final_safe_count']}"
    )
    lines.append("  defender overturn basis on FN path:")
    lines.extend(_format_counter_lines(fn["defender_overturn_basis_breakdown"], indent="    "))
    lines.append("")
    lines.append("Per-Category FN Source Breakdown")
    if not per_category:
        lines.append("  (none)")
    else:
        for hazard, mapping in per_category.items():
            lines.append(f"  {hazard}:")
            lines.extend(_format_counter_lines(mapping, indent="    "))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PMPD error buckets for a CourtGuard result ledger.")
    parser.add_argument("--results", required=True, help="Path to a PMPD results JSON file.")
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Optional explicit ground-truth dataset path. If omitted, it is inferred from the results filename.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the structured analysis JSON.",
    )
    args = parser.parse_args()

    summary = analyze_results(args.results, args.ground_truth)
    print(render_text_report(summary))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
