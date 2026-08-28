import argparse
import json
import math
import os
import re
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METRICS_PATH = os.path.join(SCRIPT_DIR, "results", "results_metrics.json")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "courtguard_aggregate_tables.tex")
TARGET_SECTION = "section_4_opposite_imputing"
SUITE_ORDER = ("main_benchmarks", "manual_labeled_suite")
SUITE_TITLES = {
    "main_benchmarks": "Main Benchmarks",
    "manual_labeled_suite": "Manual Labeled Suite",
}
VARIANT_ORDER = {
    "full_debate": 0,
    "one_round": 1,
    "attacker_only": 2,
}
TOKEN_DISPLAY_MAP = {
    "cg": "CG",
    "pmpd": "PMPD",
    "rag": "RAG",
    "gpt": "GPT",
    "oss": "OSS",
    "llm": "LLM",
    "pku": "PKU",
    "rlhf": "RLHF",
    "xstest": "XSTEST",
    "jailjudge": "JAILJUDGE",
    "wildguard": "WildGuard",
    "harmbench": "HarmBench",
    "beavertails": "BeaverTails",
    "toxic": "Toxic",
    "chat": "Chat",
    "deepseek": "DeepSeek",
    "ministral": "Ministral",
    "llama": "Llama",
    "instruct": "Instruct",
    "new": "NEW",
}
METRIC_COLUMNS = [
    ("accuracy", "Acc"),
    ("precision", "Prec"),
    ("recall", "Rec"),
    ("specificity", "Spec"),
    ("f1_score", "F1"),
    ("f2_score", "F2"),
    ("auc", "AUC"),
    ("roc_auc_score", "ROC AUC"),
]


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def prettify_token(token):
    lowered = token.lower()
    if lowered in TOKEN_DISPLAY_MAP:
        return TOKEN_DISPLAY_MAP[lowered]
    if token.isdigit():
        return token
    if token.endswith("b") and token[:-1].isdigit():
        return token.upper()
    return token.capitalize()


def prettify_identifier(identifier):
    tokens = [token for token in re.split(r"[-_]", identifier) if token]
    return " ".join(prettify_token(token) for token in tokens)


def parse_results_filename(results_path, dataset_name=None):
    filename = os.path.basename(results_path)
    match = re.match(r"^CourtGuard_(?P<mode>[^_]+)_(?P<rest>.+)_(?P<timestamp>\d{8}_\d{6})\.json$", filename)
    if not match:
        return None

    rest = match.group("rest")
    model_name = None

    if dataset_name:
        suffix = f"_{dataset_name}"
        if rest.endswith(suffix):
            model_name = rest[: -len(suffix)]

    if not model_name:
        parts = rest.split("_")
        if len(parts) < 2:
            return None
        model_name = "_".join(parts[:-1])
        dataset_name = parts[-1]

    return {
        "mode": match.group("mode"),
        "model": model_name,
        "dataset": dataset_name,
    }


def format_row_label(variant_display, mode, model):
    model_label = prettify_identifier(model)
    if variant_display:
        return f"{variant_display} {model_label}"
    return f"CG {prettify_identifier(mode)} {model_label}"


def load_metrics_entries(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError("Metrics file must contain a JSON object or array.")
    return payload


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def determine_suite(config):
    suite = config.get("dataset_suite")
    if suite:
        return suite

    ground_truth_path = str(config.get("ground_truth_dataset", ""))
    results_path = str(config.get("results_dataset", ""))

    if "manual_labeled_suite" in ground_truth_path or "manual_results" in results_path:
        return "manual_labeled_suite"
    if os.path.join("data", "datasets") in ground_truth_path or os.path.join("data", "results") in results_path:
        return "main_benchmarks"

    return "unknown_suite"


def build_row_key(config, parsed):
    variant_id = str(config.get("evaluation_variant", "full_debate"))
    variant_display = str(config.get("evaluation_variant_display", "")).strip()
    return (
        VARIANT_ORDER.get(variant_id, 99),
        variant_display.lower(),
        parsed["mode"].lower(),
        parsed["model"].lower(),
        variant_display,
        parsed["mode"],
        parsed["model"],
    )


def compute_aggregate_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1_score = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    beta = 2
    beta_sq = beta * beta
    f2_score = safe_divide((1 + beta_sq) * precision * recall, (beta_sq * precision) + recall) if ((beta_sq * precision) + recall) else 0.0

    roc_auc = (recall + specificity) / 2.0

    return {
        "valid_count": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "f2_score": f2_score,
        "auc": roc_auc,
        "roc_auc_score": roc_auc,
    }


def collect_aggregate_data(entries):
    grouped = defaultdict(
        lambda: defaultdict(
            lambda: {
                "datasets": set(),
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "runtime_covered_sample_count": 0,
                "runtime_total_tokens": 0,
                "runtime_total_latency_s": 0.0,
                "runtime_attacker_total_tokens": 0,
                "runtime_defender_total_tokens": 0,
            }
        )
    )
    suite_dataset_names = defaultdict(set)

    for entry in entries:
        config = entry.get("configuration", {})
        results_path = config.get("results_dataset")
        if not results_path:
            continue

        suite = determine_suite(config)
        ground_truth_path = config.get("ground_truth_dataset", "")
        dataset_name_from_gt = config.get("dataset_name") or (
            os.path.splitext(os.path.basename(ground_truth_path))[0] if ground_truth_path else None
        )
        parsed = parse_results_filename(results_path, dataset_name=dataset_name_from_gt)
        if not parsed:
            continue

        section_metrics = entry.get("evaluation_results", {}).get(TARGET_SECTION, {})
        confusion = section_metrics.get("confusion_matrix")
        if not confusion:
            continue

        row_key = build_row_key(config, parsed)
        grouped[suite][row_key]["datasets"].add(parsed["dataset"])
        grouped[suite][row_key]["tp"] += int(confusion.get("true_positives", 0))
        grouped[suite][row_key]["tn"] += int(confusion.get("true_negatives", 0))
        grouped[suite][row_key]["fp"] += int(confusion.get("false_positives", 0))
        grouped[suite][row_key]["fn"] += int(confusion.get("false_negatives", 0))

        runtime_stats = entry.get("runtime_statistics", {})
        if isinstance(runtime_stats, dict) and runtime_stats.get("available"):
            totals = runtime_stats.get("totals", {})
            per_agent = runtime_stats.get("per_agent", {})
            attacker_runtime = per_agent.get("attacker", {}) if isinstance(per_agent, dict) else {}
            defender_runtime = per_agent.get("defender", {}) if isinstance(per_agent, dict) else {}

            grouped[suite][row_key]["runtime_covered_sample_count"] += int(
                runtime_stats.get("covered_sample_count", 0) or 0
            )
            grouped[suite][row_key]["runtime_total_tokens"] += int(totals.get("total_tokens", 0) or 0)
            grouped[suite][row_key]["runtime_total_latency_s"] += float(totals.get("latency_s", 0.0) or 0.0)
            grouped[suite][row_key]["runtime_attacker_total_tokens"] += int(
                attacker_runtime.get("total_tokens", 0) or 0
            )
            grouped[suite][row_key]["runtime_defender_total_tokens"] += int(
                defender_runtime.get("total_tokens", 0) or 0
            )
        suite_dataset_names[suite].add(parsed["dataset"])

    suite_tables = {}
    for suite_name, suite_group in grouped.items():
        rows = {}
        for row_key, totals in suite_group.items():
            rows[row_key] = compute_aggregate_metrics(
                tp=totals["tp"],
                tn=totals["tn"],
                fp=totals["fp"],
                fn=totals["fn"],
            )
            rows[row_key]["datasets"] = sorted(totals["datasets"], key=str.lower)
            covered_samples = totals["runtime_covered_sample_count"]
            if covered_samples > 0:
                rows[row_key]["runtime_statistics"] = {
                    "available": True,
                    "covered_sample_count": covered_samples,
                    "total_tokens": totals["runtime_total_tokens"],
                    "total_latency_s": totals["runtime_total_latency_s"],
                    "mean_total_tokens": safe_divide(totals["runtime_total_tokens"], covered_samples),
                    "mean_latency_s": safe_divide(totals["runtime_total_latency_s"], covered_samples),
                    "mean_attacker_tokens": safe_divide(
                        totals["runtime_attacker_total_tokens"], covered_samples
                    ),
                    "mean_defender_tokens": safe_divide(
                        totals["runtime_defender_total_tokens"], covered_samples
                    ),
                }
            else:
                rows[row_key]["runtime_statistics"] = {"available": False}

        suite_tables[suite_name] = {
            "rows": rows,
            "dataset_names": sorted(suite_dataset_names[suite_name], key=str.lower),
        }

    return suite_tables


def format_dataset_list(dataset_names):
    return ", ".join(prettify_identifier(name) for name in dataset_names)


def find_best_values(rows):
    best_values = {}
    for metric_key, _ in METRIC_COLUMNS:
        values = []
        for row in rows.values():
            value = row.get(metric_key)
            if isinstance(value, (int, float)) and not math.isnan(value):
                values.append(value)
        best_values[metric_key] = max(values) if values else None
    return best_values


def format_metric(metric_key, value, best_value):
    if metric_key == "valid_count":
        return str(int(value))

    if not isinstance(value, (int, float)) or math.isnan(value):
        return "-"

    formatted = f"{value:.2f}"
    if best_value is not None and abs(value - best_value) < 1e-12:
        return rf"\textbf{{{formatted}}}"
    return formatted


def render_suite_table(rows, dataset_names, suite_name):
    if not rows:
        raise ValueError("No CourtGuard aggregate rows were found in the metrics file.")

    best_values = find_best_values(rows)
    suite_title = SUITE_TITLES.get(suite_name, prettify_identifier(suite_name))
    dataset_text = format_dataset_list(dataset_names)

    lines = []
    lines.append(rf"\subsection*{{{latex_escape(suite_title)}}}")
    lines.append(rf"\textbf{{Datasets Used:}} {latex_escape(dataset_text)}")
    lines.append(r"")
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(rf"    \caption{{{latex_escape(suite_title)}: CourtGuard Aggregate Performance}}")
    lines.append(r"    \resizebox{\linewidth}{!}{%")
    lines.append(r"        \begin{tabular}{l|c|cccc|cccccccc}")
    lines.append(r"            \toprule")
    lines.append(r"            \multirow{2}{*}{\textbf{CourtGuard Config}} & \textbf{Valid} & \multicolumn{4}{c|}{\textbf{Confusion Matrix}} & \multicolumn{8}{c}{\textbf{Aggregated Metrics}} \\")
    lines.append(r"             & \textbf{Count} & \textbf{TP} & \textbf{TN} & \textbf{FP} & \textbf{FN} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{Spec} & \textbf{F1} & \textbf{F2} & \textbf{AUC} & \textbf{ROC AUC} \\")
    lines.append(r"            \midrule")

    sorted_row_keys = sorted(rows)
    for row_key in sorted_row_keys:
        _, _, _, _, variant_display, mode, model = row_key
        row = rows[row_key]
        row_cells = [
            latex_escape(format_row_label(variant_display, mode, model)),
            str(int(row["valid_count"])),
            str(int(row["true_positives"])),
            str(int(row["true_negatives"])),
            str(int(row["false_positives"])),
            str(int(row["false_negatives"])),
        ]

        for metric_key, _ in METRIC_COLUMNS:
            row_cells.append(format_metric(metric_key, row[metric_key], best_values[metric_key]))

        lines.append("            " + " & ".join(row_cells) + r" \\")

    lines.append(r"            \bottomrule")
    lines.append(r"        \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"\end{table}")
    lines.append(r"")
    return "\n".join(lines)


def render_runtime_suite_table(rows, suite_name):
    suite_title = SUITE_TITLES.get(suite_name, prettify_identifier(suite_name))

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(rf"    \caption{{{latex_escape(suite_title)}: CourtGuard Aggregate Runtime Costs}}")
    lines.append(r"    \resizebox{\linewidth}{!}{%")
    lines.append(r"        \begin{tabular}{l|c|cc|cc|cc}")
    lines.append(r"            \toprule")
    lines.append(
        r"            \multirow{2}{*}{\textbf{CourtGuard Config}} & \textbf{Covered} & "
        r"\multicolumn{2}{c|}{\textbf{Tokens}} & \multicolumn{2}{c|}{\textbf{Latency}} & "
        r"\multicolumn{2}{c}{\textbf{Agent Split}} \\"
    )
    lines.append(
        r"             & \textbf{Samples} & \textbf{Total Tok} & \textbf{Mean Tok} & "
        r"\textbf{Total Lat(s)} & \textbf{Mean Lat(s)} & \textbf{AtkTok} & \textbf{DefTok} \\"
    )
    lines.append(r"            \midrule")

    sorted_row_keys = sorted(rows)
    for row_key in sorted_row_keys:
        _, _, _, _, variant_display, mode, model = row_key
        runtime = rows[row_key].get("runtime_statistics", {})
        row_label = latex_escape(format_row_label(variant_display, mode, model))

        if not runtime.get("available"):
            row_cells = [row_label, "0", "-", "-", "-", "-", "-", "-"]
        else:
            row_cells = [
                row_label,
                str(int(runtime["covered_sample_count"])),
                str(int(runtime["total_tokens"])),
                f"{runtime['mean_total_tokens']:.0f}",
                f"{runtime['total_latency_s']:.2f}",
                f"{runtime['mean_latency_s']:.2f}",
                f"{runtime['mean_attacker_tokens']:.0f}",
                f"{runtime['mean_defender_tokens']:.0f}",
            ]

        lines.append("            " + " & ".join(row_cells) + r" \\")

    lines.append(r"            \bottomrule")
    lines.append(r"        \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"\end{table}")
    lines.append(r"")
    return "\n".join(lines)


def build_document(suite_tables):
    if not suite_tables:
        raise ValueError("No CourtGuard aggregate rows were found in the metrics file.")

    lines = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=0.5in, landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{multirow}",
        r"\usepackage{amssymb}",
        r"\usepackage{float}",
        r"\usepackage{xcolor}",
        r"",
        r"\begin{document}",
        r"\section*{CourtGuard Aggregate Evaluation Tables}",
        r"",
    ]

    emitted_any_suite = False
    ordered_suite_names = [suite for suite in SUITE_ORDER if suite in suite_tables]
    ordered_suite_names.extend(sorted(suite for suite in suite_tables if suite not in SUITE_ORDER))

    for suite_name in ordered_suite_names:
        suite_payload = suite_tables[suite_name]
        rows = suite_payload["rows"]
        dataset_names = suite_payload["dataset_names"]
        if not rows or not dataset_names:
            continue

        emitted_any_suite = True
        lines.append(render_suite_table(rows, dataset_names, suite_name))
        lines.append(render_runtime_suite_table(rows, suite_name))

    if not emitted_any_suite:
        raise ValueError("No suite-specific CourtGuard aggregate rows were found in the metrics file.")

    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregate LaTeX tables for CourtGuard metrics from results_metrics.json"
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS_PATH,
        help="Path to the aggregated metrics JSON file",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output .tex file",
    )
    args = parser.parse_args()

    metrics_path = os.path.abspath(args.metrics)
    output_path = os.path.abspath(args.output)

    entries = load_metrics_entries(metrics_path)
    suite_tables = collect_aggregate_data(entries)
    document = build_document(suite_tables)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(document)

    print(f"Wrote aggregate LaTeX table to: {output_path}")


if __name__ == "__main__":
    main()
