import argparse
import json
import math
import os
import re
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METRICS_PATH = os.path.join(SCRIPT_DIR, "results", "results_metrics.json")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "courtguard_tables.tex")
TARGET_SECTION = "section_4_opposite_imputing"
METRIC_KEYS = {
    "acc_f1": ("accuracy", "f1_score"),
    "recall_f2": ("recall", "f2_score"),
}
RUNTIME_KEYS = {
    "tokens_latency": ("mean_total_tokens", "mean_latency_s"),
    "agent_tokens": ("mean_attacker_tokens", "mean_defender_tokens"),
}
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


def shorten_dataset_name(dataset_name, max_len=12):
    pretty = prettify_identifier(dataset_name)
    compact = pretty.replace(" ", "")
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def build_dataset_labels(dataset_names):
    labels = {}
    used = set()

    for dataset_name in dataset_names:
        base = shorten_dataset_name(dataset_name)
        label = base
        suffix = 2
        while label in used:
            trimmed = base[:9] if len(base) > 9 else base
            label = f"{trimmed}{suffix}"
            suffix += 1
        labels[dataset_name] = label
        used.add(label)

    return labels


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


def collect_table_data(entries):
    suite_rows = defaultdict(lambda: defaultdict(dict))
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

        section_metrics = entry.get("evaluation_results", {}).get(TARGET_SECTION)
        if not section_metrics:
            continue

        runtime_stats = entry.get("runtime_statistics", {})
        per_sample_runtime = runtime_stats.get("per_sample", {}) if isinstance(runtime_stats, dict) else {}
        per_agent_runtime = runtime_stats.get("per_agent", {}) if isinstance(runtime_stats, dict) else {}
        attacker_runtime = per_agent_runtime.get("attacker", {}) if isinstance(per_agent_runtime, dict) else {}
        defender_runtime = per_agent_runtime.get("defender", {}) if isinstance(per_agent_runtime, dict) else {}

        dataset_name = parsed["dataset"]
        row_key = build_row_key(config, parsed)
        suite_rows[suite][row_key][dataset_name] = {
            "accuracy": section_metrics.get("accuracy"),
            "f1_score": section_metrics.get("f1_score"),
            "recall": section_metrics.get("recall"),
            "f2_score": section_metrics.get("f2_score"),
            "mean_total_tokens": per_sample_runtime.get("mean_total_tokens"),
            "mean_latency_s": per_sample_runtime.get("mean_latency_s"),
            "mean_attacker_tokens": attacker_runtime.get("mean_total_tokens"),
            "mean_defender_tokens": defender_runtime.get("mean_total_tokens"),
        }
        suite_dataset_names[suite].add(dataset_name)

    suite_tables = {}
    for suite_name, rows in suite_rows.items():
        suite_tables[suite_name] = {
            "rows": dict(rows),
            "dataset_names": sorted(suite_dataset_names[suite_name], key=str.lower),
        }

    return suite_tables


def find_best_values(rows, dataset_names, metric_keys, objective="max"):
    best_values = {}
    for dataset_name in dataset_names:
        for metric_key in metric_keys:
            values = []
            for row_metrics in rows.values():
                metric_value = row_metrics.get(dataset_name, {}).get(metric_key)
                if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                    values.append(metric_value)
            if not values:
                best_values[(dataset_name, metric_key)] = None
            elif objective == "min":
                best_values[(dataset_name, metric_key)] = min(values)
            else:
                best_values[(dataset_name, metric_key)] = max(values)
    return best_values


def format_metric(metric_value, best_value):
    if not isinstance(metric_value, (int, float)) or math.isnan(metric_value):
        return "-"

    rounded = f"{metric_value:.2f}"
    if best_value is not None and abs(metric_value - best_value) < 1e-12:
        return rf"\textbf{{{rounded}}}"
    return rounded


def format_runtime_metric(metric_key, metric_value, best_value):
    if not isinstance(metric_value, (int, float)) or math.isnan(metric_value):
        return "-"

    if "latency" in metric_key:
        rounded = f"{metric_value:.2f}"
    else:
        rounded = f"{metric_value:.0f}"

    if best_value is not None and abs(metric_value - best_value) < 1e-12:
        return rf"\textbf{{{rounded}}}"
    return rounded


def render_table(rows, dataset_names, dataset_labels, metric_pair_name, suite_name):
    first_metric, second_metric = METRIC_KEYS[metric_pair_name]
    best_values = find_best_values(rows, dataset_names, (first_metric, second_metric))
    suite_title = SUITE_TITLES.get(suite_name, prettify_identifier(suite_name))

    column_spec = "l" + ("cc" * len(dataset_names))
    metric_a_label = "Acc" if first_metric == "accuracy" else "Recall"
    metric_b_label = "F1" if second_metric == "f1_score" else "F2"
    caption = (
        f"{suite_title}: Accuracy and F1 across Available Datasets"
        if metric_pair_name == "acc_f1"
        else f"{suite_title}: Recall and F2 across Available Datasets"
    )
    label = (
        f"tab:courtguard_{suite_name}_acc_f1"
        if metric_pair_name == "acc_f1"
        else f"tab:courtguard_{suite_name}_recall_f2"
    )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append(r"    \resizebox{\linewidth}{!}{%")
    lines.append(f"        \\begin{{tabular}}{{{column_spec}}}")
    lines.append(r"            \toprule")

    first_header = [r"\multirow{2}{*}{\textbf{CourtGuard Config}}"]
    for dataset_name in dataset_names:
        dataset_label = latex_escape(dataset_labels[dataset_name])
        first_header.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{dataset_label}}}}}")
    lines.append("            " + " & ".join(first_header) + r" \\")

    cmidrules = []
    start_column = 2
    for _ in dataset_names:
        end_column = start_column + 1
        cmidrules.append(rf"\cmidrule(lr){{{start_column}-{end_column}}}")
        start_column += 2
    lines.append("            " + " ".join(cmidrules))

    second_header = []
    for _ in dataset_names:
        second_header.extend([rf"\textbf{{{metric_a_label}}}", rf"\textbf{{{metric_b_label}}}"])
    lines.append("            & " + " & ".join(second_header) + r" \\")
    lines.append(r"            \midrule")

    sorted_row_keys = sorted(rows)
    for row_key in sorted_row_keys:
        _, _, _, _, variant_display, mode, model = row_key
        row_label = latex_escape(format_row_label(variant_display, mode, model))
        row_cells = [row_label]
        row_metrics = rows[row_key]

        for dataset_name in dataset_names:
            dataset_metrics = row_metrics.get(dataset_name, {})
            row_cells.append(
                format_metric(
                    dataset_metrics.get(first_metric),
                    best_values[(dataset_name, first_metric)],
                )
            )
            row_cells.append(
                format_metric(
                    dataset_metrics.get(second_metric),
                    best_values[(dataset_name, second_metric)],
                )
            )

        lines.append("            " + " & ".join(row_cells) + r" \\")

    lines.append(r"            \bottomrule")
    lines.append(r"        \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def render_runtime_table(rows, dataset_names, dataset_labels, metric_pair_name, suite_name):
    first_metric, second_metric = RUNTIME_KEYS[metric_pair_name]
    best_values = find_best_values(rows, dataset_names, (first_metric, second_metric), objective="min")
    suite_title = SUITE_TITLES.get(suite_name, prettify_identifier(suite_name))

    column_spec = "l" + ("cc" * len(dataset_names))
    if metric_pair_name == "tokens_latency":
        metric_a_label = "Tok"
        metric_b_label = "Lat(s)"
        caption = f"{suite_title}: Mean Total Tokens and Latency across Available Datasets"
        label = f"tab:courtguard_{suite_name}_runtime_tokens_latency"
    else:
        metric_a_label = "AtkTok"
        metric_b_label = "DefTok"
        caption = f"{suite_title}: Mean Attacker and Defender Tokens across Available Datasets"
        label = f"tab:courtguard_{suite_name}_runtime_agent_tokens"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append(r"    \resizebox{\linewidth}{!}{%")
    lines.append(f"        \\begin{{tabular}}{{{column_spec}}}")
    lines.append(r"            \toprule")

    first_header = [r"\multirow{2}{*}{\textbf{CourtGuard Config}}"]
    for dataset_name in dataset_names:
        dataset_label = latex_escape(dataset_labels[dataset_name])
        first_header.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{dataset_label}}}}}")
    lines.append("            " + " & ".join(first_header) + r" \\")

    cmidrules = []
    start_column = 2
    for _ in dataset_names:
        end_column = start_column + 1
        cmidrules.append(rf"\cmidrule(lr){{{start_column}-{end_column}}}")
        start_column += 2
    lines.append("            " + " ".join(cmidrules))

    second_header = []
    for _ in dataset_names:
        second_header.extend([rf"\textbf{{{metric_a_label}}}", rf"\textbf{{{metric_b_label}}}"])
    lines.append("            & " + " & ".join(second_header) + r" \\")
    lines.append(r"            \midrule")

    sorted_row_keys = sorted(rows)
    for row_key in sorted_row_keys:
        _, _, _, _, variant_display, mode, model = row_key
        row_label = latex_escape(format_row_label(variant_display, mode, model))
        row_cells = [row_label]
        row_metrics = rows[row_key]

        for dataset_name in dataset_names:
            dataset_metrics = row_metrics.get(dataset_name, {})
            row_cells.append(
                format_runtime_metric(
                    first_metric,
                    dataset_metrics.get(first_metric),
                    best_values[(dataset_name, first_metric)],
                )
            )
            row_cells.append(
                format_runtime_metric(
                    second_metric,
                    dataset_metrics.get(second_metric),
                    best_values[(dataset_name, second_metric)],
                )
            )

        lines.append("            " + " & ".join(row_cells) + r" \\")

    lines.append(r"            \bottomrule")
    lines.append(r"        \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def build_document(suite_tables):
    if not suite_tables:
        raise ValueError("No CourtGuard entries were found in the metrics file.")

    lines = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=0.5in, landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{multirow}",
        r"\usepackage{float}",
        r"",
        r"\begin{document}",
        r"",
        r"\section*{CourtGuard Evaluation Tables}",
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
        dataset_labels = build_dataset_labels(dataset_names)
        suite_title = latex_escape(SUITE_TITLES.get(suite_name, prettify_identifier(suite_name)))

        lines.append(rf"\section*{{{suite_title}}}")
        lines.append("")
        lines.append(render_table(rows, dataset_names, dataset_labels, "acc_f1", suite_name))
        lines.append("")
        lines.append(render_table(rows, dataset_names, dataset_labels, "recall_f2", suite_name))
        lines.append("")
        lines.append(render_runtime_table(rows, dataset_names, dataset_labels, "tokens_latency", suite_name))
        lines.append("")
        lines.append(render_runtime_table(rows, dataset_names, dataset_labels, "agent_tokens", suite_name))
        lines.append("")

    if not emitted_any_suite:
        raise ValueError("No suite-specific CourtGuard entries were found in the metrics file.")

    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate flexible LaTeX tables for CourtGuard metrics from results_metrics.json"
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
    suite_tables = collect_table_data(entries)
    document = build_document(suite_tables)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(document)

    print(f"Wrote LaTeX tables to: {output_path}")


if __name__ == "__main__":
    main()
