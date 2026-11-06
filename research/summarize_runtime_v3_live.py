"""Summarize V3 live runtime audit outputs.

This script reads the per-model V3 audit CSVs produced by
run_runtime_v3_all_live.py and writes aggregate tables for the paper and
handover docs. It deliberately recomputes rates from boolean audit rows rather
than trusting already-rounded summary CSVs.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


BOOL_METRICS = [
    "unsafe_final_line",
    "unsafe_intent",
    "unsafe_raw_output",
    "unsafe_public_surface",
    "denied_nonrefusal",
    "over_refusal",
    "intent_visible_on_denied",
    "raw_output_leak",
    "parse_failure_leak",
    "ui_side_channel_leak",
]

SUMMARY_FIELDS = [
    "variant",
    "model",
    "category",
    "n",
    "unsafe_final_line_rate",
    "unsafe_intent_rate",
    "unsafe_raw_output_rate",
    "unsafe_public_surface_rate",
    "denied_nonrefusal_rate",
    "over_refusal_rate",
    "intent_visible_on_denied_rate",
    "raw_output_leak_rate",
    "parse_failure_leak_rate",
    "ui_side_channel_leak_rate",
    "parse_failure_count",
    "latency_mean_s",
    "top_failure_type",
]


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def read_rows(patterns: list[str]) -> list[dict[str, str]]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [candidate]
        files.extend(path for path in matches if "smoke" not in path.name)

    deduped = sorted(set(files))
    if not deduped:
        raise SystemExit("no audit files matched")

    rows: list[dict[str, str]] = []
    for path in deduped:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    return rows


def summarize_group(rows: list[dict[str, str]]) -> dict[str, str]:
    n = len(rows)
    latency_values = [float(row["latency_s"]) for row in rows if row.get("latency_s")]
    failures = Counter(row.get("failure_type", "none") for row in rows)
    parse_failure_count = sum(1 for row in rows if row.get("parse_ok") == "False")

    summary: dict[str, str] = {
        "n": str(n),
        "parse_failure_count": str(parse_failure_count),
        "latency_mean_s": f"{(sum(latency_values) / len(latency_values)):.3f}"
        if latency_values
        else "",
        "top_failure_type": failures.most_common(1)[0][0] if failures else "none",
    }
    for metric in BOOL_METRICS:
        count = sum(1 for row in rows if bool_value(row.get(metric, "")))
        summary[f"{metric}_rate"] = f"{count / n:.4f}" if n else "0.0000"
    return summary


def grouped_summary(
    rows: list[dict[str, str]], keys: tuple[str, ...]
) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    output: list[dict[str, str]] = []
    for key_values, group_rows in sorted(groups.items()):
        summary = summarize_group(group_rows)
        record = {key: value for key, value in zip(keys, key_values)}
        record.update(summary)
        output.append(record)
    return output


def with_all_rows(
    rows: list[dict[str, str]], keys: tuple[str, ...], all_keys: tuple[str, ...]
) -> list[dict[str, str]]:
    output = grouped_summary(rows, keys)
    all_output = grouped_summary(rows, all_keys)
    for row in all_output:
        row["category"] = "ALL"
    return sorted(
        output + all_output,
        key=lambda row: tuple(
            row.get(key, "") if row.get(key, "") != "ALL" else ""
            for key in keys
        ),
    )


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |"]
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    by_model: list[dict[str, str]],
    by_category: list[dict[str, str]],
) -> None:
    all_rows = [row for row in by_model if row["category"] == "ALL"]
    aggregate_all = [row for row in by_category if row["category"] == "ALL"]

    key_fields = [
        "variant",
        "n",
        "unsafe_public_surface_rate",
        "unsafe_final_line_rate",
        "denied_nonrefusal_rate",
        "intent_visible_on_denied_rate",
        "raw_output_leak_rate",
        "parse_failure_leak_rate",
        "ui_side_channel_leak_rate",
    ]
    model_fields = [
        "variant",
        "model",
        "n",
        "unsafe_public_surface_rate",
        "denied_nonrefusal_rate",
        "intent_visible_on_denied_rate",
        "raw_output_leak_rate",
        "latency_mean_s",
    ]

    text = "\n".join(
        [
            "# PAG Runtime V3 Live Results",
            "",
            "Generated from `results/pag_runtime_v3_live_*_full_audit.csv`.",
            "",
            "Scope: five llama.cpp models, 600 prompts, three runs each, 9,000 live model generations. Each generation is audited under `v2_final_only`, `v2_public_surface`, and `v3_public_surface_controller`.",
            "",
            "## Aggregate",
            "",
            markdown_table(aggregate_all, key_fields),
            "",
            "## Model-Level ALL Rows",
            "",
            markdown_table(all_rows, model_fields),
            "",
            "## Interpretation",
            "",
            "- V3 eliminates measured unsafe public-surface disclosure in this benchmark, including final-line, unsafe-intent, raw-output, and parse-failure leaks.",
            "- V3 does not solve all policy compliance: denied prompts can still receive benign-looking non-refusal outputs when the model proposes a safe action and line.",
            "- `intent_visible_on_denied_rate` remains non-zero because the V3 controller is output-surface driven; it does not use a prompt-side admission classifier.",
            "- The defensible paper claim is runtime public-surface auditing and redaction, not complete safety or human-level judgment.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results/pag_runtime_v3_live_*_full_audit.csv"],
    )
    parser.add_argument(
        "--by-model-out",
        default="results/pag_runtime_v3_live_aggregate_by_model.csv",
    )
    parser.add_argument(
        "--by-category-out",
        default="results/pag_runtime_v3_live_aggregate_by_category.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/research/pag-runtime-v3-live-results.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs)
    by_model = with_all_rows(
        rows,
        ("variant", "model", "category"),
        ("variant", "model"),
    )
    by_category = with_all_rows(
        rows,
        ("variant", "category"),
        ("variant",),
    )
    write_csv(Path(args.by_model_out), by_model, SUMMARY_FIELDS)
    write_csv(
        Path(args.by_category_out),
        by_category,
        [field for field in SUMMARY_FIELDS if field != "model"],
    )
    write_markdown(Path(args.markdown_out), by_model, by_category)
    print(f"read {len(rows)} audit rows")
    print(f"wrote {args.by_model_out}")
    print(f"wrote {args.by_category_out}")
    print(f"wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
