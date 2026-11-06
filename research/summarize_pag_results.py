"""Summarize PAG evaluation CSV files by condition and category."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


BOOL_FIELDS = [
    "unsafe_public_disclosure",
    "over_refusal",
    "denied_action_suppressed",
    "permission_correct",
    "action_consistent",
    "proposed_permitted_gap",
]

SUMMARY_FIELDS = [
    "group",
    "model",
    "condition",
    "category",
    "generation_mode",
    "n",
    "parse_success_rate",
    "request_failure_rate",
    "denied_prompt_nonrefusal_rate",
    "unsafe_public_disclosure_rate",
    "over_refusal_rate",
    "denied_action_suppression_rate",
    "permission_accuracy",
    "action_consistency_rate",
    "proposed_permitted_gap_recall",
    "latency_mean_s",
    "latency_p95_s",
    "latency_max_s",
    "top_failure_type",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bool_rate(rows: list[dict[str, str]], field: str) -> float:
    values = [row.get(field) == "True" for row in rows if row.get(field) in {"True", "False"}]
    if not values:
        return 0.0
    return sum(values) / len(values)


def nonempty_rate(rows: list[dict[str, str]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get(field, "")) for row in rows) / len(rows)


def parse_rate(rows: list[dict[str, str]]) -> float:
    values = [row["parse_ok"] == "True" for row in rows if row.get("parse_ok") in {"True", "False"}]
    if not values:
        return 0.0
    return sum(values) / len(values)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def top_failure(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.get("failure_type", "unknown")] += 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else "none"


def summarize_group(
    *,
    group_name: str,
    condition: str,
    model: str,
    category: str,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s")]
    modes = sorted({row.get("generation_mode", "") for row in rows})
    return {
        "group": group_name,
        "model": model,
        "condition": condition,
        "category": category,
        "generation_mode": "+".join(modes),
        "n": str(len(rows)),
        "parse_success_rate": f"{parse_rate(rows):.4f}",
        "request_failure_rate": f"{nonempty_rate(rows, 'error'):.4f}",
        "denied_prompt_nonrefusal_rate": f"{bool_rate(rows, 'denied_prompt_nonrefusal'):.4f}",
        "unsafe_public_disclosure_rate": f"{bool_rate(rows, 'unsafe_public_disclosure'):.4f}",
        "over_refusal_rate": f"{bool_rate(rows, 'over_refusal'):.4f}",
        "denied_action_suppression_rate": f"{bool_rate(rows, 'denied_action_suppressed'):.4f}",
        "permission_accuracy": f"{bool_rate(rows, 'permission_correct'):.4f}",
        "action_consistency_rate": f"{bool_rate(rows, 'action_consistent'):.4f}",
        "proposed_permitted_gap_recall": f"{bool_rate(rows, 'proposed_permitted_gap'):.4f}",
        "latency_mean_s": f"{statistics.mean(latencies):.3f}" if latencies else "0.000",
        "latency_p95_s": f"{p95(latencies):.3f}",
        "latency_max_s": f"{max(latencies):.3f}" if latencies else "0.000",
        "top_failure_type": top_failure(rows),
    }


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_model_condition: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_model_condition_category: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_condition[row["condition"]].append(row)
        model = row.get("model", "unknown")
        by_model_condition[(model, row["condition"])].append(row)
        by_model_condition_category[(model, row["condition"], row["category"])].append(row)

    for condition, group_rows in sorted(by_condition.items()):
        output.append(
            summarize_group(
                group_name="condition",
                condition=condition,
                model="ALL",
                category="ALL",
                rows=group_rows,
            )
        )

    for (model, condition), group_rows in sorted(by_model_condition.items()):
        output.append(
            summarize_group(
                group_name="model_condition",
                condition=condition,
                model=model,
                category="ALL",
                rows=group_rows,
            )
        )

    for (model, condition, category), group_rows in sorted(by_model_condition_category.items()):
        output.append(
            summarize_group(
                group_name="model_condition_category",
                condition=condition,
                model=model,
                category=category,
                rows=group_rows,
            )
        )

    return output


def validate_row_count(rows: list[dict[str, str]], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"expected {expected} observations, found {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/pag_eval_mock.csv")
    parser.add_argument("--out", default="results/pag_eval_mock_summary.csv")
    parser.add_argument("--expected-rows", type=int, default=7200)
    args = parser.parse_args()

    rows = read_rows(Path(args.csv))
    validate_row_count(rows, args.expected_rows)
    summary = summarize(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    print(f"rows: {len(rows)}")
    print(f"summary rows: {len(summary)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
