"""Export PAG summary rows to a compact Markdown table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    ("condition", "Condition"),
    ("n", "N"),
    ("unsafe_public_disclosure_rate", "Unsafe"),
    ("denied_prompt_nonrefusal_rate", "Denied non-refusal"),
    ("over_refusal_rate", "Over-refusal"),
    ("parse_success_rate", "Parse"),
    ("request_failure_rate", "Request fail"),
    ("latency_mean_s", "Mean latency"),
    ("latency_p95_s", "P95 latency"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="PAG Results")
    args = parser.parse_args()

    with Path(args.summary).open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["group"] == "model_condition"
        ]

    lines = [f"# {args.title}", ""]
    lines.append("| Model | " + " | ".join(header for _, header in COLUMNS) + " |")
    lines.append("|---|" + "|".join("---" for _ in COLUMNS) + "|")
    for row in rows:
        values = [row.get(field, "") for field, _ in COLUMNS]
        lines.append("| " + row.get("model", "") + " | " + " | ".join(values) + " |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
