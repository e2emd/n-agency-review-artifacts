"""Report PAG CSV progress while long Ollama runs are active."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/pag_eval_ollama_balanced_calibration.csv")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"missing: {path}")
        return

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print(f"rows: {len(rows)}")
    if not rows:
        return

    for field in ["model", "condition", "category", "generation_mode"]:
        if field in rows[0]:
            print(f"{field}: {dict(sorted(Counter(row[field] for row in rows).items()))}")

    print(f"unsafe_public_disclosure: {sum(row['unsafe_public_disclosure'] == 'True' for row in rows)}")
    if "denied_prompt_nonrefusal" in rows[0]:
        print(f"denied_prompt_nonrefusal: {sum(row['denied_prompt_nonrefusal'] == 'True' for row in rows)}")
    print(f"over_refusal: {sum(row['over_refusal'] == 'True' for row in rows)}")
    print(f"parse_failure: {sum(row['parse_ok'] == 'False' for row in rows)}")
    if "error" in rows[0]:
        print(f"request_errors: {sum(bool(row.get('error')) for row in rows)}")


if __name__ == "__main__":
    main()
