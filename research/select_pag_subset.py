"""Select deterministic balanced subsets from the PAG benchmark."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_balanced(rows: list[dict[str, str]], per_category: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    selected: list[dict[str, str]] = []
    for category in sorted(grouped):
        category_rows = sorted(grouped[category], key=lambda item: item["prompt_id"])
        if len(category_rows) < per_category:
            raise ValueError(
                f"{category}: requested {per_category}, only {len(category_rows)} available"
            )
        selected.extend(category_rows[:per_category])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", default="data/pag_benchmark_prompts.csv")
    parser.add_argument("--out", default="data/pag_benchmark_balanced_calibration.csv")
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()

    rows = read_rows(Path(args.in_csv))
    selected = select_balanced(rows, args.per_category)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    print(
        f"selected {len(selected)} prompts "
        f"({args.per_category} per category) -> {out_path}"
    )


if __name__ == "__main__":
    main()
