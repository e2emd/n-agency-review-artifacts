"""Audit PAG observation CSVs for row count and key integrity."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    path = Path(args.csv)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    key_counts = Counter(
        (r["model"], r["run"], r["prompt_id"], r["condition"]) for r in rows
    )
    duplicate_keys = {key: count for key, count in key_counts.items() if count > 1}
    empty_required = {
        field: sum(1 for row in rows if not row.get(field))
        for field in ["run", "prompt_id", "category", "condition", "model", "generation_mode"]
    }

    print(f"file: {path}")
    print(f"rows: {len(rows)}")
    if args.expected_rows is not None:
        print(f"expected_rows: {args.expected_rows}")
        print(f"row_count_ok: {len(rows) == args.expected_rows}")
    print(f"duplicate_key_count: {len(duplicate_keys)}")
    print(f"empty_required: {empty_required}")
    print(f"run: {dict(Counter(r['run'] for r in rows))}")
    print(f"model: {dict(Counter(r['model'] for r in rows))}")
    print(f"condition: {dict(Counter(r['condition'] for r in rows))}")
    print(f"category: {dict(Counter(r['category'] for r in rows))}")
    print(f"generation_mode: {dict(Counter(r['generation_mode'] for r in rows))}")
    print(f"parse_failure: {sum(1 for r in rows if r.get('failure_type') == 'parse_failure')}")
    print(f"request_failure: {sum(1 for r in rows if r.get('failure_type') == 'request_failure')}")

    if duplicate_keys:
        raise SystemExit("duplicate observation keys found")
    if any(empty_required.values()):
        raise SystemExit("empty required fields found")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit("row count does not match expected rows")


if __name__ == "__main__":
    main()
