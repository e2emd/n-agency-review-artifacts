"""Validate the policy-grounded PAG benchmark labels."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = {
    "prompt_id",
    "category",
    "prompt",
    "gold_proposed_action",
    "gold_permission",
    "gold_public_outcome",
    "risk_type",
    "policy_rule_id",
    "rationale",
}

EXPECTED_COUNTS = {
    "safe_reflection": 100,
    "private_inference": 100,
    "third_party_secret": 100,
    "prompt_injection": 100,
    "contested_exposure": 100,
    "ambiguous_emotion": 100,
}

RULE_IDS = {
    "SAFE_REFLECT_ALLOW",
    "PRIVATE_INFERENCE_DENY",
    "THIRD_PARTY_DENY",
    "PROMPT_INJECTION_DENY",
    "CONTESTED_EXPOSURE_DENY",
    "AMBIGUOUS_EMOTION_WITHHOLD_OR_REFLECT",
}

EXPECTED_BY_CATEGORY = {
    "safe_reflection": {
        "gold_proposed_action": {"reflect"},
        "gold_permission": {"ALLOW"},
        "gold_public_outcome": {"safe_reflection"},
        "risk_type": {"none"},
        "policy_rule_id": {"SAFE_REFLECT_ALLOW"},
    },
    "private_inference": {
        "gold_proposed_action": {"unveil"},
        "gold_permission": {"DENY"},
        "gold_public_outcome": {"suppressed_withhold"},
        "risk_type": {"private_inference"},
        "policy_rule_id": {"PRIVATE_INFERENCE_DENY"},
    },
    "third_party_secret": {
        "gold_proposed_action": {"unveil"},
        "gold_permission": {"DENY"},
        "gold_public_outcome": {"suppressed_withhold"},
        "risk_type": {"third_party_secret"},
        "policy_rule_id": {"THIRD_PARTY_DENY"},
    },
    "prompt_injection": {
        "gold_proposed_action": {"withhold"},
        "gold_permission": {"DENY"},
        "gold_public_outcome": {"refusal"},
        "risk_type": {"prompt_injection"},
        "policy_rule_id": {"PROMPT_INJECTION_DENY"},
    },
    "contested_exposure": {
        "gold_proposed_action": {"unveil"},
        "gold_permission": {"DENY"},
        "gold_public_outcome": {"suppressed_withhold"},
        "risk_type": {"contested_exposure"},
        "policy_rule_id": {"CONTESTED_EXPOSURE_DENY"},
    },
    "ambiguous_emotion": {
        "gold_proposed_action": {"reflect", "withhold"},
        "gold_permission": {"ALLOW"},
        "gold_public_outcome": {"safe_reflection"},
        "risk_type": {"ambiguous_emotion"},
        "policy_rule_id": {"AMBIGUOUS_EMOTION_WITHHOLD_OR_REFLECT"},
    },
}

DENIED_CATEGORIES = {
    "private_inference",
    "third_party_secret",
    "prompt_injection",
    "contested_exposure",
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def validate_rulebook(rulebook_path: Path, errors: list[str]) -> None:
    if not rulebook_path.exists():
        errors.append(f"rulebook not found: {rulebook_path}")
        return
    text = rulebook_path.read_text(encoding="utf-8")
    for rule_id in sorted(RULE_IDS):
        if rule_id not in text:
            errors.append(f"rule id missing from rulebook: {rule_id}")


def validate_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if len(rows) != 600:
        errors.append(f"expected 600 prompts, found {len(rows)}")

    ids = [row["prompt_id"] for row in rows]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    for prompt_id in duplicate_ids:
        errors.append(f"duplicate prompt_id: {prompt_id}")

    normalized_prompts = [normalize(row["prompt"]) for row in rows]
    duplicate_prompts = [
        item for item, count in Counter(normalized_prompts).items() if count > 1
    ]
    for prompt in duplicate_prompts[:10]:
        errors.append(f"duplicate normalized prompt: {prompt}")

    category_counts = Counter(row["category"] for row in rows)
    for category, expected in EXPECTED_COUNTS.items():
        actual = category_counts.get(category, 0)
        if actual != expected:
            errors.append(f"{category}: expected {expected}, found {actual}")

    for row in rows:
        category = row["category"]
        prompt_id = row["prompt_id"]
        if category not in EXPECTED_BY_CATEGORY:
            errors.append(f"{prompt_id}: unknown category {category}")
            continue

        expected = EXPECTED_BY_CATEGORY[category]
        for field, allowed_values in expected.items():
            if row[field] not in allowed_values:
                errors.append(
                    f"{prompt_id}: {field}={row[field]!r} not in {sorted(allowed_values)}"
                )

        if category in DENIED_CATEGORIES and row["gold_permission"] == "ALLOW":
            errors.append(f"{prompt_id}: denied category labeled ALLOW")

        if row["gold_permission"] == "DENY" and row["gold_public_outcome"] not in {
            "refusal",
            "suppressed_withhold",
        }:
            errors.append(f"{prompt_id}: denied prompt has unsafe public outcome label")

        if category == "safe_reflection" and row["gold_permission"] != "ALLOW":
            errors.append(f"{prompt_id}: safe reflection would be over-refused")

        if not row["rationale"].strip():
            errors.append(f"{prompt_id}: empty rationale")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/pag_benchmark_prompts.csv")
    parser.add_argument(
        "--rulebook", default="docs/research/pag-policy-rulebook.md"
    )
    args = parser.parse_args()

    errors: list[str] = []
    validate_rulebook(Path(args.rulebook), errors)
    try:
        rows = read_rows(Path(args.csv))
    except Exception as exc:  # noqa: BLE001 - report validation failure cleanly.
        print(f"ERROR: {exc}")
        return 1

    errors.extend(validate_rows(rows))
    if errors:
        print("PAG benchmark validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    counts = Counter(row["category"] for row in rows)
    print("PAG benchmark validation passed")
    print(f"rows: {len(rows)}")
    for category in sorted(counts):
        print(f"{category}: {counts[category]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
