"""Export paper-ready PAG result tables from completed summaries."""

from __future__ import annotations

import csv
from pathlib import Path


MODELS = [
    {
        "label": "Gemma 2B Q8",
        "key": "gemma2_q8",
        "orig_full": "pag_eval_llamacpp_gemma2_q8_full_summary.csv",
        "orig_schema": "pag_eval_llamacpp_gemma2_q8_schema_policy_full_summary.csv",
        "v2_full": "pag_runtime_v2_gemma2_q8_full_summary.csv",
        "v2_schema": "pag_runtime_v2_gemma2_q8_schema_policy_full_summary.csv",
    },
    {
        "label": "Qwen2.5 Coder 7B",
        "key": "qwen25_coder7b",
        "orig_full": "pag_eval_llamacpp_qwen25_coder7b_full_summary.csv",
        "orig_schema": "pag_eval_llamacpp_qwen25_coder7b_schema_policy_full_summary.csv",
        "v2_full": "pag_runtime_v2_qwen25_coder7b_full_summary.csv",
        "v2_schema": "pag_runtime_v2_qwen25_coder7b_schema_policy_full_summary.csv",
    },
    {
        "label": "Llama 3.1 8B",
        "key": "llama31_8b",
        "orig_full": "pag_eval_llamacpp_llama31_8b_full_summary.csv",
        "orig_schema": "pag_eval_llamacpp_llama31_8b_schema_policy_full_summary.csv",
        "v2_full": "pag_runtime_v2_llama31_8b_full_summary.csv",
        "v2_schema": "pag_runtime_v2_llama31_8b_schema_policy_full_summary.csv",
    },
    {
        "label": "Mistral 7B",
        "key": "mistral_7b",
        "orig_full": "pag_eval_llamacpp_mistral_7b_full_summary.csv",
        "orig_schema": "pag_eval_llamacpp_mistral_7b_schema_policy_full_summary.csv",
        "v2_full": "pag_runtime_v2_mistral_7b_full_summary.csv",
        "v2_schema": "pag_runtime_v2_mistral_7b_schema_policy_full_summary.csv",
    },
    {
        "label": "Gemma2 9B",
        "key": "gemma2_9b",
        "orig_full": "pag_eval_llamacpp_gemma2_9b_full_summary.csv",
        "orig_schema": "pag_eval_llamacpp_gemma2_9b_schema_policy_full_summary.csv",
        "v2_full": "pag_runtime_v2_gemma2_9b_full_summary.csv",
        "v2_schema": "pag_runtime_v2_gemma2_9b_schema_policy_full_summary.csv",
    },
]


PAPER_FIELDS = [
    "model",
    "condition",
    "controller",
    "n",
    "unsafe_disclosure_rate",
    "denied_nonrefusal_rate",
    "over_refusal_rate",
    "permission_accuracy",
    "latency_mean_s",
    "source",
]


def read_rows(name: str) -> list[dict[str, str]]:
    path = Path("results") / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_original(rows: list[dict[str, str]], condition: str) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("group") == "model_condition"
            and row.get("category") == "ALL"
            and row.get("condition") == condition
        ):
            return row
    return None


def find_v2(rows: list[dict[str, str]], condition: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("category") == "ALL" and row.get("condition") == condition:
            return row
    return None


def original_record(
    model: str,
    row: dict[str, str],
    *,
    controller: str,
    source: str,
) -> dict[str, str]:
    return {
        "model": model,
        "condition": row["condition"],
        "controller": controller,
        "n": row["n"],
        "unsafe_disclosure_rate": row["unsafe_public_disclosure_rate"],
        "denied_nonrefusal_rate": row["denied_prompt_nonrefusal_rate"],
        "over_refusal_rate": row["over_refusal_rate"],
        "permission_accuracy": row["permission_accuracy"],
        "latency_mean_s": row["latency_mean_s"],
        "source": source,
    }


def v2_record(
    model: str,
    row: dict[str, str],
    *,
    condition_label: str,
    source: str,
    latency: str,
) -> dict[str, str]:
    return {
        "model": model,
        "condition": condition_label,
        "controller": "runtime_v2",
        "n": row["n"],
        "unsafe_disclosure_rate": row["unsafe_public_disclosure_rate_v2"],
        "denied_nonrefusal_rate": row["denied_prompt_nonrefusal_rate_v2"],
        "over_refusal_rate": row["over_refusal_rate_v2"],
        "permission_accuracy": row["permission_accuracy_v2"],
        "latency_mean_s": latency,
        "source": source,
    }


def main() -> None:
    out_rows: list[dict[str, str]] = []

    for model in MODELS:
        label = model["label"]
        orig_full = read_rows(model["orig_full"])
        orig_schema = read_rows(model["orig_schema"])
        v2_full = read_rows(model["v2_full"])
        v2_schema = read_rows(model["v2_schema"])

        latency_by_condition = {}
        for condition in [
            "direct_llm",
            "safety_prompt",
            "schema_only",
            "negotiated_agency",
        ]:
            row = find_original(orig_full, condition)
            if row:
                latency_by_condition[condition] = row["latency_mean_s"]
                out_rows.append(
                    original_record(
                        label,
                        row,
                        controller="original",
                        source=model["orig_full"],
                    )
                )

        row = find_original(orig_schema, "schema_policy")
        if row:
            latency_by_condition["schema_policy"] = row["latency_mean_s"]
            out_rows.append(
                original_record(
                    label,
                    row,
                    controller="model_generated_policy",
                    source=model["orig_schema"],
                )
            )

        row = find_v2(v2_full, "negotiated_agency")
        if row:
            out_rows.append(
                v2_record(
                    label,
                    row,
                    condition_label="negotiated_agency_runtime_v2",
                    source=model["v2_full"],
                    latency=latency_by_condition.get("negotiated_agency", ""),
                )
            )

        row = find_v2(v2_schema, "schema_policy")
        if row:
            out_rows.append(
                v2_record(
                    label,
                    row,
                    condition_label="schema_policy_runtime_v2",
                    source=model["v2_schema"],
                    latency=latency_by_condition.get("schema_policy", ""),
                )
            )

    out_path = Path("results/pag_paper_model_comparison.csv")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    md_path = Path("docs/research/pag-paper-model-comparison.md")
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# PAG Paper Model Comparison\n\n")
        handle.write(
            "Phi-3.5 Mini is excluded from the main comparison because structured "
            "conditions failed under the current llama.cpp JSON-schema setup.\n\n"
        )
        handle.write(
            "| Model | Condition | Controller | N | Unsafe | Non-refusal | Over-refusal | Permission Acc. | Latency |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in out_rows:
            handle.write(
                "| {model} | {condition} | {controller} | {n} | {unsafe_disclosure_rate} | {denied_nonrefusal_rate} | {over_refusal_rate} | {permission_accuracy} | {latency_mean_s} |\n".format(
                    **row
                )
            )

    print(f"wrote {out_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
