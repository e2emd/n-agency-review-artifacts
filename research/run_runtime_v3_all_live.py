"""Run live V3 public-surface audit for all valid llama.cpp models.

This orchestrates one model at a time:

1. start llama-server with the model blob;
2. wait for the OpenAI-compatible endpoint;
3. run ``run_runtime_v3_live.py``;
4. stop the server before moving to the next model.

Phi-3.5 Mini is intentionally not included because prior PAG sweeps excluded it
from the valid comparison set after llama.cpp schema failures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


MODELS = [
    {
        "key": "gemma2_q8",
        "label": "gemma2-2b-q8-llamacpp-live",
        "blob": "sha256-9a31a1608e139e7ba6ceb61db8c49022eb26886d715363b21a01943f93fbc0d6",
    },
    {
        "key": "qwen25_coder7b",
        "label": "qwen25-coder-7b-llamacpp-live",
        "blob": "sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463",
    },
    {
        "key": "llama31_8b",
        "label": "llama31-8b-llamacpp-live",
        "blob": "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29",
    },
    {
        "key": "mistral_7b",
        "label": "mistral-7b-llamacpp-live",
        "blob": "sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f",
    },
    {
        "key": "gemma2_9b",
        "label": "gemma2-9b-llamacpp-live",
        "blob": "sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373",
    },
]


def llama_server_path() -> Path:
    return (
        Path(os.environ["LOCALAPPDATA"])
        / "Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe/llama-server.exe"
    )


def blob_path(blob: str) -> Path:
    return Path(os.environ["USERPROFILE"]) / ".ollama/models/blobs" / blob


def wait_ready(port: int, timeout_s: int = 180) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - readiness loop records last error.
            last_error = repr(exc)
        time.sleep(2)
    raise RuntimeError(f"llama-server did not become ready: {last_error}")


def run_model(args: argparse.Namespace, model: dict[str, str]) -> None:
    server = llama_server_path()
    blob = blob_path(model["blob"])
    if not server.exists():
        raise FileNotFoundError(server)
    if not blob.exists():
        raise FileNotFoundError(blob)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    server_out = open(log_dir / f"llama-server-v3-{model['key']}.out", "wb")
    server_err = open(log_dir / f"llama-server-v3-{model['key']}.err", "wb")
    server_args = [
        str(server),
        "-m",
        str(blob),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--parallel",
        "1",
    ]

    print(f"[{model['key']}] starting llama-server")
    process = subprocess.Popen(
        server_args,
        stdout=server_out,
        stderr=server_err,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    try:
        wait_ready(args.port, args.server_timeout_s)
        print(f"[{model['key']}] server ready")

        prefix = results_dir / f"pag_runtime_v3_live_{model['key']}_full"
        runner_out = open(log_dir / f"v3-live-{model['key']}.log", "wb")
        runner_err = open(log_dir / f"v3-live-{model['key']}.err", "wb")
        runner_args = [
            sys.executable,
            "-u",
            "research/run_runtime_v3_live.py",
            "--runs",
            str(args.runs),
            "--out",
            str(prefix) + "_raw.csv",
            "--audit-out",
            str(prefix) + "_audit.csv",
            "--summary-out",
            str(prefix) + "_summary.csv",
            "--model",
            model["label"],
            "--url",
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            "--timeout-s",
            str(args.timeout_s),
            "--max-tokens",
            str(args.max_tokens),
            "--progress-every",
            str(args.progress_every),
        ]
        if args.prompt_limit is not None:
            runner_args.extend(["--prompt-limit", str(args.prompt_limit)])
        if args.prompt_offset:
            runner_args.extend(["--prompt-offset", str(args.prompt_offset)])

        print(f"[{model['key']}] running V3 live")
        completed = subprocess.run(runner_args, stdout=runner_out, stderr=runner_err)
        if completed.returncode != 0:
            raise RuntimeError(f"{model['key']} runner failed with {completed.returncode}")
        print(f"[{model['key']}] done")
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)
        server_out.close()
        server_err.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(model["key"] for model in MODELS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--server-timeout-s", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-dir", default="results/logs")
    args = parser.parse_args()

    wanted = {item.strip() for item in args.models.split(",") if item.strip()}
    model_map = {model["key"]: model for model in MODELS}
    unknown = sorted(wanted - set(model_map))
    if unknown:
        raise ValueError(f"unknown model keys: {', '.join(unknown)}")

    for model in MODELS:
        if model["key"] in wanted:
            run_model(args, model)


if __name__ == "__main__":
    main()
