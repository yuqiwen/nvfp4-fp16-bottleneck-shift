#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/edgellm_llama31_8b_precision_sweep.yaml}
ONLY=${1:-all}

python - "$CONFIG" "$ONLY" <<'PY'
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
only = sys.argv[2].lower()
cfg = yaml.safe_load(config_path.read_text())

workspace = Path(os.path.expanduser(cfg["workspace_dir"]))
repo = Path(os.path.expanduser(cfg["edgellm_repo"]))
model_name = cfg["model"]["name"]
build = cfg["build"]
run_cfg = cfg["run"]
prompt_repeat = int(run_cfg.get("prompt_repeat", 1))
prompt_file = run_cfg.get("prompt_file")
if prompt_file:
    prompt_path = Path(prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = (config_path.parent / prompt_path).resolve()
    prompt_base = prompt_path.read_text().strip()
else:
    prompt_base = str(run_cfg["prompt"]).strip()
prompt_text = ((prompt_base + "\n\n") * prompt_repeat).strip()

llm_build = repo / "build" / "examples" / "llm" / "llm_build"
if not llm_build.exists():
    raise SystemExit(f"Missing EdgeLLM builder: {llm_build}")

def run(cmd, cwd=None):
    print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True, cwd=cwd)

for exp in cfg["experiments"]:
    if not exp.get("enabled", True):
        continue
    if only != "all" and only not in {exp["name"].lower(), exp["precision"].lower()}:
        continue

    precision = exp["precision"]
    root = workspace / model_name / precision
    onnx_dir = root / "onnx" / "llm"
    engine_dir = root / "engines" / "llm"
    config_file = onnx_dir / "config.json"
    if not config_file.exists():
        raise SystemExit(
            f"ONNX export is incomplete for {precision}: missing {config_file}. "
            f"Run edgellm_export_host.sh {precision} and only build after export succeeds."
        )
    engine_dir.mkdir(parents=True, exist_ok=True)

    run([
        llm_build,
        "--onnxDir", onnx_dir,
        "--engineDir", engine_dir,
        "--maxBatchSize", build["max_batch_size"],
        "--maxInputLen", build["max_input_len"],
        "--maxKVCacheCapacity", build["max_kv_cache_capacity"],
    ], cwd=repo)

    input_file = root / "input.json"
    input_file.write_text(json.dumps({
        "batch_size": run_cfg["batch_size"],
        "temperature": run_cfg["temperature"],
        "top_p": run_cfg["top_p"],
        "top_k": run_cfg["top_k"],
        "max_generate_length": run_cfg["max_generate_length"],
        "requests": [{
            "messages": [
                {"role": "system", "content": "You are a concise GPU performance analyst."},
                {"role": "user", "content": prompt_text},
            ]
        }],
    }, indent=2) + "\n")

    print(f"Built {exp['name']} engine at {engine_dir}", flush=True)
    print(f"Wrote input file at {input_file}", flush=True)
PY
