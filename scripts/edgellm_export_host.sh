#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/edgellm_llama31_8b_precision_sweep.yaml}
ONLY=${1:-all}

export PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-/usr/local/cuda-13.0/bin/ptxas}
export TRITON_PTXAS_BLACKWELL_PATH=${TRITON_PTXAS_BLACKWELL_PATH:-/usr/local/cuda-13.0/bin/ptxas}

python - "$CONFIG" "$ONLY" <<'PY'
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
model = cfg["model"]["hf_id"]
model_name = cfg["model"]["name"]
workspace.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)

for exp in cfg["experiments"]:
    if not exp.get("enabled", True):
        continue
    if only != "all" and only not in {exp["name"].lower(), exp["precision"].lower()}:
        continue

    name = exp["name"]
    precision = exp["precision"]
    quantization = str(exp["quantization"]).lower()
    root = workspace / model_name / precision
    onnx_dir = root / "onnx" / "llm"

    if quantization in {"none", "fp16"}:
        run([
            "tensorrt-edgellm-export-llm",
            "--model_dir", model,
            "--output_dir", onnx_dir,
        ])
    else:
        quantized_dir = root / "quantized"
        run([
            "tensorrt-edgellm-quantize-llm",
            "--model_dir", model,
            "--quantization", quantization,
            "--output_dir", quantized_dir,
        ])
        run([
            "tensorrt-edgellm-export-llm",
            "--model_dir", quantized_dir,
            "--output_dir", onnx_dir,
        ])

    print(f"Exported {name} to {onnx_dir}", flush=True)
PY
