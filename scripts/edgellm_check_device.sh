#!/usr/bin/env bash
set -u

CONFIG=${CONFIG:-configs/edgellm_llama31_8b_precision_sweep.yaml}
export PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin:$PATH

echo "===== TensorRT EdgeLLM Device Check ====="
python - "$CONFIG" <<'PY'
import os
import shutil
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
repo = Path(os.path.expanduser(cfg["edgellm_repo"]))
workspace = Path(os.path.expanduser(cfg["workspace_dir"]))
print("EdgeLLM repo:", repo, "exists=", repo.exists())
for rel in ["build/examples/llm/llm_build", "build/examples/llm/llm_inference"]:
    path = repo / rel
    print(rel + ":", path, "exists=", path.exists())
print("workspace:", workspace, "exists=", workspace.exists())
for tool in ["nvidia-smi", "tegrastats", "nsys", "ncu"]:
    print(tool + ":", shutil.which(tool))
PY

echo
echo "===== GPU ====="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 || true
else
    echo "nvidia-smi not found; this is common on Jetson/Thor-style devices."
    command -v tegrastats >/dev/null 2>&1 && echo "tegrastats found: $(command -v tegrastats)"
fi

echo
echo "===== CUDA tools ====="
nvcc --version 2>&1 || true
ncu --version 2>&1 || true
nsys --version 2>&1 || true
