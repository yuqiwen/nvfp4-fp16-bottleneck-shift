#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/edgellm_llama31_8b_precision_sweep.yaml}
PRECISION=${1:-fp16}
PROFILE=${PROFILE:-none}

export PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin:$PATH

python - "$CONFIG" "$PRECISION" "$PROFILE" <<'PY'
import os
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
precision = sys.argv[2].lower()
profile = sys.argv[3].lower()
cfg = yaml.safe_load(config_path.read_text())

workspace = Path(os.path.expanduser(cfg["workspace_dir"]))
repo = Path(os.path.expanduser(cfg["edgellm_repo"]))
model_name = cfg["model"]["name"]
run_cfg = cfg["run"]
run_name = str(run_cfg.get("name", "default")).lower().replace(" ", "_")
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
profile_tag = os.environ.get("PROFILE_TAG") or os.environ.get("NCU_PROFILE_TAG") or ""
profile_tag = profile_tag.lower().strip().replace(" ", "_")
profile_suffix = f"_{profile_tag}" if profile_tag else ""

llm_inference = repo / "build" / "examples" / "llm" / "llm_inference"
if not llm_inference.exists():
    raise SystemExit(f"Missing EdgeLLM runtime: {llm_inference}")

root = workspace / model_name / precision
engine_dir = root / "engines" / "llm"
run_dir = root / "runs" / run_name
input_file = run_dir / "input.json"
output_file = run_dir / "output.json"
profile_dir = root / "profiles" / run_name
if not engine_dir.exists():
    raise SystemExit(
        f"Engine directory is missing for {precision}: {engine_dir}. "
        f"Run edgellm_build_device.sh {precision} after a successful export."
    )
run_dir.mkdir(parents=True, exist_ok=True)
profile_dir.mkdir(parents=True, exist_ok=True)
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

app_cmd = [
    llm_inference,
    "--engineDir", engine_dir,
    "--inputFile", input_file,
    "--outputFile", output_file,
]

if profile == "nsys":
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt",
        "--cuda-memory-usage=true",
        "--cuda-graph-trace=node",
        "--stats=true",
        "--force-overwrite=true",
        "-o", profile_dir / f"{precision}_{run_name}{profile_suffix}_nsys",
        *app_cmd,
    ]
elif profile == "ncu":
    ncu_name = f"{precision}_{run_name}{profile_suffix}_ncu"
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--set", os.environ.get("NCU_SET", "roofline"),
        "--launch-skip", os.environ.get("NCU_LAUNCH_SKIP", "10"),
        "--launch-count", os.environ.get("NCU_LAUNCH_COUNT", "50"),
        "--force-overwrite",
        "-o", profile_dir / ncu_name,
    ]
    nvtx_include = os.environ.get("NCU_NVTX_INCLUDE")
    if nvtx_include:
        cmd.extend(["--nvtx", "--nvtx-include", nvtx_include])
        range_filter = os.environ.get("NCU_RANGE_FILTER")
        if range_filter:
            cmd.extend(["--range-filter", range_filter])
    cmd = [*cmd, *app_cmd]
elif profile != "none":
    raise SystemExit("PROFILE must be one of: none, nsys, ncu")
else:
    cmd = app_cmd

print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
subprocess.run([str(x) for x in cmd], check=True, cwd=repo)
print(f"Input: {input_file}", flush=True)
print(f"Output: {output_file}", flush=True)
PY
