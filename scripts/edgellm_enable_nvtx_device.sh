#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/edgellm_llama31_8b_precision_sweep.yaml}
JOBS=${JOBS:-4}

python - "$CONFIG" <<'PY'
import os
import shutil
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
repo = Path(os.path.expanduser(cfg["edgellm_repo"]))
header = repo / "cpp" / "profiling" / "nvtx_wrapper.h"
runtime = repo / "cpp" / "runtime" / "llmInferenceRuntime.cpp"
if not header.exists():
    raise SystemExit(f"Missing Edge-LLM NVTX header: {header}")
if not runtime.exists():
    raise SystemExit(f"Missing Edge-LLM runtime source: {runtime}")

text = header.read_text()
if "constexpr auto PURPLE" not in text:
    backup = header.with_suffix(header.suffix + ".bak.rs")
    if not backup.exists():
        shutil.copy2(header, backup)
    marker = "constexpr auto MAGENTA = NVTX_RGB(255, 0, 255);"
    if marker not in text:
        raise SystemExit(f"Could not find color insertion point in {header}")
    text = text.replace(marker, marker + "\nconstexpr auto PURPLE = NVTX_RGB(160, 100, 255);")
    header.write_text(text)
    print(f"Patched missing PURPLE color in {header}")
else:
    print(f"PURPLE color already present in {header}")

runtime_text = runtime.read_text()
if "LLM_DECODE_FORWARD[" in runtime_text and "LLM_DECODE_SAMPLING[" in runtime_text:
    print(f"Decode forward/sampling NVTX ranges already present in {runtime}")
else:
    backup = runtime.with_suffix(runtime.suffix + ".bak.rs")
    if not backup.exists():
        shutil.copy2(runtime, backup)
    old = """            // Perform embedding lookup for the selected token indices (decode only has text, no images)
            kernel::embeddingLookup(mSelectedIndices, mEmbeddingTable, mInputsEmbeds, stream);

            // Use the embedded tokens as input for the decoding step.
            // No hidden states output needed for standard LLM decoding.
            rt::OptionalOutputTensor const outputHiddenStates{std::nullopt};
            bool decodingStatus = mLLMEngineRunner->executeVanillaDecodingStep(
                mInputsEmbeds, mOutputLogits, outputHiddenStates, stream);
            if (!decodingStatus)
            {
                LOG_ERROR("LLMInferenceRuntime(): Failed to execute decoding step.");
                return false;
            }

            sampleTokens();
"""
    new = """            {
                NVTX_SCOPED_RANGE(decode_forward_range,
                    ("LLM_DECODE_FORWARD[" + std::to_string(generationIter) + "/"
                        + std::to_string(maxGenerationLength) + ",Active=" + std::to_string(unFinishedBatchNum) + "]")
                        .c_str(),
                    nvtx_colors::ORANGE);

                // Perform embedding lookup for the selected token indices (decode only has text, no images)
                kernel::embeddingLookup(mSelectedIndices, mEmbeddingTable, mInputsEmbeds, stream);

                // Use the embedded tokens as input for the decoding step.
                // No hidden states output needed for standard LLM decoding.
                rt::OptionalOutputTensor const outputHiddenStates{std::nullopt};
                bool decodingStatus = mLLMEngineRunner->executeVanillaDecodingStep(
                    mInputsEmbeds, mOutputLogits, outputHiddenStates, stream);
                if (!decodingStatus)
                {
                    LOG_ERROR("LLMInferenceRuntime(): Failed to execute decoding step.");
                    return false;
                }
            }

            {
                NVTX_SCOPED_RANGE(decode_sampling_range,
                    ("LLM_DECODE_SAMPLING[" + std::to_string(generationIter) + "/"
                        + std::to_string(maxGenerationLength) + ",Active=" + std::to_string(unFinishedBatchNum) + "]")
                        .c_str(),
                    nvtx_colors::YELLOW);
                sampleTokens();
            }
"""
    if old not in runtime_text:
        raise SystemExit(f"Could not find decode loop insertion point in {runtime}")
    runtime.write_text(runtime_text.replace(old, new))
    print(f"Patched decode forward/sampling NVTX ranges in {runtime}")
PY

export PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin:$PATH

python - "$CONFIG" "$JOBS" <<'PY'
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
jobs = sys.argv[2]
repo = Path(os.path.expanduser(cfg["edgellm_repo"]))
build = repo / "build"
build.mkdir(parents=True, exist_ok=True)

cmake = os.environ.get("CMAKE", "cmake")
commands = [
    [
        cmake, "..",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTRT_PACKAGE_DIR=/usr",
        "-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake",
        "-DEMBEDDED_TARGET=jetson-thor",
        "-DENABLE_NVTX_PROFILING=ON",
    ],
    [cmake, "--build", ".", "--target", "llm_inference", f"-j{jobs}"],
]

for cmd in commands:
    print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=build)

print("Edge-LLM llm_inference rebuilt with NVTX profiling enabled.")
PY
