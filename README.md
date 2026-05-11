# NVFP4 vs FP16 Bottleneck Shift in LLM Inference

This repository contains the experiment scaffold we used to study how inference bottlenecks change when moving from FP16 to NVFP4 on NVIDIA Thor/Blackwell-class hardware using TensorRT Edge-LLM.

The final experiment design is organized around two model roles:

- **Llama 3.1 8B Instruct**: baseline, long-context, and extra-long-context studies for understanding how bottlenecks evolve as prompt length and KV-cache size increase.
- **Qwen2.5-14B Instruct**: cross-model comparison under baseline and long-context settings to test whether the FP16-to-NVFP4 speedup pattern generalizes beyond a single model family.

The project focuses on **bottleneck shift**, not just raw speedup. We compare:

- stage-level behavior in **prefill** and **decode**
- kernel-family changes under **FP16** and **NVFP4**
- context scaling from baseline to long and extra-long prompt regimes

## Repository Layout

### Configs

- `configs/edgellm_llama31_8b_baseline.yaml`  
  Llama 3.1 8B baseline configuration.

- `configs/edgellm_llama31_8b_longctx.yaml`  
  Llama 3.1 8B long-context configuration.

- `configs/edgellm_llama31_8b_exlongctx.yaml`  
  Llama 3.1 8B extra-long-context configuration driven by a very long prompt file.

- `configs/edgellm_qwen25_14b_baseline.yaml`  
  Qwen2.5-14B baseline configuration.

- `configs/edgellm_qwen25_14b_longctx.yaml`  
  Qwen2.5-14B long-context configuration.

### Scripts

- `scripts/edgellm_export_host.sh`  
  Export or quantize-and-export the selected model to TensorRT Edge-LLM ONNX format.

- `scripts/edgellm_build_device.sh`  
  Build TensorRT engines with the configured `max_input_len` and `max_kv_cache_capacity`.

- `scripts/edgellm_run_device.sh`  
  Run inference normally, with `nsys`, or with `ncu`. The `nsys` path includes CUDA graph node tracing so decode-stage graph nodes are visible in the timeline.

- `scripts/edgellm_enable_nvtx_device.sh`  
  Apply the local NVTX instrumentation patch to TensorRT Edge-LLM on the target machine.

- `scripts/query_nsys_sqlite.py`  
  Helper script for extracting `LLM_PREFILL`, `LLM_GENERATION`, and `Decode_Iter` timing from exported Nsight Systems SQLite traces.

- `scripts/prompt.txt`  
  Long-form prompt used for Qwen long-context experiments.

- `scripts/exlong_prompt.txt`  
  Extra-long prompt used for the Llama 3.1 8B extra-long-context experiments.

### Patches

- `patches/tensorrt-edgellm-nvtx-decode-ranges.patch`  
  Adds additional NVTX ranges for cleaner prefill/decode profiling.

- `patches/tensorrt-edgellm-input-limit-512k.patch`  
  Raises the Edge-LLM JSON message content size limit from `128 * 1024` to `512 * 1024` bytes so extra-long prompt inputs can be parsed by `llm_inference`.

## Environment

For NVIDIA Thor/Tegra-style `aarch64` systems, we use a user-managed Conda environment on top of the host CUDA and TensorRT installation rather than a TensorRT-LLM container.

The host development system provided:

```text
CUDA 13.0 tools
TensorRT 10.13.3.9
Nsight Systems
Nsight Compute
standard build toolchain
```

Inside the `trtllm` Conda environment, we used:

```text
Python 3.12
PyTorch 2.10.0+cu130
TensorRT-LLM 1.1.0
Transformers 4.57.6
OpenMPI from conda-forge
huggingface_hub
safetensors
sentencepiece
accelerate
datasets
```

TensorRT Edge-LLM was cloned and built from source against this combined host-plus-Conda setup.

Important environment settings:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export TLLM_WORKER_USE_SINGLE_PROCESS=1
```

`TLLM_WORKER_USE_SINGLE_PROCESS=1` helps avoid the default MPI worker path for single-GPU testing on Thor.

If the Conda environment needs to reuse the host TensorRT Python bindings:

```bash
echo /usr/lib/python3.12/dist-packages > ~/miniforge3/envs/trtllm/lib/python3.12/site-packages/system-tensorrt.pth
```

To install Python-side dependencies:

```bash
pip install -r requirements.txt
```

For gated Llama checkpoints:

```bash
huggingface-cli login
```

## Why TensorRT Edge-LLM

On Thor, the regular TensorRT-LLM Python LLM path was not the stable choice for this project. TensorRT Edge-LLM successfully supported the export/build/run flow we needed for FP16 and NVFP4 experiments, including stage-level and kernel-level profiling.

The effective workflow is:

```text
Hugging Face model
-> quantize/export to Edge-LLM ONNX
-> build TensorRT engine on Thor
-> run C++ inference on Thor
-> profile with Nsight Systems / Nsight Compute
```

## Current Experiment Structure

### Llama 3.1 8B

- `baseline`: moderate prompt length
- `longctx`: long prompt / large KV-cache regime
- `exlongctx`: extra-long prompt regime used to study more severe context growth

This model is used primarily to study how bottlenecks move as context increases.

### Qwen2.5-14B

- `baseline`: moderate prompt length
- `longctx`: long prompt regime

This model is used primarily for cross-model comparison and to test whether the FP16-to-NVFP4 speedup pattern remains similar across model families.

## Local Edge-LLM Modifications

In addition to the normal TensorRT Edge-LLM flow, this project used a few local modifications for profiling and extra-long input support:

1. **NVTX decode instrumentation**
   - Added additional NVTX ranges around decode forward and decode sampling.
   - This makes Nsight Systems and Nsight Compute filtering cleaner for stage-level analysis.

2. **CUDA graph node tracing in `nsys`**
   - `scripts/edgellm_run_device.sh` adds:

   ```text
   --cuda-graph-trace=node
   ```

   - This improves visibility into decode-stage GEMM and related graph nodes in Nsight Systems.

3. **Expanded JSON message content limit**
   - `patches/tensorrt-edgellm-input-limit-512k.patch` raises:

   ```cpp
   constexpr size_t kMaxMessageContentSizeBytes
   ```

   from `128 * 1024` to `512 * 1024`.

   - This is required for the extra-long prompt experiments, where the input JSON payload itself becomes very large.

## Export, Build, and Run

Check the device:

```bash
CONFIG=configs/edgellm_llama31_8b_baseline.yaml ./scripts/edgellm_check_device.sh
```

Export a model:

```bash
CONFIG=configs/edgellm_qwen25_14b_baseline.yaml ./scripts/edgellm_export_host.sh fp16
CONFIG=configs/edgellm_qwen25_14b_baseline.yaml ./scripts/edgellm_export_host.sh nvfp4
```

Build the engine:

```bash
CONFIG=configs/edgellm_qwen25_14b_baseline.yaml ./scripts/edgellm_build_device.sh fp16
CONFIG=configs/edgellm_qwen25_14b_baseline.yaml ./scripts/edgellm_build_device.sh nvfp4
```

Run inference:

```bash
CONFIG=configs/edgellm_qwen25_14b_baseline.yaml ./scripts/edgellm_run_device.sh fp16
CONFIG=configs/edgellm_qwen25_14b_baseline.yaml ./scripts/edgellm_run_device.sh nvfp4
```

The same pattern applies to:

- `configs/edgellm_llama31_8b_baseline.yaml`
- `configs/edgellm_llama31_8b_longctx.yaml`
- `configs/edgellm_llama31_8b_exlongctx.yaml`
- `configs/edgellm_qwen25_14b_longctx.yaml`

## Profiling

### Nsight Systems

Nsight Systems is used to measure:

- `LLM_PREFILL`
- `LLM_GENERATION`
- `Decode_Iter`

Example:

```bash
PROFILE=nsys PROFILE_TAG=full CONFIG=configs/edgellm_qwen25_14b_baseline.yaml \
  ./scripts/edgellm_run_device.sh fp16
```

The resulting `.sqlite` files can be queried with:

```bash
python scripts/query_nsys_sqlite.py <trace.sqlite>
```

### Nsight Compute

Nsight Compute is used to inspect kernel-level behavior in:

- **prefill**
- **decode iteration**

We filter NCU collection using NVTX ranges so that prefill and decode can be analyzed separately.

Example prefill run:

```bash
nohup env PROFILE=ncu PROFILE_TAG=prefill NCU_SET=full NCU_NVTX_INCLUDE='regex:LLM_PREFILL.*/' NCU_LAUNCH_SKIP=0 NCU_LAUNCH_COUNT=50 CONFIG=configs/edgellm_qwen25_14b_longctx.yaml \
  ./scripts/edgellm_run_device.sh nvfp4 \
  > results/qwen25_14b_longctx_nvfp4_prefill_ncu.log 2>&1 &
```

Example decode run:

```bash
nohup env PROFILE=ncu PROFILE_TAG=decodeiter NCU_SET=full NCU_NVTX_INCLUDE='regex:Decode_Iter.*/' NCU_LAUNCH_SKIP=0 NCU_LAUNCH_COUNT=100 CONFIG=configs/edgellm_qwen25_14b_longctx.yaml \
  ./scripts/edgellm_run_device.sh nvfp4 \
  > results/qwen25_14b_longctx_nvfp4_decodeiter_ncu.log 2>&1 &
```

## What We Compare

The project is designed around **bottleneck shift**, not only total speedup.

The main comparisons are:

- **FP16 vs NVFP4** under the same workload
- **baseline vs long-context / extra-long-context** within the same model
- **Llama 3.1 8B vs Qwen2.5-14B** under comparable baseline and long-context conditions

At the kernel level, the most important categories are:

- GEMM / fused GEMM kernels
- FMHA / attention kernels
- KV-update kernels such as `applyRopeWriteKV`
- memory-movement or layout kernels such as cast / reshape / move kernels

This allows the study to answer not only whether NVFP4 is faster, but also:

- whether GEMM remains dominant after precision reduction
- whether attention becomes more important as context increases
- whether speedup drops in extra-long-context regimes where KV-cache traffic matters more

## Notes

- NVFP4 is a Blackwell-oriented precision mode; unsupported devices should be expected to fail or skip NVFP4 paths.
- TensorRT Edge-LLM clearly exposes **FP8 KV-cache** support in its own codebase, but our current experiments do not enable a dedicated low-precision KV-cache mode. Our current NVFP4 results therefore mainly reflect acceleration of the dominant low-precision compute path rather than a separately quantized NVFP4 KV cache.
- Extra-long-context experiments may require both:
  - a larger `max_input_len` / `max_kv_cache_capacity` build configuration
  - the local `inputLimits.h` patch recorded in `patches/tensorrt-edgellm-input-limit-512k.patch`
