# NVFP4 vs FP16 GPU Bottleneck Shift Benchmark

This repository contains a profiling scaffold for studying how reduced precision changes GPU bottlenecks during LLM inference. The current working target is Llama 3.1 8B Instruct on NVIDIA Thor/Blackwell-class hardware, with FP16 as the baseline and NVFP4 as the primary low-precision target. Llama 3.3 70B remains a follow-up scale-up target once the pipeline is stable.

## Repository Layout

- `configs/llama31_8b_precision_sweep.yaml`: default TensorRT-LLM precision sweep config for initial single-GPU testing.
- `configs/llama33_70b_precision_sweep.yaml`: larger Llama 3.3 70B config for future multi-GPU experiments.
- `configs/edgellm_llama31_8b_precision_sweep.yaml`: TensorRT Edge-LLM FP16/FP8/NVFP4 config.
- `configs/edgellm_llama31_8b_longctx.yaml`: long-context workload used for prefill/decode profiling.
- `scripts/run_precision_sweep.py`: Python benchmark driver for TensorRT-LLM and Transformers fallback runs.
- `scripts/edgellm_*.sh`: TensorRT Edge-LLM export, build, run, and profiling helpers.
- `scripts/check_blackwell_env.sh`: quick device/environment checker.
- `patches/tensorrt-edgellm-nvtx-decode-ranges.patch`: local Edge-LLM runtime instrumentation patch used for NVTX profiling.

## Environment

For standard x86 Linux GPU servers, NVIDIA TensorRT-LLM containers are usually the simplest option. For NVIDIA Thor/Tegra-style `aarch64` systems, this project uses a user-level conda/miniforge environment and the existing system CUDA/TensorRT installation.

The Thor environment used for development had:

```text
Python 3.12
PyTorch 2.9.0+cu130
TensorRT 10.13.3.9 from host system packages
TensorRT-LLM 1.1.0
Transformers 4.56.0
OpenMPI from conda-forge
CUDA 13.0 tools
Nsight Systems / Nsight Compute
```

Important environment settings:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export TLLM_WORKER_USE_SINGLE_PROCESS=1
```

`TLLM_WORKER_USE_SINGLE_PROCESS=1` is useful for single-GPU TensorRT-LLM runs because it avoids the default MPI worker spawn path, which was unstable on the Thor test system.

If the conda environment needs to reuse the host TensorRT Python bindings, add the host path to the environment:

```bash
echo /usr/lib/python3.12/dist-packages > ~/miniforge3/envs/trtllm/lib/python3.12/site-packages/system-tensorrt.pth
```

To inspect a new Blackwell/Thor machine:

```bash
chmod +x scripts/check_blackwell_env.sh
./scripts/check_blackwell_env.sh | tee blackwell_env_check.txt
```

For gated Llama or NVIDIA quantized checkpoints, authenticate with Hugging Face first:

```bash
huggingface-cli login
```

## Model Choices

Recommended initial model:

```text
FP16/BF16 baseline: meta-llama/Llama-3.1-8B-Instruct
FP8 target: optional Edge-LLM/ModelOpt export path
NVFP4 target: Edge-LLM/ModelOpt export path
```

Llama 3.1 8B is small enough to iterate on a single device while still exercising realistic transformer inference behavior: prefill attention, autoregressive decode, KV-cache updates, MLP GEMMs, fused normalization/cast kernels, and sampling overhead.

## TensorRT Edge-LLM on Thor

On NVIDIA Thor, use TensorRT Edge-LLM instead of the regular TensorRT-LLM Python LLM API. The standard TensorRT-LLM Python API hit unsupported fused-attention kernels on the target system, while Edge-LLM successfully built and ran FP16 and NVFP4 engines.

The Edge-LLM flow is:

```text
Hugging Face model -> quantize/export ONNX -> build TensorRT engine on Thor -> run C++ inference on Thor
```

Check the device:

```bash
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_check_device.sh
```

Export on the machine where the Edge-LLM Python export tools are available:

```bash
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_export_host.sh fp16
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_export_host.sh fp8
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_export_host.sh nvfp4
```

Build and run on Thor:

```bash
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_build_device.sh fp16
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_run_device.sh fp16
```

Repeat the build/run commands for `fp8` and `nvfp4` after export succeeds.

## Edge-LLM NVTX Instrumentation

For cleaner prefill/decode separation, enable Edge-LLM's NVTX profiling path and apply the local instrumentation patch:

```bash
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_enable_nvtx_device.sh
```

The runtime patch is stored in:

```text
patches/tensorrt-edgellm-nvtx-decode-ranges.patch
```

The patch records our local Edge-LLM changes:

- add `LLM_DECODE_FORWARD[...]` around decode model-forward enqueue;
- add `LLM_DECODE_SAMPLING[...]` around topK/sampling and token bookkeeping;
- add a small NVTX color-definition fix needed by the current Edge-LLM source.

These labels are used for profiling and are not intended to change model outputs.

## Profiling

Nsight Systems captures the full timeline:

```bash
PROFILE=nsys PROFILE_TAG=full CONFIG=configs/edgellm_llama31_8b_longctx.yaml \
  ./scripts/edgellm_run_device.sh fp16
```

Nsight Compute captures detailed kernel metrics. The trailing `/` in `NCU_NVTX_INCLUDE` is required because Edge-LLM emits push/pop NVTX ranges.

Prefill:

```bash
nohup env PROFILE=ncu PROFILE_TAG=prefill NCU_SET=full NCU_NVTX_INCLUDE='regex:LLM_PREFILL.*/' NCU_LAUNCH_SKIP=0 NCU_LAUNCH_COUNT=50 CONFIG=configs/edgellm_llama31_8b_longctx.yaml \
  ./scripts/edgellm_run_device.sh fp16 \
  > results/fp16_longctx_prefill_ncu_full_nvtx.log 2>&1 &
```

Decode forward:

```bash
nohup env PROFILE=ncu PROFILE_TAG=decodeforward NCU_SET=full NCU_NVTX_INCLUDE='regex:LLM_DECODE_FORWARD.*/' NCU_LAUNCH_SKIP=0 NCU_LAUNCH_COUNT=50 CONFIG=configs/edgellm_llama31_8b_longctx.yaml \
  ./scripts/edgellm_run_device.sh fp16 \
  > results/fp16_longctx_decodeforward_ncu_full_nvtx.log 2>&1 &
```

Sampling overhead:

```bash
nohup env PROFILE=ncu PROFILE_TAG=decodesampling NCU_SET=full NCU_NVTX_INCLUDE='regex:LLM_DECODE_SAMPLING.*/' NCU_LAUNCH_SKIP=0 NCU_LAUNCH_COUNT=20 CONFIG=configs/edgellm_llama31_8b_longctx.yaml \
  ./scripts/edgellm_run_device.sh fp16 \
  > results/fp16_longctx_decodesampling_ncu_full_nvtx.log 2>&1 &
```

Important interpretation note: TensorRT Edge-LLM uses CUDA graph execution and stream synchronization, so CPU-side NVTX wall time can include asynchronous enqueue and synchronization effects. Use Nsight Systems for stage/timeline structure and Nsight Compute for kernel-level bottleneck analysis.

## TensorRT-LLM Python API Fallback

The original TensorRT-LLM Python benchmark path is still available for non-Thor systems or smoke tests:

```bash
python scripts/run_precision_sweep.py --config configs/llama31_8b_precision_sweep.yaml
```

Results are written as JSONL:

```text
results/llama31_8b_precision_sweep.jsonl
```

Each row records precision, model, backend, batch size, prompt/output token estimates, latency, throughput, and peak GPU memory.

The Transformers fallback config can be used for a simple FP16 sanity check:

```bash
python scripts/run_precision_sweep.py --config configs/llama31_8b_transformers_smoke.yaml
```

## What to Compare

The main evaluation compares FP16 and NVFP4 under the same long-context workload:

- dominant kernel categories in prefill and decode;
- Tensor Core/GEMM utilization;
- DRAM/L2/L1 throughput;
- roofline position;
- scheduler and warp stall reasons;
- whether non-GEMM kernels such as KV-cache update, RoPE, fused cast/norm, quantization, or sampling become more visible after precision reduction.

The expected research question is not only whether NVFP4 is faster, but whether it shifts the bottleneck differently in prefill and decode.

## Notes

- NVFP4 is a Blackwell-oriented format; unsupported devices should be expected to fail or skip NVFP4 paths.
- FP8 is useful as an intermediate precision comparison, but the Edge-LLM FP8 export path may require additional memory/toolchain work.
- Llama 3.3 70B FP16 typically requires multi-GPU tensor parallelism; Llama 3.1 8B is the recommended first-pass model for pipeline validation.
