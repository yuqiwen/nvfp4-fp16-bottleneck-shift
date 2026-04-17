# NVFP4 vs FP16 GPU bottleneck shift benchmark

这个目录是给 Llama 做不同精度推理对比的实验脚手架，目标是后面搬到 Blackwell/Thor 机器上跑 NVFP4，同时先把 FP16/FP8/NVFP4 的代码路径和输出格式固定下来。当前推荐先用 Llama 3.1 8B 跑通 pipeline，再回到 Llama 3.3 70B。


## Files

- `configs/llama31_8b_precision_sweep.yaml`: 推荐的默认实验配置，适合先在单 GPU 上跑通。
- `configs/llama33_70b_precision_sweep.yaml`: Llama 3.3 70B 配置，适合后续多卡大显存机器。
- `scripts/run_precision_sweep.py`: 统一跑 FP16 / FP8 / NVFP4 的 benchmark driver。
- `requirements.txt`: Python 依赖占位。真正部署时建议优先用 NVIDIA TensorRT-LLM NGC container。

## Environment

推荐在 Linux GPU 机器上使用 NVIDIA TensorRT-LLM container。Windows 本机通常适合写代码和准备配置，不适合作为 70B TensorRT-LLM benchmark 环境。

### NVIDIA Thor / aarch64 conda path

如果机器是 NVIDIA Thor / Tegra / `aarch64`，可以不走 Docker，直接用 conda 环境。当前已验证可 import 的组合是：

```text
Python 3.12
PyTorch 2.9.0+cu130
TensorRT 10.13.3.9 from host system packages
TensorRT-LLM 1.1.0
Transformers 4.56.0
OpenMPI from conda-forge
```

关键点：

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export TLLM_WORKER_USE_SINGLE_PROCESS=1
```

`TLLM_WORKER_USE_SINGLE_PROCESS=1` 对单 GPU / TP=1 很重要，可以避免 TensorRT-LLM 默认用 `MpiPoolSession` spawn worker 时在 Thor 环境里卡住。

TensorRT 的 Python binding 在 host 系统路径里：

```text
/usr/lib/python3.12/dist-packages
```

如果 conda env 里要复用 host TensorRT，可以加：

```bash
echo /usr/lib/python3.12/dist-packages > ~/miniforge3/envs/trtllm/lib/python3.12/site-packages/system-tensorrt.pth
```

这台机器上 TensorRT-LLM 1.2.0 会尝试安装 TensorRT 10.14 pip wheel，但 Tegra 平台不支持该 TensorRT wheel；因此当前先用 TensorRT-LLM 1.1.0 对齐 host TensorRT 10.13.3。

先在 Blackwell 机器上检查已有环境：

```bash
chmod +x scripts/check_blackwell_env.sh
./scripts/check_blackwell_env.sh | tee blackwell_env_check.txt
```

把 `blackwell_env_check.txt` 里的缺项整理给管理员即可。

最少需要：

```bash
pip install -r requirements.txt
```

如果用 Hugging Face 上的 gated Llama 3.3 权重或 NVIDIA 发布的量化权重，先登录并确认 license：

```bash
huggingface-cli login
```

## Prepare models

推荐先用 Llama 3.1 8B：

```yaml
FP16/BF16 baseline: meta-llama/Llama-3.1-8B-Instruct
FP8: nvidia/Llama-3.1-8B-Instruct-FP8
NVFP4: nvidia/Llama-3.1-8B-Instruct-NVFP4
```

FP16/BF16 baseline 可以直接指向 HF model id 或本地目录：

```yaml
model: meta-llama/Llama-3.3-70B-Instruct
```

现在 Hugging Face 上已经有 NVIDIA 发布的 Llama 3.3 70B 量化 checkpoint，可以先直接用：

```yaml
model: nvidia/Llama-3.3-70B-Instruct-NVFP4
precision: nvfp4
quantization:
  mode: prequantized
```

FP8 对应：

```yaml
model: nvidia/Llama-3.3-70B-Instruct-FP8
precision: fp8
quantization:
  mode: prequantized
  kv_cache: fp8
```

如果后面需要自己控制 calibration dataset、scaling policy 或 TensorRT-LLM/ModelOpt 版本，再单独生成自己的 FP8/NVFP4 checkpoint。

## Run

### TensorRT Edge-LLM on Thor

For NVIDIA Thor, use TensorRT Edge-LLM instead of the regular TensorRT-LLM Python LLM API.

Official TensorRT Edge-LLM docs describe the flow as:

```text
Hugging Face model -> quantize on x86 -> export ONNX on x86 -> transfer ONNX -> build engine on Thor -> run C++ inference on Thor
```

The Edge-LLM config for the three precision variants is:

```bash
configs/edgellm_llama31_8b_precision_sweep.yaml
```

On an x86 host with the Edge-LLM export tools:

```bash
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_export_host.sh fp16
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_export_host.sh fp8
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_export_host.sh nvfp4
```

Then transfer the exported ONNX folders to Thor under the same workspace path.

On Thor:

```bash
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_check_device.sh
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_build_device.sh fp16
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_run_device.sh fp16
```

For profiling:

```bash
PROFILE=nsys ./scripts/edgellm_run_device.sh fp16
PROFILE=ncu NCU_LAUNCH_SKIP=10 NCU_LAUNCH_COUNT=50 ./scripts/edgellm_run_device.sh fp16
```

For cleaner prefill/decode separation, enable Edge-LLM's built-in NVTX ranges once after cloning/building Edge-LLM:

```bash
CONFIG=configs/edgellm_llama31_8b_precision_sweep.yaml ./scripts/edgellm_enable_nvtx_device.sh
```

The Edge-LLM runtime instrumentation used by this script is also saved as a repository patch:

```text
patches/tensorrt-edgellm-nvtx-decode-ranges.patch
```

This patch records our local Edge-LLM changes: adding decode-forward and decode-sampling NVTX ranges in `llmInferenceRuntime.cpp`, plus a small NVTX color-definition fix needed by the current Edge-LLM source.

Then NCU can filter by runtime stage instead of relying only on global kernel launch numbers:

```bash
# Prefill only
nohup env PROFILE=ncu NCU_SET=full NCU_NVTX_INCLUDE='regex:LLM_PREFILL.*/' NCU_LAUNCH_SKIP=0 NCU_LAUNCH_COUNT=50 CONFIG=configs/edgellm_llama31_8b_prefill.yaml \
  ./scripts/edgellm_run_device.sh fp16 \
  > results/fp16_prefill_ncu_full_nvtx.log 2>&1 &

# Decode forward only
nohup env PROFILE=ncu NCU_SET=full NCU_NVTX_INCLUDE='regex:LLM_DECODE_FORWARD.*/' NCU_LAUNCH_SKIP=0 NCU_LAUNCH_COUNT=50 CONFIG=configs/edgellm_llama31_8b_decode.yaml \
  ./scripts/edgellm_run_device.sh fp16 \
  > results/fp16_decode_ncu_full_nvtx.log 2>&1 &
```

The trailing `/` in `NCU_NVTX_INCLUDE` is required because Edge-LLM emits push/pop NVTX ranges. `LLM_DECODE_FORWARD` captures the decode model-forward path, while `LLM_DECODE_SAMPLING` can be used separately for topK/sampling overhead.

Repeat `fp8` and `nvfp4` after the corresponding engines are built.

### TensorRT-LLM Python API fallback

先编辑 `configs/llama33_70b_precision_sweep.yaml` 里的模型路径、tensor parallel size、batch/input/output token 长度，然后运行：

```bash
python scripts/run_precision_sweep.py --config configs/llama31_8b_precision_sweep.yaml
```

结果会写到：

```text
results/llama31_8b_precision_sweep.jsonl
```

每行是一条实验，包括：

- precision / model / backend
- batch size、prompt/output token 估计
- end-to-end latency
- token throughput
- GPU peak memory

## Profiling bottleneck shift

这个脚手架默认记录端到端指标。真正写 paper/report 时建议同一组配置再用 Nsight Systems / Nsight Compute 包一层：

```bash
nsys profile -o results/fp16_nsys python scripts/run_precision_sweep.py --config configs/llama33_70b_precision_sweep.yaml --only fp16
nsys profile -o results/nvfp4_nsys python scripts/run_precision_sweep.py --config configs/llama33_70b_precision_sweep.yaml --only nvfp4
```

也可以用封装脚本。Nsight Systems 用来先看端到端 timeline：

```bash
WORKLOAD=balanced ./scripts/profile_with_nsight.sh fp16 nsys
WORKLOAD=balanced ./scripts/profile_with_nsight.sh fp8 nsys
WORKLOAD=balanced ./scripts/profile_with_nsight.sh nvfp4 nsys
```

Nsight Compute 用来抓少量 kernel 的硬件计数器，默认只抓 50 个 kernel launch，避免整段 decode 被极慢地 profile：

```bash
WORKLOAD=prefill NCU_LAUNCH_SKIP=20 NCU_LAUNCH_COUNT=50 ./scripts/profile_with_nsight.sh fp16 ncu
WORKLOAD=prefill NCU_LAUNCH_SKIP=20 NCU_LAUNCH_COUNT=50 ./scripts/profile_with_nsight.sh fp8 ncu
WORKLOAD=prefill NCU_LAUNCH_SKIP=20 NCU_LAUNCH_COUNT=50 ./scripts/profile_with_nsight.sh nvfp4 ncu

WORKLOAD=decode NCU_LAUNCH_SKIP=20 NCU_LAUNCH_COUNT=50 ./scripts/profile_with_nsight.sh fp16 ncu
WORKLOAD=decode NCU_LAUNCH_SKIP=20 NCU_LAUNCH_COUNT=50 ./scripts/profile_with_nsight.sh fp8 ncu
WORKLOAD=decode NCU_LAUNCH_SKIP=20 NCU_LAUNCH_COUNT=50 ./scripts/profile_with_nsight.sh nvfp4 ncu
```

`WORKLOAD` 默认是 `balanced`，可选：

- `prefill`: 4096 prompt tokens, 1 output token
- `decode`: 128 prompt tokens, 512 output tokens
- `balanced`: 1024 prompt tokens, 128 output tokens

也可以手动覆盖 token 数：

```bash
BATCH_SIZE=1 WORKLOAD=prefill PROMPT_TOKENS=8192 OUTPUT_TOKENS=1 ./scripts/profile_with_nsight.sh nvfp4 nsys
```

重点看：

- GEMM/Tensor Core utilization 是否提升。
- DRAM throughput 是否从主瓶颈变弱。
- dequant / quantize / KV cache kernel 是否变成新瓶颈。
- prefill 和 decode 阶段的瓶颈是否不同。

## Notes

- NVFP4 主要是 Blackwell 路线；在非 Blackwell 设备上应该预期跳过或失败。
- FP8 在 Hopper/Blackwell 上更成熟，适合作为中间精度对照组。
- 70B 做 FP16 通常需要多卡 tensor parallelism；默认配置按 4 卡起步写。
