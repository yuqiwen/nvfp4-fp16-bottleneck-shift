# NVFP4 vs FP16 GPU bottleneck shift benchmark

这个目录是给 Llama 3.3 70B 做不同精度推理对比的实验脚手架，目标是后面搬到 Blackwell 机器上跑 NVFP4，同时先把 FP16/FP8/NVFP4 的代码路径和输出格式固定下来。

结论先放前面：如果研究对象是 NVIDIA GPU 上的 LLM inference，尤其是 Blackwell 的 NVFP4，TensorRT-LLM + TensorRT Model Optimizer 是最直接的路线。普通 TensorRT 更偏底层 engine/runtime，LLM 这类多卡、KV cache、paged attention、quantized checkpoint 的工作建议用 TensorRT-LLM。

## Files

- `configs/llama33_70b_precision_sweep.yaml`: 一份默认实验配置。
- `scripts/run_precision_sweep.py`: 统一跑 FP16 / FP8 / NVFP4 的 benchmark driver。
- `requirements.txt`: Python 依赖占位。真正部署时建议优先用 NVIDIA TensorRT-LLM NGC container。

## Environment

推荐在 Linux GPU 机器上使用 NVIDIA TensorRT-LLM container。Windows 本机通常适合写代码和准备配置，不适合作为 70B TensorRT-LLM benchmark 环境。

最少需要：

```bash
pip install -r requirements.txt
```

如果用 Hugging Face 上的 gated Llama 3.3 权重或 NVIDIA 发布的量化权重，先登录并确认 license：

```bash
huggingface-cli login
```

## Prepare models

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

先编辑 `configs/llama33_70b_precision_sweep.yaml` 里的模型路径、tensor parallel size、batch/input/output token 长度，然后运行：

```bash
python scripts/run_precision_sweep.py --config configs/llama33_70b_precision_sweep.yaml
```

结果会写到：

```text
results/precision_sweep.jsonl
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
