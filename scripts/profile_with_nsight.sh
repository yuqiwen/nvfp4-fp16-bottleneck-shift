#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/llama31_8b_precision_sweep.yaml}
PRECISION=${1:-fp16}
TOOL=${2:-nsys}
OUT_DIR=${OUT_DIR:-results/nsight}
BATCH_SIZE=${BATCH_SIZE:-1}
WORKLOAD=${WORKLOAD:-balanced}
WARMUP_ITERS=${WARMUP_ITERS:-1}
TIMED_ITERS=${TIMED_ITERS:-1}

case "${WORKLOAD}" in
  prefill)
    PROMPT_TOKENS=${PROMPT_TOKENS:-4096}
    OUTPUT_TOKENS=${OUTPUT_TOKENS:-1}
    ;;
  decode)
    PROMPT_TOKENS=${PROMPT_TOKENS:-128}
    OUTPUT_TOKENS=${OUTPUT_TOKENS:-512}
    ;;
  balanced)
    PROMPT_TOKENS=${PROMPT_TOKENS:-1024}
    OUTPUT_TOKENS=${OUTPUT_TOKENS:-128}
    ;;
  *)
    echo "WORKLOAD must be one of: prefill, decode, balanced" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"
export TLLM_WORKER_USE_SINGLE_PROCESS=${TLLM_WORKER_USE_SINGLE_PROCESS:-1}

BASE_CMD=(
  python scripts/run_precision_sweep.py
  --config "${CONFIG}"
  --only "${PRECISION}"
  --batch-size "${BATCH_SIZE}"
  --prompt-tokens "${PROMPT_TOKENS}"
  --max-output-tokens "${OUTPUT_TOKENS}"
  --warmup-iters "${WARMUP_ITERS}"
  --timed-iters "${TIMED_ITERS}"
  --output-path "${OUT_DIR}/${PRECISION}_${WORKLOAD}_${TOOL}.jsonl"
)

case "${TOOL}" in
  nsys)
    nsys profile \
      --trace=cuda,nvtx,osrt,cublas,cudnn \
      --cuda-memory-usage=true \
      --stats=true \
      --force-overwrite=true \
      --kill=none \
      -o "${OUT_DIR}/${PRECISION}_${WORKLOAD}_b${BATCH_SIZE}_p${PROMPT_TOKENS}_o${OUTPUT_TOKENS}" \
      "${BASE_CMD[@]}"
    ;;
  ncu)
    ncu \
      --target-processes all \
      --set "${NCU_SET:-roofline}" \
      --launch-skip "${NCU_LAUNCH_SKIP:-20}" \
      --launch-count "${NCU_LAUNCH_COUNT:-50}" \
      --force-overwrite \
      -o "${OUT_DIR}/${PRECISION}_${WORKLOAD}_b${BATCH_SIZE}_p${PROMPT_TOKENS}_o${OUTPUT_TOKENS}" \
      "${BASE_CMD[@]}"
    ;;
  *)
    echo "Usage: $0 <fp16|fp8|nvfp4> <nsys|ncu>" >&2
    exit 2
    ;;
esac
