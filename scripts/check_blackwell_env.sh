#!/usr/bin/env bash
set -u

section() {
  printf "\n===== %s =====\n" "$1"
}

try() {
  local label=$1
  shift
  printf "\n--- %s ---\n" "${label}"
  "$@" 2>&1 || true
}

section "Host"
try "OS" bash -lc 'cat /etc/os-release 2>/dev/null || uname -a'
try "Kernel" uname -a
try "Disk space" df -h .

section "NVIDIA GPU / Driver"
try "nvidia-smi" nvidia-smi
try "GPU names" bash -lc 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null'
try "CUDA compiler" nvcc --version

section "Containers"
try "docker version" docker --version
try "docker info" docker info
try "nvidia container runtime" bash -lc 'docker info 2>/dev/null | grep -i "nvidia\\|runtimes"'
try "docker GPU smoke test" docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

section "Profilers"
try "nsys" nsys --version
try "ncu" ncu --version

section "Git"
try "git" git --version
try "git-lfs" git lfs version

section "Python"
try "python" python --version
try "python3" python3 --version
try "pip" python -m pip --version

section "Python packages, current environment"
try "torch" python -c 'import torch; print(torch.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda", torch.version.cuda)'
try "tensorrt_llm" python -c 'import tensorrt_llm; print(tensorrt_llm.__version__ if hasattr(tensorrt_llm, "__version__") else "import ok")'
try "tensorrt" python -c 'import tensorrt as trt; print(trt.__version__)'
try "transformers" python -c 'import transformers; print(transformers.__version__)'
try "huggingface_hub" python -c 'import huggingface_hub; print(huggingface_hub.__version__)'

section "Hugging Face"
try "HF token presence" bash -lc 'if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then echo "HF token env var is set"; else echo "No HF token env var detected"; fi'
try "huggingface-cli" huggingface-cli whoami

section "Done"
echo "Paste this full output back into the thread or send it to Eric."
