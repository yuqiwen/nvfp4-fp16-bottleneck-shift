#!/usr/bin/env python3
"""Run a precision sweep for TensorRT-LLM Llama inference."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


torch = _optional_import("torch")
pynvml = _optional_import("pynvml")


@dataclass
class GpuSnapshot:
    memory_used_bytes: int | None
    memory_total_bytes: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML sweep config.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only this precision/name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print experiments without loading models.",
    )
    parser.add_argument("--batch-size", type=int, action="append", default=[], help="Override batch_sizes.")
    parser.add_argument("--warmup-iters", type=int, help="Override warmup_iters.")
    parser.add_argument("--timed-iters", type=int, help="Override timed_iters.")
    parser.add_argument("--max-output-tokens", type=int, help="Override max_output_tokens.")
    parser.add_argument("--prompt-tokens", type=int, help="Override synthetic_prompt_token_count.")
    parser.add_argument("--output-path", help="Override JSONL output path.")
    return parser.parse_args()


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_trtllm_api(backend: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    if backend == "torch":
        torch_api = importlib.import_module("tensorrt_llm._torch")
        root_api = importlib.import_module("tensorrt_llm")
        llmapi = importlib.import_module("tensorrt_llm.llmapi")
        LLM = getattr(torch_api, "LLM")
        SamplingParams = getattr(root_api, "SamplingParams")
        BuildConfig = getattr(root_api, "BuildConfig", None)
    else:
        root_api = importlib.import_module("tensorrt_llm")
        llmapi = importlib.import_module("tensorrt_llm.llmapi")
        LLM = getattr(root_api, "LLM")
        SamplingParams = getattr(root_api, "SamplingParams")
        BuildConfig = getattr(root_api, "BuildConfig", None)

    QuantConfig = getattr(llmapi, "QuantConfig", None)
    QuantAlgo = getattr(llmapi, "QuantAlgo", None)
    CalibConfig = getattr(llmapi, "CalibConfig", None)
    KvCacheConfig = getattr(llmapi, "KvCacheConfig", None)
    return LLM, SamplingParams, BuildConfig, QuantConfig, QuantAlgo, CalibConfig, KvCacheConfig


def make_build_config(BuildConfig: Any, runtime_cfg: dict[str, Any]) -> Any | None:
    if BuildConfig is None:
        return None

    kwargs = {}
    if runtime_cfg.get("max_num_tokens"):
        kwargs["max_num_tokens"] = int(runtime_cfg["max_num_tokens"])
    if runtime_cfg.get("max_batch_size"):
        kwargs["max_batch_size"] = int(runtime_cfg["max_batch_size"])

    build_config = BuildConfig(**kwargs)
    plugin_config = getattr(build_config, "plugin_config", None)
    if plugin_config is not None:
        if hasattr(plugin_config, "use_paged_context_fmha"):
            plugin_config.use_paged_context_fmha = True
        if hasattr(plugin_config, "multiple_profiles"):
            plugin_config.multiple_profiles = True
    return build_config


def make_quant_config(
    QuantConfig: Any,
    QuantAlgo: Any,
    exp_cfg: dict[str, Any],
) -> Any | None:
    quant_cfg = exp_cfg.get("quantization") or {}
    mode = str(quant_cfg.get("mode", "none")).lower()
    precision = str(exp_cfg.get("precision", "")).lower()

    if mode in {"none", "prequantized"}:
        return None
    if QuantConfig is None or QuantAlgo is None:
        raise RuntimeError("TensorRT-LLM QuantConfig/QuantAlgo is unavailable in this install.")

    algo_name = {"fp8": "FP8", "nvfp4": "NVFP4"}.get(precision)
    if algo_name is None:
        raise ValueError(f"Unsupported inline quantization precision: {precision}")

    quant_algo = getattr(QuantAlgo, algo_name)
    kwargs = {"quant_algo": quant_algo}

    kv_cache = quant_cfg.get("kv_cache")
    if kv_cache:
        kv_algo_name = str(kv_cache).upper()
        if hasattr(QuantAlgo, kv_algo_name):
            kwargs["kv_cache_quant_algo"] = getattr(QuantAlgo, kv_algo_name)

    return QuantConfig(**kwargs)


def make_calib_config(CalibConfig: Any, exp_cfg: dict[str, Any]) -> Any | None:
    quant_cfg = exp_cfg.get("quantization") or {}
    if str(quant_cfg.get("mode", "none")).lower() != "inline":
        return None
    if CalibConfig is None:
        return None

    keys = [
        "calib_batches",
        "calib_batch_size",
        "calib_max_seq_length",
        "tokenizer_max_seq_length",
    ]
    kwargs = {key: int(quant_cfg[key]) for key in keys if quant_cfg.get(key) is not None}
    return CalibConfig(**kwargs)


def make_kv_cache_config(KvCacheConfig: Any, runtime_cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> Any | None:
    quant_cfg = exp_cfg.get("quantization") or {}
    dtype = quant_cfg.get("kv_cache") or runtime_cfg.get("kv_cache_dtype")
    if not dtype or KvCacheConfig is None:
        return None
    return KvCacheConfig(dtype=str(dtype))


def load_tokenizer(model: str) -> Any | None:
    transformers = _optional_import("transformers")
    if transformers is None:
        return None
    try:
        return transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return None


def make_prompt(tokenizer: Any | None, target_tokens: int) -> str:
    seed_text = (
        "You are analyzing GPU inference performance. "
        "Explain how quantization can move a workload between memory bandwidth, tensor core compute, "
        "KV cache bandwidth, and dequantization overhead. "
    )
    if tokenizer is None:
        return seed_text * max(1, math.ceil(target_tokens / 32))

    ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not ids:
        return seed_text
    repeated = (ids * math.ceil(target_tokens / len(ids)))[:target_tokens]
    return tokenizer.decode(repeated, skip_special_tokens=True)


def sync_cuda() -> None:
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_bytes() -> int | None:
    if torch is None or getattr(torch, "cuda", None) is None or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def gpu_snapshot() -> GpuSnapshot:
    if pynvml is None:
        return GpuSnapshot(None, None)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuSnapshot(int(info.used), int(info.total))
    except Exception:
        return GpuSnapshot(None, None)


def extract_texts(outputs: Any) -> list[str]:
    if not isinstance(outputs, list):
        outputs = [outputs]

    texts: list[str] = []
    for item in outputs:
        candidates = getattr(item, "outputs", None)
        if candidates:
            first = candidates[0]
            texts.append(str(getattr(first, "text", first)))
        else:
            texts.append(str(item))
    return texts


def count_tokens(tokenizer: Any | None, texts: list[str]) -> int:
    if tokenizer is None:
        return sum(max(1, len(text) // 4) for text in texts)
    return sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)


def build_llm(exp_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> tuple[Any, Any]:
    backend = str(runtime_cfg.get("backend", "tensorrt")).lower()
    if backend == "transformers":
        return build_transformers_llm(exp_cfg, runtime_cfg)

    if int(runtime_cfg.get("tensor_parallel_size", 1)) == 1:
        os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    LLM, SamplingParams, BuildConfig, QuantConfig, QuantAlgo, CalibConfig, KvCacheConfig = get_trtllm_api(backend)

    kwargs: dict[str, Any] = {
        "model": exp_cfg["model"],
        "tensor_parallel_size": int(runtime_cfg.get("tensor_parallel_size", 1)),
        "pipeline_parallel_size": int(runtime_cfg.get("pipeline_parallel_size", 1)),
    }

    if runtime_cfg.get("trust_remote_code") is not None:
        kwargs["trust_remote_code"] = bool(runtime_cfg["trust_remote_code"])
    if runtime_cfg.get("attn_backend") is not None:
        kwargs["attn_backend"] = str(runtime_cfg["attn_backend"])

    build_config = make_build_config(BuildConfig, runtime_cfg)
    quant_config = make_quant_config(QuantConfig, QuantAlgo, exp_cfg)
    calib_config = make_calib_config(CalibConfig, exp_cfg)
    kv_cache_config = make_kv_cache_config(KvCacheConfig, runtime_cfg, exp_cfg)

    if build_config is not None:
        kwargs["build_config"] = build_config
    if quant_config is not None:
        kwargs["quant_config"] = quant_config
    if calib_config is not None:
        kwargs["calib_config"] = calib_config
    if kv_cache_config is not None:
        kwargs["kv_cache_config"] = kv_cache_config

    return LLM(**kwargs), SamplingParams


class TransformersSamplingParams:
    def __init__(self, max_tokens: int, temperature: float = 0.0):
        self.max_tokens = max_tokens
        self.temperature = temperature


class TransformersLLM:
    def __init__(self, model: str, dtype: str = "auto", trust_remote_code: bool = True):
        if torch is None:
            raise RuntimeError("PyTorch is required for the transformers backend.")
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = "auto"
        if dtype in {"float16", "fp16"}:
            torch_dtype = torch.float16
        elif dtype in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def generate(self, prompts: list[str], sampling_params: TransformersSamplingParams) -> list[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        do_sample = sampling_params.temperature > 0
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=sampling_params.max_tokens,
                do_sample=do_sample,
                temperature=sampling_params.temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)


def build_transformers_llm(exp_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> tuple[Any, Any]:
    dtype = str(runtime_cfg.get("dtype", "auto"))
    trust_remote_code = bool(runtime_cfg.get("trust_remote_code", True))
    return TransformersLLM(exp_cfg["model"], dtype=dtype, trust_remote_code=trust_remote_code), TransformersSamplingParams


def run_one_batch(
    llm: Any,
    SamplingParams: Any,
    prompts: list[str],
    max_output_tokens: int,
) -> tuple[float, list[str]]:
    sampling_params = SamplingParams(max_tokens=max_output_tokens, temperature=0.0)
    sync_cuda()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    sync_cuda()
    elapsed_s = time.perf_counter() - start
    return elapsed_s, extract_texts(outputs)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def run_experiment(exp_cfg: dict[str, Any], cfg: dict[str, Any]) -> None:
    run_cfg = cfg["run"]
    runtime_cfg = cfg["runtime"]
    out_path = Path(run_cfg["output_path"])

    tokenizer = load_tokenizer(exp_cfg["model"])
    prompt = make_prompt(tokenizer, int(run_cfg["synthetic_prompt_token_count"]))
    prompt_tokens = count_tokens(tokenizer, [prompt])
    llm, SamplingParams = build_llm(exp_cfg, runtime_cfg)

    for batch_size in run_cfg["batch_sizes"]:
        prompts = [prompt] * int(batch_size)

        for _ in range(int(run_cfg["warmup_iters"])):
            run_one_batch(llm, SamplingParams, prompts, int(run_cfg["max_output_tokens"]))

        reset_peak_memory()
        before = gpu_snapshot()
        timings = []
        output_tokens = []
        sample_text = ""

        for _ in range(int(run_cfg["timed_iters"])):
            elapsed_s, texts = run_one_batch(
                llm,
                SamplingParams,
                prompts,
                int(run_cfg["max_output_tokens"]),
            )
            timings.append(elapsed_s)
            output_tokens.append(count_tokens(tokenizer, texts))
            sample_text = texts[0][:300] if texts else ""

        after = gpu_snapshot()
        total_output_tokens = sum(output_tokens)
        total_elapsed_s = sum(timings)
        output_tps = total_output_tokens / total_elapsed_s if total_elapsed_s > 0 else None
        e2e_tps = (
            (prompt_tokens * int(batch_size) * len(timings) + total_output_tokens) / total_elapsed_s
            if total_elapsed_s > 0
            else None
        )

        record = {
            "experiment": exp_cfg["name"],
            "precision": exp_cfg["precision"],
            "model": exp_cfg["model"],
            "backend": runtime_cfg.get("backend", "tensorrt"),
            "tensor_parallel_size": runtime_cfg.get("tensor_parallel_size", 1),
            "pipeline_parallel_size": runtime_cfg.get("pipeline_parallel_size", 1),
            "batch_size": int(batch_size),
            "prompt_tokens_per_request": prompt_tokens,
            "max_output_tokens": int(run_cfg["max_output_tokens"]),
            "timed_iters": len(timings),
            "latency_s_mean": sum(timings) / len(timings),
            "latency_s_min": min(timings),
            "latency_s_max": max(timings),
            "output_tokens_total": total_output_tokens,
            "output_tokens_per_s": output_tps,
            "e2e_tokens_per_s": e2e_tps,
            "torch_peak_memory_bytes": get_peak_memory_bytes(),
            "nvml_memory_used_before_bytes": before.memory_used_bytes,
            "nvml_memory_used_after_bytes": after.memory_used_bytes,
            "nvml_memory_total_bytes": after.memory_total_bytes or before.memory_total_bytes,
            "sample_output": sample_text,
            "timestamp_unix": time.time(),
        }
        append_jsonl(out_path, record)
        print(json.dumps(record, indent=2, sort_keys=True))

    del llm
    gc.collect()
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.batch_size:
        cfg["run"]["batch_sizes"] = args.batch_size
    if args.warmup_iters is not None:
        cfg["run"]["warmup_iters"] = args.warmup_iters
    if args.timed_iters is not None:
        cfg["run"]["timed_iters"] = args.timed_iters
    if args.max_output_tokens is not None:
        cfg["run"]["max_output_tokens"] = args.max_output_tokens
    if args.prompt_tokens is not None:
        cfg["run"]["synthetic_prompt_token_count"] = args.prompt_tokens
    if args.output_path is not None:
        cfg["run"]["output_path"] = args.output_path
    random.seed(int(cfg["run"].get("seed", 1234)))

    selected = {value.lower() for value in args.only}
    experiments = [exp for exp in cfg["experiments"] if exp.get("enabled", True)]
    if selected:
        experiments = [
            exp
            for exp in experiments
            if str(exp.get("precision", "")).lower() in selected or str(exp.get("name", "")).lower() in selected
        ]

    if args.dry_run:
        print(yaml.safe_dump({"runtime": cfg["runtime"], "experiments": experiments}, sort_keys=False))
        return

    for exp in experiments:
        print(f"=== Running {exp['name']} ({exp['precision']}) ===")
        run_experiment(exp, cfg)


if __name__ == "__main__":
    main()
