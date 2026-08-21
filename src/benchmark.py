from datetime import datetime
import argparse
import os

import pandas as pd
import torch
import transformers
from transformers import set_seed

from src.metrics import (
    generated_token_count,
    get_device_memory_mb,
    get_memory_mb,
    get_peak_device_memory_mb,
    measure_latency,
    reset_peak_device_memory,
    run_warmup,
    summarize_latencies,
    tokens_per_second,
)
from src.models import SUPPORTED_DEVICES, SUPPORTED_DTYPES, load_model


def _round_or_none(value, ndigits):
    if value is None:
        return None

    return round(value, ndigits)


def _dtype_name(model):
    return str(next(model.parameters()).dtype).replace("torch.", "")


def normalize_batch_sizes(batch_sizes):
    """Validate batch sizes and drop duplicates while preserving order."""
    if not batch_sizes:
        raise ValueError("batch_sizes must be non-empty")

    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("All batch sizes must be greater than 0")

    unique = []
    for batch_size in batch_sizes:
        if batch_size not in unique:
            unique.append(batch_size)

    return unique


def is_out_of_memory_error(exc):
    """Return True for CUDA/MPS allocator failures."""
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True

    message = str(exc).lower()
    return "out of memory" in message


def clear_device_cache(device):
    """Release cached allocator memory after an OOM so later sizes can run."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def failure_result(
    *,
    model_name,
    device,
    dtype,
    seed,
    batch_size,
    iterations,
    max_new_tokens,
    status,
    error,
):
    """CSV row for a batch size that could not complete measurement."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "device": str(device),
        "dtype": dtype,
        "seed": seed,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "batch_size": batch_size,
        "iterations": iterations,
        "prompt_length_tokens": None,
        "max_new_tokens": max_new_tokens,
        "total_generated_tokens": None,
        "stopped_early": None,
        "latency_seconds": None,
        "p50_latency_seconds": None,
        "p95_latency_seconds": None,
        "tokens_per_second": None,
        "memory_before_mb": None,
        "memory_after_mb": None,
        "memory_delta_mb": None,
        "device_memory_before_mb": None,
        "device_peak_memory_mb": None,
        "status": status,
        "error": error,
    }


def append_result(output_path, result):
    df = pd.DataFrame([result])

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        existing_columns = pd.read_csv(output_path, nrows=0).columns.tolist()
        new_columns = df.columns.tolist()

        if existing_columns != new_columns:
            raise ValueError(
                f"CSV schema mismatch in {output_path}. "
                "Move or delete the existing file before writing results "
                "with a different column set."
            )

        df.to_csv(output_path, mode="a", index=False, header=False)
        return

    df.to_csv(output_path, mode="w", index=False, header=True)


def run_benchmark(
    model_name="distilgpt2",
    prompt="Artificial intelligence is transforming",
    max_new_tokens=50,
    batch_size=1,
    iterations=1,
    seed=42,
    device="auto",
    dtype="auto",
    model=None,
    tokenizer=None,
):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    set_seed(seed)

    if model is None or tokenizer is None:
        model, tokenizer, resolved_device = load_model(
            model_name,
            device=device,
            dtype=dtype,
        )
    elif isinstance(device, torch.device):
        resolved_device = device
    else:
        resolved_device = next(model.parameters()).device

    resolved_dtype = _dtype_name(model)

    prompts = [prompt] * batch_size

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    inputs = {key: value.to(resolved_device) for key, value in inputs.items()}

    prompt_length_tokens = inputs["input_ids"].shape[1]

    def generate():
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    run_warmup(generate, device=resolved_device)

    memory_before = get_memory_mb()
    device_memory_before = get_device_memory_mb(resolved_device)
    reset_peak_device_memory(resolved_device)

    latencies = []
    token_counts = []
    outputs = None

    for _ in range(iterations):
        outputs, latency = measure_latency(generate, device=resolved_device)
        latencies.append(latency)
        token_counts.append(
            generated_token_count(outputs, prompt_length_tokens, batch_size)
        )

    memory_after = get_memory_mb()
    device_peak_memory = get_peak_device_memory_mb(resolved_device)

    mean_generated_tokens = sum(token_counts) / len(token_counts)
    mean_latency, p50_latency, p95_latency = summarize_latencies(latencies)
    throughput = tokens_per_second(token_counts, latencies)

    memory_delta = memory_after - memory_before
    stopped_early = any(
        count < (max_new_tokens * batch_size) for count in token_counts
    )

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "device": str(resolved_device),
        "dtype": resolved_dtype,
        "seed": seed,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "batch_size": batch_size,
        "iterations": iterations,
        "prompt_length_tokens": prompt_length_tokens,
        "max_new_tokens": max_new_tokens,
        "total_generated_tokens": round(mean_generated_tokens),
        "stopped_early": stopped_early,
        "latency_seconds": round(mean_latency, 4),
        "p50_latency_seconds": round(p50_latency, 4),
        "p95_latency_seconds": _round_or_none(p95_latency, 4),
        "tokens_per_second": round(throughput, 2),
        "memory_before_mb": round(memory_before, 2),
        "memory_after_mb": round(memory_after, 2),
        "memory_delta_mb": round(memory_delta, 2),
        "device_memory_before_mb": round(device_memory_before, 2),
        "device_peak_memory_mb": round(device_peak_memory, 2),
        "status": "ok",
        "error": None,
    }

    return result, decoded


def run_batch_sweep(
    *,
    model_name,
    prompt,
    max_new_tokens,
    batch_sizes,
    iterations,
    seed,
    device,
    dtype,
    output_path,
    model,
    tokenizer,
):
    """Benchmark each batch size with one loaded model. OOM does not abort."""
    results = []
    resolved_dtype = _dtype_name(model)

    for batch_size in batch_sizes:
        try:
            result, outputs = run_benchmark(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                iterations=iterations,
                seed=seed,
                device=device,
                dtype=dtype,
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            if not is_out_of_memory_error(exc):
                raise

            clear_device_cache(device)
            result = failure_result(
                model_name=model_name,
                device=device,
                dtype=resolved_dtype,
                seed=seed,
                batch_size=batch_size,
                iterations=iterations,
                max_new_tokens=max_new_tokens,
                status="oom",
                error=str(exc),
            )
            outputs = None

        print("\n=== Benchmark Results ===")
        for key, value in result.items():
            print(f"{key}: {value}")

        if result.get("status") == "ok" and outputs:
            print("\n=== Generated Text ===")
            print(outputs[0])
        elif result.get("status") == "oom":
            print("\n=== Generation failed: out of memory ===")

        append_result(output_path, result)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference performance."
    )

    parser.add_argument(
        "--model",
        default="distilgpt2",
        help="Hugging Face causal language model.",
    )

    parser.add_argument(
        "--prompt",
        default="Artificial intelligence is transforming",
        help="Prompt used for inference.",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts processed per inference call.",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Run benchmarks for multiple batch sizes.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of measured inference iterations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=SUPPORTED_DEVICES,
        help="Device to run on. auto prefers CUDA, then MPS, then CPU.",
    )

    parser.add_argument(
        "--dtype",
        default="auto",
        choices=SUPPORTED_DTYPES,
        help="Model dtype. auto is float32. CPU always uses float32.",
    )
    args = parser.parse_args()

    if args.tokens <= 0:
        parser.error("--tokens must be greater than 0")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")

    if args.iterations <= 0:
        parser.error("--iterations must be greater than 0")

    try:
        batch_sizes = normalize_batch_sizes(
            args.batch_sizes if args.batch_sizes else [args.batch_size]
        )
    except ValueError as exc:
        parser.error(str(exc))

    os.makedirs("results", exist_ok=True)
    output_path = "results/benchmark_results.csv"

    model, tokenizer, device = load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
    )

    run_batch_sweep(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        batch_sizes=batch_sizes,
        iterations=args.iterations,
        seed=args.seed,
        device=device,
        dtype=args.dtype,
        output_path=output_path,
        model=model,
        tokenizer=tokenizer,
    )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
