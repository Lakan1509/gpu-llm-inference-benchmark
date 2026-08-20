from datetime import datetime
import argparse
import os

import pandas as pd
import torch

from src.metrics import get_memory_mb, measure_latency
from src.models import load_model


def run_benchmark(
    model_name="distilgpt2",
    prompt="Artificial intelligence is transforming",
    max_new_tokens=50,
    batch_size=1,
    iterations=1,
):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")
    model, tokenizer, device = load_model(model_name)

    prompts = [prompt] * batch_size

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    prompt_length_tokens = inputs["input_ids"].shape[1]

    def generate():
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Warm-up run to reduce one-time initialization overhead.
    generate()

    memory_before = get_memory_mb()

    outputs, latency = measure_latency(generate, device=device)

    memory_after = get_memory_mb()

    generated_tokens_per_sample = outputs.shape[1] - prompt_length_tokens
    total_generated_tokens = generated_tokens_per_sample * batch_size

    throughput = (
        total_generated_tokens / latency
        if latency > 0
        else 0.0
    )

    memory_delta = memory_after - memory_before

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "iterations": iterations,
        "prompt_length_tokens": prompt_length_tokens,
        "max_new_tokens": max_new_tokens,
        "total_generated_tokens": total_generated_tokens,
        "latency_seconds": round(latency, 4),
        "tokens_per_second": round(throughput, 2),
        "memory_before_mb": round(memory_before, 2),
        "memory_after_mb": round(memory_after, 2),
        "memory_delta_mb": round(memory_delta, 2),
    }

    return result, decoded


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
        "--iterations",
        type=int,
        default=1,
        help="Number of measured inference iterations.",
    )
    args = parser.parse_args()

    if args.tokens <= 0:
        parser.error("--tokens must be greater than 0")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")

    result, outputs = run_benchmark(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        batch_size=args.batch_size,
        iterations=args.iterations,
    )

    print("\n=== Benchmark Results ===")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Generated Text ===")
    print(outputs[0])

    os.makedirs("results", exist_ok=True)

    output_path = "results/benchmark_results.csv"

    df = pd.DataFrame([result])

    df.to_csv(
        output_path,
        mode="a",
        index=False,
        header=not os.path.exists(output_path),
    )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
