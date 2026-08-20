import argparse
import pandas as pd
import torch

from src.models import load_model
from src.metrics import get_memory_mb, measure_latency


def run_benchmark(
    model_name="distilgpt2",
    prompt="Artificial intelligence is transforming",
    max_new_tokens=50,
    batch_size=1,
):
    model, tokenizer, device = load_model(model_name)

    prompts = [prompt] * batch_size

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    memory_before = get_memory_mb()

    def generate():
        with torch.no_grad():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    outputs, latency = measure_latency(generate)

    memory_after = get_memory_mb()

    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

    total_generated_tokens = generated_tokens * batch_size

    throughput = total_generated_tokens / latency

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    result = {
        "model": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "latency_seconds": round(latency, 4),
        "tokens_per_second": round(throughput, 2),
        "memory_before_mb": round(memory_before, 2),
        "memory_after_mb": round(memory_after, 2),
    }

    return result, decoded


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference performance."
    )

    parser.add_argument(
        "--model",
        default="distilgpt2",
    )

    parser.add_argument(
        "--prompt",
        default="Artificial intelligence is transforming",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    result, outputs = run_benchmark(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        batch_size=args.batch_size,
    )

    print("\n=== Benchmark Results ===")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Generated Text ===")
    print(outputs[0])

    df = pd.DataFrame([result])

    output_path = "results/benchmark_results.csv"

    df.to_csv(
        output_path,
        mode="a",
        index=False,
        header=not pd.io.common.file_exists(output_path),
    )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
