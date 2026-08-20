import pandas as pd


def generate_report(input_path="results/benchmark_results.csv"):
    df = pd.read_csv(input_path)

    if df.empty:
        print("No benchmark results found.")
        return

    columns = [
        "timestamp",
        "model",
        "device",
        "batch_size",
        "prompt_length_tokens",
        "max_new_tokens",
        "total_generated_tokens",
        "latency_seconds",
        "tokens_per_second",
        "memory_delta_mb",
    ]

    available_columns = [column for column in columns if column in df.columns]
    report = df[available_columns].copy()

    report = report.sort_values(
        by=["batch_size", "tokens_per_second"],
        ascending=[True, False],
    )

    print("\n=== LLM Inference Benchmark Report ===\n")
    print(report.to_string(index=False))

    print("\n=== Best Throughput by Batch Size ===\n")

    best = (
        report.loc[
            report.groupby("batch_size")["tokens_per_second"].idxmax()
        ]
        .sort_values("batch_size")
    )

    for _, row in best.iterrows():
        print(
            f"Batch {int(row['batch_size'])}: "
            f"{row['tokens_per_second']:.2f} tokens/sec "
            f"({row['latency_seconds']:.4f}s latency)"
        )


if __name__ == "__main__":
    generate_report()
