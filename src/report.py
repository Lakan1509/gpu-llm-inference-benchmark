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
        "iterations",
        "prompt_length_tokens",
        "max_new_tokens",
        "total_generated_tokens",
        "latency_seconds",
        "p50_latency_seconds",
        "p95_latency_seconds",
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
            f"(mean: {row['latency_seconds']:.4f}s, "
            f"p50: {row['p50_latency_seconds']:.4f}s, "
            f"p95: {row['p95_latency_seconds']:.4f}s)"
        )

    best_throughput_row = report.loc[
        report["tokens_per_second"].idxmax()
    ]

    best_latency_row = report.loc[
        report["latency_seconds"].idxmin()
    ]

    batch_1 = report[report["batch_size"] == 1]

    print("\n=== Performance Summary ===\n")

    print(
        f"Best throughput: "
        f"Batch {int(best_throughput_row['batch_size'])} "
        f"({best_throughput_row['tokens_per_second']:.2f} tokens/sec)"
    )

    print(
        f"Best latency: "
        f"Batch {int(best_latency_row['batch_size'])} "
        f"({best_latency_row['latency_seconds']:.4f}s mean latency)"
    )

    if not batch_1.empty:
        baseline_throughput = batch_1["tokens_per_second"].max()
        best_throughput = best_throughput_row["tokens_per_second"]

        if baseline_throughput > 0:
            improvement = best_throughput / baseline_throughput

            print(
                f"Throughput improvement: "
                f"Batch 1 → Batch "
                f"{int(best_throughput_row['batch_size'])}: "
                f"{improvement:.2f}x"
            )

    print(
        f"Recommendation: Batch size "
        f"{int(best_throughput_row['batch_size'])} "
        f"provides the highest measured throughput among "
        f"the tested configurations."
    )


if __name__ == "__main__":
    generate_report()
