from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results/benchmark_results.csv")
OUTPUT_DIR = Path("results/charts")


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    successful = df[df["status"] == "ok"].copy()

    if successful.empty:
        raise ValueError("No successful benchmark results found.")

    return successful.sort_values("batch_size")


def plot_throughput(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["batch_size"],
        df["tokens_per_second"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("DistilGPT2 Inference Throughput — Apple Silicon MPS")
    plt.xticks(df["batch_size"])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "throughput_vs_batch_size.png", dpi=180)
    plt.close()


def plot_latency(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["batch_size"],
        df["latency_seconds"],
        marker="o",
        linewidth=2,
        label="Mean",
    )
    plt.plot(
        df["batch_size"],
        df["p50_latency_seconds"],
        marker="o",
        label="p50",
    )
    plt.plot(
        df["batch_size"],
        df["p95_latency_seconds"],
        marker="o",
        label="p95",
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (seconds)")
    plt.title("DistilGPT2 Generation Latency — Apple Silicon MPS")
    plt.xticks(df["batch_size"])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latency_vs_batch_size.png", dpi=180)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()

    plot_throughput(df)
    plot_latency(df)

    print("Created:")
    print(OUTPUT_DIR / "throughput_vs_batch_size.png")
    print(OUTPUT_DIR / "latency_vs_batch_size.png")


if __name__ == "__main__":
    main()
