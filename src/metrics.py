import os
import time

import numpy as np
import psutil
import torch


P95_MIN_SAMPLES = 5


def get_memory_mb():
    """Return the current Python process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_device_memory_mb(device):
    """Return current device memory usage in MB when supported."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 * 1024)

    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024 * 1024)

    return 0.0


def reset_peak_device_memory(device):
    """Reset CUDA peak-memory tracking. No-op on other devices."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def get_peak_device_memory_mb(device):
    """Return peak device memory in MB.

    CUDA reports max_memory_allocated. MPS has no peak API, so current
    allocated memory is used as a best-effort stand-in. CPU returns 0.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return get_device_memory_mb(device)


def synchronize_device(device):
    """Synchronize supported accelerator devices before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elif device.type == "mps":
        torch.mps.synchronize()


def run_warmup(func, device=None):
    """Run an untimed warmup with device synchronization on both sides."""
    if device is not None:
        synchronize_device(device)

    result = func()

    if device is not None:
        synchronize_device(device)

    return result


def measure_latency(func, device=None):
    """Measure execution latency with optional accelerator synchronization."""
    if device is not None:
        synchronize_device(device)

    start = time.perf_counter()

    result = func()

    if device is not None:
        synchronize_device(device)

    end = time.perf_counter()

    return result, end - start


def percentile(values, q):
    """Return the q-th percentile using linear interpolation."""
    if not values:
        raise ValueError("values must be non-empty")

    return float(np.percentile(values, q))


def summarize_latencies(latencies, p95_min_samples=P95_MIN_SAMPLES):
    """Return mean, p50, and p95 latency.

    p95 is None when there are fewer than p95_min_samples measurements.
    """
    if not latencies:
        raise ValueError("latencies must be non-empty")

    mean_latency = sum(latencies) / len(latencies)
    p50_latency = percentile(latencies, 50)
    p95_latency = (
        percentile(latencies, 95)
        if len(latencies) >= p95_min_samples
        else None
    )

    return mean_latency, p50_latency, p95_latency


def tokens_per_second(token_counts, latencies):
    """Return aggregate tokens/sec: sum(tokens) / sum(latency)."""
    if len(token_counts) != len(latencies):
        raise ValueError("token_counts and latencies must have the same length")

    total_time = sum(latencies)

    if total_time <= 0:
        return 0.0

    return sum(token_counts) / total_time


def generated_token_count(output_ids, prompt_length, batch_size):
    """Count new tokens produced by one generate() call."""
    new_tokens_per_sample = output_ids.shape[1] - prompt_length

    if new_tokens_per_sample < 0:
        raise ValueError("output sequence is shorter than the prompt")

    if output_ids.shape[0] != batch_size:
        raise ValueError("batch_size does not match generate() output")

    return int(new_tokens_per_sample * batch_size)
