import os
import time

import psutil
import torch


def get_memory_mb():
    """Return the current Python process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_device_memory_mb(device):
    """Return device memory usage in MB when supported."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 * 1024)

    return 0.0


def synchronize_device(device):
    """Synchronize supported accelerator devices before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elif device.type == "mps":
        torch.mps.synchronize()


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
