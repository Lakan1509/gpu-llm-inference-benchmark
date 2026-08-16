import os
import time
import psutil


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_latency(func):
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()

    latency = end - start
    return result, latency
