from src.metrics import (
    get_memory_mb,
    get_device_memory_mb,
    measure_latency,
    synchronize_device,
)


def test_memory_positive():
    memory = get_memory_mb()
    assert memory > 0


def test_latency_measurement():
    def sample_function():
        return 42

    result, latency = measure_latency(sample_function)

    assert result == 42
    assert latency >= 0


def test_device_memory_cpu():
    import torch

    device = torch.device("cpu")

    memory = get_device_memory_mb(device)

    assert memory == 0.0


def test_synchronize_cpu():
    import torch

    device = torch.device("cpu")

    # CPU synchronization should be a safe no-op.
    synchronize_device(device)


def test_latency_with_device():
    import torch

    device = torch.device("cpu")

    def sample_function():
        return "benchmark"

    result, latency = measure_latency(
        sample_function,
        device=device,
    )

    assert result == "benchmark"
    assert latency >= 0
