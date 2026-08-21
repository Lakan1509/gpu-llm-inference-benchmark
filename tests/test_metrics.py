import pytest
import torch

from src.metrics import (
    generated_token_count,
    get_device_memory_mb,
    get_memory_mb,
    get_peak_device_memory_mb,
    measure_latency,
    percentile,
    reset_peak_device_memory,
    run_warmup,
    summarize_latencies,
    synchronize_device,
    tokens_per_second,
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
    device = torch.device("cpu")

    memory = get_device_memory_mb(device)

    assert memory == 0.0


def test_peak_device_memory_cpu():
    device = torch.device("cpu")

    reset_peak_device_memory(device)

    assert get_peak_device_memory_mb(device) == 0.0


def test_synchronize_cpu():
    device = torch.device("cpu")

    # CPU synchronization should be a safe no-op.
    synchronize_device(device)


def test_latency_with_device():
    device = torch.device("cpu")

    def sample_function():
        return "benchmark"

    result, latency = measure_latency(
        sample_function,
        device=device,
    )

    assert result == "benchmark"
    assert latency >= 0


def test_warmup_returns_function_result():
    device = torch.device("cpu")

    result = run_warmup(lambda: "warm", device=device)

    assert result == "warm"


def test_percentile_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == 2.5


def test_percentile_rejects_empty_values():
    try:
        percentile([], 50)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_summarize_latencies_omits_p95_for_small_samples():
    latencies = [0.10, 0.20, 0.30]

    mean_latency, p50_latency, p95_latency = summarize_latencies(latencies)

    assert mean_latency == pytest.approx(0.20)
    assert p50_latency == pytest.approx(0.20)
    assert p95_latency is None


def test_summarize_latencies_computes_p95_with_enough_samples():
    latencies = [float(i) for i in range(1, 6)]

    mean_latency, p50_latency, p95_latency = summarize_latencies(latencies)

    assert mean_latency == 3.0
    assert p50_latency == 3.0
    assert p95_latency == percentile(latencies, 95)


def test_tokens_per_second_uses_total_tokens_over_total_time():
    token_counts = [10, 30]
    latencies = [1.0, 1.0]

    assert tokens_per_second(token_counts, latencies) == 20.0


def test_tokens_per_second_zero_when_time_is_zero():
    assert tokens_per_second([10], [0.0]) == 0.0


def test_tokens_per_second_rejects_mismatched_lengths():
    try:
        tokens_per_second([10, 20], [1.0])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generated_token_count():
    output_ids = torch.zeros((2, 8), dtype=torch.long)

    assert generated_token_count(output_ids, prompt_length=3, batch_size=2) == 10


def test_generated_token_count_rejects_shorter_output():
    output_ids = torch.zeros((1, 2), dtype=torch.long)

    with pytest.raises(ValueError, match="shorter than the prompt"):
        generated_token_count(output_ids, prompt_length=4, batch_size=1)


def test_generated_token_count_rejects_batch_mismatch():
    output_ids = torch.zeros((2, 8), dtype=torch.long)

    with pytest.raises(ValueError, match="does not match"):
        generated_token_count(output_ids, prompt_length=3, batch_size=1)
