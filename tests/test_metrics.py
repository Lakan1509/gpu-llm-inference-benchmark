from src.metrics import get_memory_mb, measure_latency


def test_memory_positive():
    memory = get_memory_mb()
    assert memory > 0


def test_latency_measurement():
    def sample_function():
        return 42

    result, latency = measure_latency(sample_function)

    assert result == 42
    assert latency >= 0