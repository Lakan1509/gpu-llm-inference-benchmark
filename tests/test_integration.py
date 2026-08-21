import os

import pytest

from src.benchmark import run_benchmark

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to download and run DistilGPT2 on CPU.",
)
def test_distilgpt2_cpu_smoke():
    result, decoded = run_benchmark(
        model_name="distilgpt2",
        prompt="Hello",
        max_new_tokens=4,
        batch_size=1,
        iterations=2,
        seed=42,
        device="cpu",
        dtype="float32",
    )

    assert result["status"] == "ok"
    assert result["device"] == "cpu"
    assert result["total_generated_tokens"] > 0
    assert result["p95_latency_seconds"] is None
    assert len(decoded) == 1
    assert decoded[0]
