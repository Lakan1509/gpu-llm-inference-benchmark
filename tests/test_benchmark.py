import sys

import pandas as pd
import pytest
import torch

from src.benchmark import (
    append_result,
    clear_device_cache,
    is_out_of_memory_error,
    main,
    normalize_batch_sizes,
    run_batch_sweep,
    run_benchmark,
)


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, prompts, return_tensors="pt", padding=True):
        batch_size = len(prompts)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }

    def batch_decode(self, outputs, skip_special_tokens=True):
        return ["decoded"] * outputs.shape[0]


class FakeModel:
    def __init__(self, oom_at_batch_size=None):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.generate_calls = 0
        self.oom_at_batch_size = oom_at_batch_size

    def parameters(self):
        yield self._param

    def generate(self, **kwargs):
        self.generate_calls += 1
        batch_size = kwargs["input_ids"].shape[0]
        if (
            self.oom_at_batch_size is not None
            and batch_size >= self.oom_at_batch_size
        ):
            raise RuntimeError("CUDA out of memory")

        prompt_length = kwargs["input_ids"].shape[1]
        new_tokens = kwargs["max_new_tokens"]
        return torch.zeros(
            (batch_size, prompt_length + new_tokens),
            dtype=torch.long,
        )


def test_normalize_batch_sizes_deduplicates_and_preserves_order():
    assert normalize_batch_sizes([1, 4, 1, 8]) == [1, 4, 8]


def test_normalize_batch_sizes_rejects_non_positive():
    with pytest.raises(ValueError, match="greater than 0"):
        normalize_batch_sizes([1, 0])


def test_normalize_batch_sizes_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        normalize_batch_sizes([])


def test_is_out_of_memory_error_detects_message():
    assert is_out_of_memory_error(RuntimeError("CUDA out of memory"))
    assert not is_out_of_memory_error(RuntimeError("invalid argument"))


def test_clear_device_cache_cpu_is_noop():
    clear_device_cache(torch.device("cpu"))


def test_run_benchmark_with_preloaded_model_does_not_reload():
    model = FakeModel()
    tokenizer = FakeTokenizer()

    result, decoded = run_benchmark(
        model_name="fake-model",
        prompt="hello",
        max_new_tokens=3,
        batch_size=2,
        iterations=2,
        seed=0,
        device=torch.device("cpu"),
        model=model,
        tokenizer=tokenizer,
    )

    # One warmup + two measured iterations.
    assert model.generate_calls == 3
    assert decoded == ["decoded", "decoded"]
    assert result["status"] == "ok"
    assert result["batch_size"] == 2
    assert result["total_generated_tokens"] == 6
    assert result["p95_latency_seconds"] is None
    assert result["tokens_per_second"] > 0
    assert result["error"] is None


def test_run_benchmark_rejects_invalid_iterations():
    with pytest.raises(ValueError, match="iterations"):
        run_benchmark(
            iterations=0,
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )


def test_run_batch_sweep_records_oom_and_continues(tmp_path):
    model = FakeModel(oom_at_batch_size=8)
    tokenizer = FakeTokenizer()
    output_path = tmp_path / "results.csv"

    results = run_batch_sweep(
        model_name="fake-model",
        prompt="hello",
        max_new_tokens=3,
        batch_sizes=[1, 8, 2],
        iterations=1,
        seed=0,
        device=torch.device("cpu"),
        dtype="float32",
        output_path=str(output_path),
        model=model,
        tokenizer=tokenizer,
    )

    assert [row["status"] for row in results] == ["ok", "oom", "ok"]
    assert results[1]["tokens_per_second"] is None
    assert results[1]["batch_size"] == 8
    assert results[2]["batch_size"] == 2

    saved = pd.read_csv(output_path)
    assert list(saved["status"]) == ["ok", "oom", "ok"]


def test_run_batch_sweep_reraises_non_oom_errors(tmp_path):
    class BrokenModel(FakeModel):
        def generate(self, **kwargs):
            raise RuntimeError("invalid argument")

    with pytest.raises(RuntimeError, match="invalid argument"):
        run_batch_sweep(
            model_name="fake-model",
            prompt="hello",
            max_new_tokens=3,
            batch_sizes=[1],
            iterations=1,
            seed=0,
            device=torch.device("cpu"),
            dtype="float32",
            output_path=str(tmp_path / "results.csv"),
            model=BrokenModel(),
            tokenizer=FakeTokenizer(),
        )


def test_main_rejects_non_positive_tokens(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--tokens", "0"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


def test_append_result_rejects_schema_mismatch(tmp_path):
    path = tmp_path / "results.csv"
    append_result(str(path), {"batch_size": 1, "status": "ok"})

    with pytest.raises(ValueError, match="CSV schema mismatch"):
        append_result(str(path), {"batch_size": 1, "status": "ok", "seed": 42})
