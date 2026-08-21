from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.models import configure_tokenizer, resolve_device, resolve_dtype


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("tpu")


def test_resolve_device_cuda_unavailable():
    with patch("src.models.torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")


def test_resolve_device_mps_unavailable():
    with patch(
        "src.models.torch.backends.mps.is_available",
        return_value=False,
    ):
        with pytest.raises(RuntimeError, match="MPS was requested"):
            resolve_device("mps")


def test_resolve_device_auto_prefers_cuda():
    with (
        patch("src.models.torch.cuda.is_available", return_value=True),
        patch("src.models.torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device("auto").type == "cuda"


def test_resolve_device_auto_falls_back_to_mps():
    with (
        patch("src.models.torch.cuda.is_available", return_value=False),
        patch("src.models.torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device("auto").type == "mps"


def test_resolve_device_auto_falls_back_to_cpu():
    with (
        patch("src.models.torch.cuda.is_available", return_value=False),
        patch("src.models.torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device("auto").type == "cpu"


def test_resolve_dtype_auto_is_float32():
    assert resolve_dtype("auto") == torch.float32
    assert resolve_dtype("auto", torch.device("cpu")) == torch.float32


def test_resolve_dtype_cpu_forces_float32():
    assert resolve_dtype("float16", torch.device("cpu")) == torch.float32


def test_resolve_dtype_gpu_honors_float16():
    assert resolve_dtype("float16", torch.device("cuda")) == torch.float16


def test_resolve_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("int8")


def test_configure_tokenizer_left_padding_and_pad_token():
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")

    configure_tokenizer(tokenizer)

    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "left"
