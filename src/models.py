import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")
SUPPORTED_DTYPES = ("auto", "float32", "float16", "bfloat16")

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested="auto"):
    """Return a torch.device, failing if an explicit backend is unavailable."""
    if isinstance(requested, torch.device):
        return requested

    name = str(requested).lower()

    if name not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device '{requested}'. "
            f"Choose from: {', '.join(SUPPORTED_DEVICES)}"
        )

    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_dtype(requested="auto", device=None):
    """Resolve a torch dtype. CPU always uses float32."""
    if isinstance(requested, torch.dtype):
        dtype = requested
    else:
        name = str(requested).lower()

        if name not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{requested}'. "
                f"Choose from: {', '.join(SUPPORTED_DTYPES)}"
            )

        if name == "auto":
            # Default matches prior Hugging Face loads (fp32) so historical
            # CSV numbers stay comparable. Pass float16/bfloat16 explicitly.
            dtype = torch.float32
        else:
            dtype = _DTYPE_MAP[name]

    if device is not None and device.type == "cpu" and dtype != torch.float32:
        return torch.float32

    return dtype


def configure_tokenizer(tokenizer):
    """Configure padding for decoder-only batched generation."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name: str = "distilgpt2", device="auto", dtype="auto"):
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_name))
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolved_dtype,
    )
    model.to(device=resolved_device, dtype=resolved_dtype)
    model.eval()

    return model, tokenizer, resolved_device
