"""Device selection helpers for optional tensor backends."""

from __future__ import annotations


def best_torch_device() -> tuple[str, str]:
    """Return the best available PyTorch device and a human-readable reason."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu", "PyTorch is not installed"

    if torch.cuda.is_available():
        return "cuda", "CUDA is available"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Metal/MPS is available"
    return "cpu", "no PyTorch GPU backend is available"

