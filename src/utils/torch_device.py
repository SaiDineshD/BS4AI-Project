"""Pick the best available PyTorch device (CUDA > MPS > CPU)."""

from __future__ import annotations

import os

import torch


def get_torch_device(prefer: str | None = None) -> torch.device:
    """Return torch.device for training/inference.

    Set ``prefer="cpu"`` or env ``TORCH_DEVICE=cpu`` to force CPU (e.g. MPS op bugs).
    """
    forced = prefer or os.environ.get("TORCH_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
