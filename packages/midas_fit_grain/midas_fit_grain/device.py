"""Device / dtype resolution.

Mirrors :mod:`midas_index.device`:
  - device default: cuda > mps > cpu (auto-detect)
  - dtype default: f64 on cpu, f32 on cuda/mps
  - both float32 and float64 supported on every device
  - env vars MIDAS_FIT_GRAIN_DEVICE / MIDAS_FIT_GRAIN_DTYPE override defaults
  - explicit kwargs override env vars
"""

from __future__ import annotations

import os

import torch

_DTYPE_ALIAS = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "single": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
}


def resolve_device(user: str | torch.device | None) -> torch.device:
    """Pick a torch.device. Precedence: arg > env var > auto-detect."""
    if isinstance(user, torch.device):
        return user
    if user is None:
        user = os.environ.get("MIDAS_FIT_GRAIN_DEVICE")
    if user is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(user)


def resolve_dtype(device: torch.device, user: str | torch.dtype | None) -> torch.dtype:
    """Pick a torch.dtype. Precedence: arg > env var > per-device default."""
    if isinstance(user, torch.dtype):
        return user
    if user is None:
        user = os.environ.get("MIDAS_FIT_GRAIN_DTYPE")
    if user is not None:
        try:
            return _DTYPE_ALIAS[user.lower()]
        except KeyError as e:
            raise ValueError(
                f"Unsupported dtype '{user}'. Use one of {sorted(_DTYPE_ALIAS)}."
            ) from e
    return torch.float64 if device.type == "cpu" else torch.float32


def apply_cpu_threads(num_procs: int, device: torch.device) -> None:
    """Honor the legacy ``numProcs`` argv on CPU only."""
    if device.type == "cpu" and num_procs > 0:
        torch.set_num_threads(num_procs)
