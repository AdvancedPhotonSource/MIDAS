"""Device / dtype resolution.

Same contract as ``midas-index``:
  - device default: cuda > mps > cpu (auto-detect)
  - dtype default: float64 on cpu (matches the C binaries),
                   float32 on cuda/mps (matches the GPU codepath idioms).
  - Both float32 and float64 are supported on every device.
  - Env vars MIDAS_TRANSFORMS_DEVICE / MIDAS_TRANSFORMS_DTYPE override defaults.
  - Library ``device=``/``dtype=`` kwargs override env vars.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch

_DTYPE_ALIAS = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
    "single": torch.float32,
}


def resolve_device(user: Optional[Union[str, torch.device]]) -> torch.device:
    """Pick a torch.device. Precedence: arg > env var > auto-detect."""
    if isinstance(user, torch.device):
        return user
    if user is None:
        user = os.environ.get("MIDAS_TRANSFORMS_DEVICE")
    if user is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(user)


def resolve_dtype(
    device: torch.device, user: Optional[Union[str, torch.dtype]]
) -> torch.dtype:
    """Pick a torch.dtype. Precedence: arg > env var > per-device default."""
    if isinstance(user, torch.dtype):
        return user
    if user is None:
        user = os.environ.get("MIDAS_TRANSFORMS_DTYPE")
    if user is not None:
        try:
            return _DTYPE_ALIAS[user.lower()]
        except KeyError as e:
            raise ValueError(
                f"Unsupported dtype '{user}'. Use one of {sorted(_DTYPE_ALIAS)}."
            ) from e
    return torch.float64 if device.type == "cpu" else torch.float32


def apply_cpu_threads(num_procs: int, device: torch.device) -> None:
    """Honor a legacy ``numProcs`` argv on CPU only (no-op on GPU)."""
    if device.type == "cpu" and num_procs > 0:
        torch.set_num_threads(num_procs)
