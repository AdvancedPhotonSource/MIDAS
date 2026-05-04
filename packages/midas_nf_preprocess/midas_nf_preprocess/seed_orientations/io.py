"""CSV writers for seed orientations.

Two formats:

  - **Plain quaternion CSV**: one comma-separated ``(w, x, y, z)`` per line.
    This is the format consumed by
    :func:`midas_nf_preprocess.diffr_spots.read_seed_orientations`.

  - **With-lattice CSV** (the GenSeedOrientationsFF2NFHEDM output): 11 columns,
    ``w, x, y, z, a, b, c, alpha, beta, gamma, GrainID``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch

from .from_grains import GrainOrientation


def write_seeds_csv(
    quats: torch.Tensor,
    path: Union[str, Path],
    *,
    fmt: str = "%.7f",
) -> None:
    """Write a ``(N, 4)`` quaternion tensor to a comma-separated CSV."""
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) quaternions, got shape {tuple(quats.shape)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = quats.detach().cpu().numpy().astype(np.float64)
    np.savetxt(path, arr, delimiter=",", fmt=fmt)


def read_seeds_csv(path: Union[str, Path]) -> torch.Tensor:
    """Inverse of :func:`write_seeds_csv`."""
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 4:
        raise ValueError(
            f"{path}: expected 4 columns, got {arr.shape[1]}"
        )
    return torch.from_numpy(arr)


def write_seeds_with_lattice_csv(
    grains: Iterable[GrainOrientation],
    path: Union[str, Path],
) -> int:
    """Write the FF-derived 11-column format (matches GenSeedOrientationsFF2NFHEDM).

    Each output row: ``w, x, y, z, a, b, c, alpha, beta, gamma, GrainID``.

    Returns the number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for g in grains:
            q = g.quat.detach().cpu().numpy()
            row = (
                f"{q[0]:f},{q[1]:f},{q[2]:f},{q[3]:f},"
                f"{g.lattice[0]:f},{g.lattice[1]:f},{g.lattice[2]:f},"
                f"{g.lattice[3]:f},{g.lattice[4]:f},{g.lattice[5]:f},"
                f"{g.grain_id}\n"
            )
            f.write(row)
            n += 1
    return n
