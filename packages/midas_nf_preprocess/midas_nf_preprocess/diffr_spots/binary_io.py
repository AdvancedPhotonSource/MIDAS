"""Binary I/O for the three MakeDiffrSpots output files.

File formats (from MakeDiffrSpots.c L595-L626):

  - ``Key.bin``              : ``int32 [N_orient * 2]`` little-endian, with
                               entries ``(count_i, offset_i)`` for each orientation.
  - ``DiffractionSpots.bin`` : ``float64 [TotalSpots * 3]`` little-endian, with
                               row layout ``(yl, zl, omega)``.
  - ``OrientMat.bin``        : ``float64 [N_orient * 9]`` little-endian, with
                               row-major flattened 3x3 matrices.

Total spots = sum of per-orientation counts; ``offset_i`` is the cumulative
prefix sum (excluding ``count_i`` itself) so that orientation i's spots span
``DiffractionSpots[offset_i : offset_i + count_i]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------


def write_key_bin(
    counts: torch.Tensor, path: Union[str, Path]
) -> None:
    """Write ``Key.bin`` from a ``(N,)`` int tensor of per-orientation spot counts."""
    if counts.ndim != 1:
        raise ValueError(f"Expected 1D counts tensor, got shape {tuple(counts.shape)}")
    counts_np = counts.detach().cpu().numpy().astype(np.int32)
    offsets = np.zeros_like(counts_np)
    if counts_np.size > 0:
        offsets[1:] = np.cumsum(counts_np[:-1])
    key = np.stack([counts_np, offsets], axis=1)  # (N, 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key.tofile(path)


def write_diffr_spots_bin(
    spots: torch.Tensor, path: Union[str, Path]
) -> None:
    """Write ``DiffractionSpots.bin`` from a ``(TotalSpots, 3)`` float tensor."""
    if spots.ndim != 2 or spots.shape[1] != 3:
        raise ValueError(
            f"Expected (TotalSpots, 3) tensor, got shape {tuple(spots.shape)}"
        )
    arr = spots.detach().cpu().numpy().astype(np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def write_orient_mat_bin(
    orient_mats: torch.Tensor, path: Union[str, Path]
) -> None:
    """Write ``OrientMat.bin`` from a ``(N, 3, 3)`` float tensor."""
    if orient_mats.ndim != 3 or orient_mats.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected (N, 3, 3) tensor, got shape {tuple(orient_mats.shape)}"
        )
    arr = orient_mats.detach().cpu().numpy().astype(np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def write_all(
    output_dir: Union[str, Path],
    counts: torch.Tensor,
    spots: torch.Tensor,
    orient_mats: torch.Tensor,
) -> dict[str, Path]:
    """Convenience wrapper that writes all three files into ``output_dir``."""
    output_dir = Path(output_dir)
    paths = {
        "Key.bin": output_dir / "Key.bin",
        "DiffractionSpots.bin": output_dir / "DiffractionSpots.bin",
        "OrientMat.bin": output_dir / "OrientMat.bin",
    }
    write_key_bin(counts, paths["Key.bin"])
    write_diffr_spots_bin(spots, paths["DiffractionSpots.bin"])
    write_orient_mat_bin(orient_mats, paths["OrientMat.bin"])
    return paths


# -----------------------------------------------------------------------------
# Readers (for parity tests / round-trip)
# -----------------------------------------------------------------------------


def read_key_bin(path: Union[str, Path], n_orient: int) -> torch.Tensor:
    """Read ``Key.bin`` and return a ``(N, 2)`` int64 tensor ``(count_i, offset_i)``."""
    arr = np.fromfile(path, dtype=np.int32, count=n_orient * 2).reshape(n_orient, 2)
    return torch.from_numpy(arr.astype(np.int64))


def read_diffr_spots_bin(
    path: Union[str, Path], n_spots: int
) -> torch.Tensor:
    """Read ``DiffractionSpots.bin`` and return a ``(N_spots, 3)`` float64 tensor."""
    arr = np.fromfile(path, dtype=np.float64, count=n_spots * 3).reshape(n_spots, 3)
    return torch.from_numpy(arr)


def read_orient_mat_bin(
    path: Union[str, Path], n_orient: int
) -> torch.Tensor:
    """Read ``OrientMat.bin`` and return a ``(N, 3, 3)`` float64 tensor."""
    arr = np.fromfile(path, dtype=np.float64, count=n_orient * 9).reshape(n_orient, 3, 3)
    return torch.from_numpy(arr)
