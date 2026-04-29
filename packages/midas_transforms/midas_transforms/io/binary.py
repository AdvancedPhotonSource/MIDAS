"""Binary readers / writers for ``Spots.bin``, ``ExtraInfo.bin``, ``Data.bin``,
and ``nData.bin``.

All files are native-endian, packed (no struct padding), no header — same as
the C output. Layouts (from ``SaveBinData.c:170-330``):

- ``Spots.bin``: ``[N, 9]`` float64. Columns:
  ``Y, Z, Omega, GrainRadius, SpotID, RingNumber, Eta, Ttheta, RadiusDistIdeal``.
- ``ExtraInfo.bin``: ``[N, 16]`` float64.
- ``Data.bin``: ragged int32; one int per spot, in
  ``(ring, iEta, iOme)`` major order. Reconstructable from ``nData.bin``.
- ``nData.bin``: ``[n_ring × n_eta × n_ome, 2]`` int32 — ``(count, offset)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np


def write_spots_bin(path: Union[str, Path], spots: np.ndarray) -> None:
    if spots.shape[1] != 9:
        raise ValueError(f"Spots.bin requires 9 cols, got {spots.shape[1]}")
    np.ascontiguousarray(spots, dtype=np.float64).tofile(path)


def read_spots_bin(path: Union[str, Path]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64).reshape(-1, 9)


def write_extrainfo_bin(path: Union[str, Path], extra: np.ndarray) -> None:
    if extra.shape[1] != 16:
        raise ValueError(f"ExtraInfo.bin requires 16 cols, got {extra.shape[1]}")
    np.ascontiguousarray(extra, dtype=np.float64).tofile(path)


def read_extrainfo_bin(path: Union[str, Path]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64).reshape(-1, 16)


def write_data_ndata_bin(
    data_path: Union[str, Path],
    ndata_path: Union[str, Path],
    data: np.ndarray,
    ndata_pairs: np.ndarray,
) -> None:
    """Write ``Data.bin`` (1-D int32) and ``nData.bin`` (Mx2 int32 — count/offset)."""
    np.ascontiguousarray(data, dtype=np.int32).tofile(data_path)
    np.ascontiguousarray(ndata_pairs.reshape(-1, 2), dtype=np.int32).tofile(ndata_path)


def read_data_bin(path: Union[str, Path]) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32)


def read_ndata_bin(path: Union[str, Path]) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32).reshape(-1, 2)
