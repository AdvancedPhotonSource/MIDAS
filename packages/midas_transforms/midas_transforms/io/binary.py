"""Binary readers / writers for ``Spots.bin``, ``ExtraInfo.bin``, ``Data.bin``,
and ``nData.bin``.

All files are native-endian, packed (no struct padding), no header — same as
the C output. Layouts (from ``SaveBinData.c:170-330`` and
``SaveBinDataScanning.c:394-700``):

- ``Spots.bin`` (FF mode): ``[N, 9]`` float64. Columns:
  ``Y, Z, Omega, GrainRadius, SpotID, RingNumber, Eta, Ttheta, RadiusDistIdeal``.
- ``Spots.bin`` (PF / scanning mode, ``SaveBinDataScanning.c:394-409``):
  ``[N, 10]`` float64. Columns are the FF nine + ``ScanNr`` as col 9
  (zero-based). The C indexer keys ``ObsSpotsLab[spotRow * 10 + 9]`` to
  look up ``ypos[scanNr]``; we preserve that layout exactly.
- ``ExtraInfo.bin``: ``[N, 16]`` float64 (same for FF + PF).
- ``Data.bin`` (FF mode): ragged int32; one int per spot, in
  ``(ring, iEta, iOme)`` major order. Reconstructable from ``nData.bin``.
- ``nData.bin`` (FF mode): ``[n_ring × n_eta × n_ome, 2]`` int32 —
  ``(count, offset)``.
- ``Data.bin`` (PF mode, ``SaveBinDataScanning.c:662-700``): native-endian
  ``size_t`` (8 bytes on 64-bit Linux/macOS); stride 2 per spot, storing
  ``(rowno, scanno)`` pairs. The C code uses ``size_t`` because total
  spots can exceed ``INT_MAX`` across many scans.
- ``nData.bin`` (PF mode): ``[n_ring × n_eta × n_ome, 2]`` ``size_t`` —
  ``(count, offset)`` in the same units.
- ``voxel_scan_pos.bin`` (PF sidecar, NEW with the voxel-binner): float64
  ``(n_scans,)`` — 1-D Y per scan (in µm). The 2-D voxel grid is the
  Cartesian product of two sorted copies of this 1-D array (audit §1a;
  see ``IndexerScanningOMP.c:1667-1683``). Replaces ``positions.csv`` for
  the new Python indexer; FF mode emits no sidecar.
- ``positions.csv`` (PF sidecar, C-compat): one Y value per line (one per
  scan). Kept alongside ``voxel_scan_pos.bin`` because the legacy C
  indexer reads it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np


def write_spots_bin(path: Union[str, Path], spots: np.ndarray) -> None:
    """Write ``Spots.bin`` for FF (9 cols) or PF (10 cols) mode."""
    if spots.ndim != 2 or spots.shape[1] not in (9, 10):
        raise ValueError(
            f"Spots.bin requires 9 (FF) or 10 (PF) cols, got shape {spots.shape}"
        )
    np.ascontiguousarray(spots, dtype=np.float64).tofile(path)


def read_spots_bin(path: Union[str, Path], *, ncols: int = 9) -> np.ndarray:
    """Read ``Spots.bin``. Pass ``ncols=10`` for the PF / scanning layout."""
    if ncols not in (9, 10):
        raise ValueError(f"ncols must be 9 (FF) or 10 (PF), got {ncols}")
    return np.fromfile(path, dtype=np.float64).reshape(-1, ncols)


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
    """Write ``Data.bin`` (1-D int32) and ``nData.bin`` (Mx2 int32 — count/offset).

    FF mode only. PF (scanning) variants use 64-bit ``size_t`` instead; see
    ``write_data_ndata_bin_scanning``.
    """
    np.ascontiguousarray(data, dtype=np.int32).tofile(data_path)
    np.ascontiguousarray(ndata_pairs.reshape(-1, 2), dtype=np.int32).tofile(ndata_path)


def read_data_bin(path: Union[str, Path]) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32)


def read_ndata_bin(path: Union[str, Path]) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32).reshape(-1, 2)


# --- PF (scanning) Data.bin / nData.bin -----------------------------------
#
# The scanning C binary uses ``size_t`` (uint64) instead of int32 so that
# spot counts across many scans don't overflow. Each Data.bin entry is a
# ``(rowno, scanno)`` pair (two size_t values).


def write_data_ndata_bin_scanning(
    data_path: Union[str, Path],
    ndata_path: Union[str, Path],
    data_pairs: np.ndarray,
    ndata_pairs: np.ndarray,
) -> None:
    """Write PF-mode ``Data.bin`` (Kx2 uint64 ``(rowno, scanno)``) and
    ``nData.bin`` (Mx2 uint64 ``(count, offset)``).
    """
    np.ascontiguousarray(data_pairs.reshape(-1, 2), dtype=np.uint64).tofile(data_path)
    np.ascontiguousarray(ndata_pairs.reshape(-1, 2), dtype=np.uint64).tofile(ndata_path)


def read_data_bin_scanning(path: Union[str, Path]) -> np.ndarray:
    """Read PF-mode ``Data.bin`` as a (K, 2) uint64 ``(rowno, scanno)`` array."""
    return np.fromfile(path, dtype=np.uint64).reshape(-1, 2)


def read_ndata_bin_scanning(path: Union[str, Path]) -> np.ndarray:
    """Read PF-mode ``nData.bin`` as a (M, 2) uint64 ``(count, offset)`` array."""
    return np.fromfile(path, dtype=np.uint64).reshape(-1, 2)


# --- Voxel scan-position sidecar (PF mode only) ---------------------------


def write_voxel_scan_pos_bin(path: Union[str, Path], scan_positions: np.ndarray) -> None:
    """Write 1-D Y scan positions as float64 ``voxel_scan_pos.bin``.

    The companion of ``positions.csv``: same content, binary layout for
    fast numpy.memmap loading in the Python indexer.
    """
    arr = np.asarray(scan_positions, dtype=np.float64).ravel()
    np.ascontiguousarray(arr).tofile(path)


def read_voxel_scan_pos_bin(path: Union[str, Path]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64)
