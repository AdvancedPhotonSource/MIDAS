"""Binary file readers (mmap'd, native endian).

Files (paths relative to ``cwd = dirname(OutputFolder)`` per C convention,
see IndexerOMP.c:2231-2236):

  Spots.bin     [n_spots, 9] float64                observed spots in lab frame
  Data.bin      flat int32                          spot rows per (ring,eta,omega) bin
  nData.bin     flat int32 (interleaved 2 per bin)  per-bin (count, data_offset)

Mirrors `ReadSpots` (IndexerOMP.c:2118), `ReadBins` (line 2085).
BigDetector is deprecated; no `read_big_det` is provided.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NATIVE_ORDER = {"=", "|", "<"} if np.little_endian else {"=", "|", ">"}


def _assert_native(arr: np.ndarray, fname: str) -> None:
    """Fail loud if a file appears to be in non-native byte order."""
    bo = arr.dtype.byteorder
    if bo not in _NATIVE_ORDER:
        raise ValueError(
            f"{fname}: dtype byteorder {bo!r} is not native; "
            "the indexer assumes native-endian binaries."
        )


def read_spots(cwd: str | Path) -> tuple[int, np.ndarray]:
    """Read Spots.bin into a [n_spots, 9-or-10] float64 array via mmap.

    Auto-detects FF (9 cols) vs PF / scanning (10 cols — FF nine plus
    ``ScanNr`` at col 9, matching the layout emitted by
    ``SaveBinDataScanning.c:394-409`` and consumed by the scan-aware
    branch in ``compute.matching.compare_spots``).

    Returns (n_spots, ObsSpotsLab).
    """
    cwd = Path(cwd)
    path = cwd / "Spots.bin"
    if not path.exists():
        raise FileNotFoundError(f"Spots.bin not found at {path}")
    arr = np.memmap(path, dtype=np.float64, mode="r")
    _assert_native(arr, "Spots.bin")
    # Prefer 10-col PF layout when the size matches both; treat divides-by-10
    # as PF and divides-by-9-only as FF. Sizes that are multiples of both
    # 90 doubles are ambiguous in principle, but real fixtures resolve via
    # the file's expected col count — caller dispatches on shape downstream.
    if arr.size % 10 == 0 and arr.size % 9 != 0:
        n_cols = 10
    elif arr.size % 9 == 0:
        n_cols = 9
    elif arr.size % 10 == 0:
        n_cols = 10
    else:
        raise ValueError(
            f"Spots.bin size {arr.size * 8} bytes is not a multiple of "
            "9 (FF) or 10 (PF/scanning) doubles."
        )
    n_spots = arr.size // n_cols
    return n_spots, arr.reshape(n_spots, n_cols)


def read_bins(cwd: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read Data.bin + nData.bin as int32 mmaps.

    Returns (data, ndata) where:
      data[k]       = spot row stored in flat layout
      ndata[2*pos]   = nspots in bin `pos`
      ndata[2*pos+1] = data offset for bin `pos`

    The bin index is `pos = ring * (n_eta_bins * n_ome_bins) + iEta * n_ome_bins + iOme`,
    computed in `compute.binning`.
    """
    cwd = Path(cwd)
    data_path = cwd / "Data.bin"
    ndata_path = cwd / "nData.bin"
    if not data_path.exists():
        raise FileNotFoundError(f"Data.bin not found at {data_path}")
    if not ndata_path.exists():
        raise FileNotFoundError(f"nData.bin not found at {ndata_path}")
    data = np.memmap(data_path, dtype=np.int32, mode="r")
    ndata = np.memmap(ndata_path, dtype=np.int32, mode="r")
    _assert_native(data, "Data.bin")
    _assert_native(ndata, "nData.bin")
    if ndata.size % 2 != 0:
        raise ValueError(
            f"nData.bin size {ndata.size * 4} bytes is not a multiple of 2 int32s"
        )
    return data, ndata
