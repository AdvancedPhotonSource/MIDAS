"""Readers for the consolidated indexer output files.

Three files share a common header layout (see
``FF_HEDM/src/IndexerConsolidatedIO.h``):

  - ``IndexBest_all.bin``       — float64[16] per solution per voxel
  - ``IndexKey_all.bin``        — uint64[4]  per solution per voxel
  - ``IndexBest_IDs_all.bin``   — int32       variable per voxel (matched spot IDs)

Header layout::

    int32  nVoxels
    int32  nSolArr[nVoxels]     # number of solutions (or total IDs) per voxel
    int64  offArr[nVoxels]      # byte offset into the file at which the
                                # voxel's data begins (measured from BOF)

Header size = ``4 + 4 * nVoxels + 8 * nVoxels``.

The reader uses :class:`numpy.memmap` so a 30k-voxel run with 100k spots
stays a few hundred MB on disk and is paged-in on demand. The returned
arrays are read-only views — copy if you need to mutate.

Bit-exact parity with C reader (``ConsolidatedReader_open`` in
``IndexerConsolidatedIO.h``): same offsets, same dtypes, same alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Sizes that match the C header. Frozen ABI — do NOT change.
CONSOLIDATED_VALS_COLS = 16
CONSOLIDATED_KEY_COLS = 4


def _header_size(n_voxels: int) -> int:
    """Bytes the header occupies. Matches ``IndexerConsolidatedIO.h``."""
    return 4 + 4 * n_voxels + 8 * n_voxels


@dataclass
class ConsolidatedReader:
    """Read-side handle for one consolidated indexer output file.

    Open with :func:`open_vals`, :func:`open_keys`, or :func:`open_ids`
    (or use the convenience :func:`open_all_three`). Access voxel data
    via :meth:`get_vals`, :meth:`get_keys`, :meth:`get_ids` — only one is
    legal per reader, matching the file kind.

    Attributes
    ----------
    path : Path
        Path to the file on disk.
    kind : str
        One of ``"vals"``, ``"keys"``, ``"ids"``.
    n_voxels : int
        Number of voxels recorded in the header.
    n_sol_arr : np.ndarray, dtype int32, shape (n_voxels,)
        For ``"vals"`` / ``"keys"``: number of candidate solutions per voxel.
        For ``"ids"``: total spot IDs stored for that voxel (sum across
        all solutions of nIDs).
    off_arr : np.ndarray, dtype int64, shape (n_voxels,)
        Byte offset from BOF where this voxel's data block begins.
    header_size : int
        Bytes consumed by the header (where data starts).
    raw : np.memmap
        Read-only file-backed array (uint8). Use the typed accessors
        instead of poking directly.
    """

    path: Path
    kind: str
    n_voxels: int
    n_sol_arr: np.ndarray
    off_arr: np.ndarray
    header_size: int
    raw: np.memmap

    def get_vals(self, vox_nr: int) -> Optional[np.ndarray]:
        """Return the (nSol[vox], 16) float64 array for ``vox_nr`` in a
        ``vals`` reader. ``None`` if the voxel has no candidates."""
        if self.kind != "vals":
            raise ValueError(f"get_vals invalid for kind={self.kind!r}")
        n_sol = int(self.n_sol_arr[vox_nr])
        if n_sol <= 0:
            return None
        off = int(self.off_arr[vox_nr])
        # raw is a memmap of uint8 covering the whole file. The 16-col
        # f64 block starts exactly at offset `off` from BOF.
        n_bytes = n_sol * CONSOLIDATED_VALS_COLS * 8
        buf = self.raw[off : off + n_bytes]
        return np.frombuffer(buf, dtype=np.float64).reshape(n_sol, CONSOLIDATED_VALS_COLS)

    def get_keys(self, vox_nr: int) -> Optional[np.ndarray]:
        """Return the (nSol[vox], 4) uint64 array for ``vox_nr`` in a
        ``keys`` reader. ``None`` if the voxel has no candidates."""
        if self.kind != "keys":
            raise ValueError(f"get_keys invalid for kind={self.kind!r}")
        n_sol = int(self.n_sol_arr[vox_nr])
        if n_sol <= 0:
            return None
        off = int(self.off_arr[vox_nr])
        n_bytes = n_sol * CONSOLIDATED_KEY_COLS * 8
        buf = self.raw[off : off + n_bytes]
        return np.frombuffer(buf, dtype=np.uint64).reshape(n_sol, CONSOLIDATED_KEY_COLS)

    def get_ids(self, vox_nr: int) -> Optional[np.ndarray]:
        """Return the (nIDs[vox],) int32 array for ``vox_nr`` in an
        ``ids`` reader. ``None`` if the voxel has no IDs."""
        if self.kind != "ids":
            raise ValueError(f"get_ids invalid for kind={self.kind!r}")
        n_ids = int(self.n_sol_arr[vox_nr])  # for ids file, n_sol_arr is per-voxel total IDs
        if n_ids <= 0:
            return None
        off = int(self.off_arr[vox_nr])
        n_bytes = n_ids * 4
        buf = self.raw[off : off + n_bytes]
        return np.frombuffer(buf, dtype=np.int32)


def _open_generic(path: Path, kind: str) -> ConsolidatedReader:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    # Read header eagerly (small — 4 + 12 * nVoxels bytes).
    with open(p, "rb") as f:
        n_voxels = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n_sol_arr = np.frombuffer(f.read(4 * n_voxels), dtype=np.int32).copy()
        off_arr = np.frombuffer(f.read(8 * n_voxels), dtype=np.int64).copy()
    header_size = _header_size(n_voxels)
    # File-backed memmap of the whole file as uint8 so we can slice by byte.
    raw = np.memmap(p, dtype=np.uint8, mode="r")
    return ConsolidatedReader(
        path=p, kind=kind, n_voxels=n_voxels,
        n_sol_arr=n_sol_arr, off_arr=off_arr,
        header_size=header_size, raw=raw,
    )


def open_vals(path: str | Path) -> ConsolidatedReader:
    """Open ``IndexBest_all.bin`` for reading."""
    return _open_generic(Path(path), "vals")


def open_keys(path: str | Path) -> ConsolidatedReader:
    """Open ``IndexKey_all.bin`` for reading."""
    return _open_generic(Path(path), "keys")


def open_ids(path: str | Path) -> ConsolidatedReader:
    """Open ``IndexBest_IDs_all.bin`` for reading."""
    return _open_generic(Path(path), "ids")


def open_all_three(output_dir: str | Path) -> tuple[ConsolidatedReader, ConsolidatedReader, ConsolidatedReader]:
    """Open the trio of consolidated indexer files from ``output_dir``.

    Returns ``(vals_reader, keys_reader, ids_reader)``.
    """
    d = Path(output_dir)
    return (
        open_vals(d / "IndexBest_all.bin"),
        open_keys(d / "IndexKey_all.bin"),
        open_ids(d / "IndexBest_IDs_all.bin"),
    )


# ---------------------------------------------------------------------------
# Writers — used by tests to synthesize fixtures matching the C byte layout.
# These are NOT used in the production path; consolidation files are written
# by the C indexer or its Python successor in midas-index.
# ---------------------------------------------------------------------------


def write_vals_bin(path: str | Path, vals_per_voxel: list[np.ndarray]) -> None:
    """Write an ``IndexBest_all.bin``-formatted file.

    Parameters
    ----------
    path : path-like
    vals_per_voxel : list of (n_sol_v, 16) float64 arrays, one per voxel
        Empty list / empty rows for voxels with no solutions are fine.
    """
    n_voxels = len(vals_per_voxel)
    n_sol_arr = np.array([int(v.shape[0]) if v.size else 0 for v in vals_per_voxel], dtype=np.int32)
    header = _header_size(n_voxels)
    off_arr = np.zeros(n_voxels, dtype=np.int64)
    cur = header
    for v in range(n_voxels):
        off_arr[v] = cur
        cur += int(n_sol_arr[v]) * CONSOLIDATED_VALS_COLS * 8
    with open(path, "wb") as f:
        f.write(np.int32(n_voxels).tobytes())
        f.write(n_sol_arr.tobytes())
        f.write(off_arr.tobytes())
        for v in range(n_voxels):
            if n_sol_arr[v] > 0:
                a = np.ascontiguousarray(vals_per_voxel[v], dtype=np.float64)
                f.write(a.tobytes())


def write_keys_bin(path: str | Path, keys_per_voxel: list[np.ndarray]) -> None:
    """Write an ``IndexKey_all.bin``-formatted file.

    Each per-voxel array is (n_sol_v, 4) uint64.
    """
    n_voxels = len(keys_per_voxel)
    n_sol_arr = np.array([int(k.shape[0]) if k.size else 0 for k in keys_per_voxel], dtype=np.int32)
    header = _header_size(n_voxels)
    off_arr = np.zeros(n_voxels, dtype=np.int64)
    cur = header
    for v in range(n_voxels):
        off_arr[v] = cur
        cur += int(n_sol_arr[v]) * CONSOLIDATED_KEY_COLS * 8
    with open(path, "wb") as f:
        f.write(np.int32(n_voxels).tobytes())
        f.write(n_sol_arr.tobytes())
        f.write(off_arr.tobytes())
        for v in range(n_voxels):
            if n_sol_arr[v] > 0:
                a = np.ascontiguousarray(keys_per_voxel[v], dtype=np.uint64)
                f.write(a.tobytes())


def write_ids_bin(path: str | Path, ids_per_voxel: list[np.ndarray]) -> None:
    """Write an ``IndexBest_IDs_all.bin``-formatted file.

    Each per-voxel array is (n_ids_v,) int32.
    """
    n_voxels = len(ids_per_voxel)
    n_ids_arr = np.array([int(i.size) for i in ids_per_voxel], dtype=np.int32)
    header = _header_size(n_voxels)
    off_arr = np.zeros(n_voxels, dtype=np.int64)
    cur = header
    for v in range(n_voxels):
        off_arr[v] = cur
        cur += int(n_ids_arr[v]) * 4
    with open(path, "wb") as f:
        f.write(np.int32(n_voxels).tobytes())
        f.write(n_ids_arr.tobytes())
        f.write(off_arr.tobytes())
        for v in range(n_voxels):
            if n_ids_arr[v] > 0:
                a = np.ascontiguousarray(ids_per_voxel[v], dtype=np.int32)
                f.write(a.tobytes())
