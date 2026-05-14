"""21×21 patch extraction (centered on a detector coord).

The C ``extract_patches`` (findSingleSolutionPFRefactored.c:2427–2992)
reads the spotMapping output, walks each (g, h, c) cell with a matched
spot ID, opens the underlying Zarr frame, applies the configured image
transforms (HFlip/VFlip/Transpose), and writes a 21×21 patch float32
centered on the (y, z) detector coord into a contiguous output array.

For this P4 pass we provide:

  - :func:`extract_patch_from_frame` — the pure mathematical kernel
    (clip to image bounds, copy 21×21 window, zero-pad).
  - :func:`extract_patches_from_spot_map` — top-level driver that walks
    the spotMapping array, calls a user-supplied frame loader, and
    writes the output ``patches_*.bin`` + ``spotPos_*.bin`` files.

The Zarr → frame loading is delegated to a callable so the heavy
dependencies (blosc2, zipfile) don't infect this module. The default
loader raises ``NotImplementedError`` — callers must pass an explicit
loader. ``extract_patch_from_frame`` is fully testable without any
Zarr machinery.

PATCH_SIZE == 21 (matches PATCH_HALF_SIZE = 10 in the C source).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

PATCH_HALF_SIZE = 10
PATCH_SIZE = 2 * PATCH_HALF_SIZE + 1  # 21


def extract_patch_from_frame(
    frame: np.ndarray,
    y_center: float,
    z_center: float,
) -> np.ndarray:
    """Extract a 21×21 patch centered on ``(y_center, z_center)``.

    The frame is indexed as ``frame[z_int, y_int]`` (C/MIDAS convention:
    z is the slow axis, y is the fast axis). Out-of-bound pixels are
    filled with 0 (matches the C ``calloc``).

    Parameters
    ----------
    frame : ndarray shape (nz, ny), any numeric dtype
    y_center, z_center : float — center in pixel coordinates

    Returns
    -------
    patch : ndarray (21, 21) float32
    """
    nz, ny = frame.shape
    yc = int(round(y_center))
    zc = int(round(z_center))
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for dz in range(-PATCH_HALF_SIZE, PATCH_HALF_SIZE + 1):
        zi = zc + dz
        if zi < 0 or zi >= nz:
            continue
        for dy in range(-PATCH_HALF_SIZE, PATCH_HALF_SIZE + 1):
            yi = yc + dy
            if yi < 0 or yi >= ny:
                continue
            patch[dz + PATCH_HALF_SIZE, dy + PATCH_HALF_SIZE] = float(frame[zi, yi])
    return patch


def extract_patches_from_spot_map(
    *,
    spot_id_arr: np.ndarray,           # (nG, nH, nS) int32
    spot_meta: np.ndarray,             # (nG, nH, nS, 4) float64 — eta, 2theta, yCen, zCen
    n_grains: int,
    max_n_hkls: int,
    n_scans: int,
    frame_loader: Callable[[int], Optional[np.ndarray]],
    output_dir: str | Path,
) -> tuple[str, str]:
    """Walk the spotMapping array, build the patches + spotPos arrays.

    Parameters
    ----------
    spot_id_arr : ndarray (nG, nH, nS) int32
        From :mod:`._sinogen`. ``-1`` means no spot at that cell.
    spot_meta : ndarray (nG, nH, nS, 4) float64
        Columns: eta, 2theta, yCen, zCen.
    n_grains, max_n_hkls, n_scans : int
    frame_loader : callable
        ``frame_loader(spot_id) -> ndarray (nz, ny) | None``. ``None``
        means the loader couldn't find a frame for that spot.
    output_dir : path-like

    Returns
    -------
    (patches_path, spot_pos_path) : tuple[str, str]
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nG, nH, nS = int(n_grains), int(max_n_hkls), int(n_scans)
    total = nG * nH * nS
    patch_px = PATCH_SIZE * PATCH_SIZE
    patches = np.zeros((nG, nH, nS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    spot_pos = np.full((nG, nH, nS, 2), -1.0, dtype=np.float64)

    for g in range(nG):
        for h in range(nH):
            for c in range(nS):
                sid = int(spot_id_arr[g, h, c])
                if sid < 0:
                    continue
                y_cen = float(spot_meta[g, h, c, 2])
                z_cen = float(spot_meta[g, h, c, 3])
                frame = frame_loader(sid)
                if frame is None:
                    continue
                patches[g, h, c] = extract_patch_from_frame(frame, y_cen, z_cen)
                spot_pos[g, h, c, 0] = y_cen
                spot_pos[g, h, c, 1] = z_cen

    patches_fn = out_dir / f"patches_{nG}_{nH}_{nS}.bin"
    spot_pos_fn = out_dir / f"spotPos_{nG}_{nH}_{nS}.bin"
    patches_fn.write_bytes(patches.astype(np.float32, copy=False).tobytes())
    spot_pos_fn.write_bytes(spot_pos.astype(np.float64, copy=False).tobytes())
    return str(patches_fn), str(spot_pos_fn)
