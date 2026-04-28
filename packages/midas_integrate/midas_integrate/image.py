"""Image dtype dispatch and decoding helpers.

Mirrors the dtype scheme used by ``IntegratorFitPeaksGPUStream`` over the
socket protocol::

    code 0 → uint8     (1 byte/pixel)
    code 1 → uint16    (2)
    code 2 → uint32    (4)
    code 3 → int64     (8)
    code 4 → float32   (4)
    code 5 → float64   (8)
    code 6 → hybrid    (variable: 4-byte overflow-count + uint16 base
                        + (overflow_count × (int32 idx + int64 value)))

Plus convenience loaders for TIFF and raw-binary disk files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

DTYPE_CODE_UINT8 = 0
DTYPE_CODE_UINT16 = 1
DTYPE_CODE_UINT32 = 2
DTYPE_CODE_INT64 = 3
DTYPE_CODE_FLOAT32 = 4
DTYPE_CODE_FLOAT64 = 5
DTYPE_CODE_HYBRID = 6

NUMPY_DTYPE_FOR_CODE = {
    DTYPE_CODE_UINT8: np.uint8,
    DTYPE_CODE_UINT16: np.uint16,
    DTYPE_CODE_UINT32: np.uint32,
    DTYPE_CODE_INT64: np.int64,
    DTYPE_CODE_FLOAT32: np.float32,
    DTYPE_CODE_FLOAT64: np.float64,
}

_BYTES_PER_PIXEL = {
    DTYPE_CODE_UINT8: 1,
    DTYPE_CODE_UINT16: 2,
    DTYPE_CODE_UINT32: 4,
    DTYPE_CODE_INT64: 8,
    DTYPE_CODE_FLOAT32: 4,
    DTYPE_CODE_FLOAT64: 8,
}


def bytes_per_pixel(code: int) -> int:
    """Bytes per pixel for the given dtype code (excluding hybrid)."""
    if code == DTYPE_CODE_HYBRID:
        raise ValueError("hybrid dtype has variable size; use decode_hybrid_payload")
    if code not in _BYTES_PER_PIXEL:
        raise ValueError(f"unknown dtype code {code}")
    return _BYTES_PER_PIXEL[code]


def decode_payload(
    payload: bytes,
    *,
    dtype_code: int,
    n_pixels_y: int,
    n_pixels_z: int,
) -> np.ndarray:
    """Decode a non-hybrid socket payload into a (n_pixels_z, n_pixels_y) array."""
    if dtype_code == DTYPE_CODE_HYBRID:
        return decode_hybrid_payload(payload,
                                     n_pixels_y=n_pixels_y,
                                     n_pixels_z=n_pixels_z)
    npdt = np.dtype(NUMPY_DTYPE_FOR_CODE[dtype_code])
    expected = n_pixels_y * n_pixels_z * npdt.itemsize
    if len(payload) != expected:
        raise ValueError(
            f"payload size {len(payload)} != expected {expected} "
            f"({n_pixels_z}x{n_pixels_y} of {npdt})"
        )
    arr = np.frombuffer(payload, dtype=npdt).reshape(n_pixels_z, n_pixels_y)
    return arr


def decode_hybrid_payload(
    payload: bytes,
    *,
    n_pixels_y: int,
    n_pixels_z: int,
) -> np.ndarray:
    """Decode a dtype=6 hybrid payload (uint16 base + overflow patches).

    Layout (matches ProcessImageGPU_Hybrid_Impl):
      [0..3]            : uint32 overflow_count C
      [4..4+N*2]        : uint16 base[NrPixels]
      [..]              : int32  idx[C]
      [..]              : int64  val[C]

    Returns a (NrPixelsZ, NrPixelsY) float32 array with overflow values
    overlaid in place.
    """
    n_pixels = n_pixels_y * n_pixels_z
    if len(payload) < 4 + n_pixels * 2:
        raise ValueError("payload too small for hybrid base+count")
    overflow_count = int(np.frombuffer(payload, dtype=np.uint32, count=1)[0])
    expected = 4 + n_pixels * 2 + overflow_count * (4 + 8)
    if len(payload) != expected:
        raise ValueError(
            f"hybrid payload size {len(payload)} != expected {expected} "
            f"(N={n_pixels}, C={overflow_count})"
        )
    base_off = 4
    base = np.frombuffer(payload, dtype=np.uint16,
                         count=n_pixels, offset=base_off).copy()
    out = base.astype(np.float32).reshape(n_pixels_z, n_pixels_y)
    if overflow_count > 0:
        idx_off = base_off + n_pixels * 2
        val_off = idx_off + overflow_count * 4
        idx = np.frombuffer(payload, dtype=np.int32,
                            count=overflow_count, offset=idx_off)
        val = np.frombuffer(payload, dtype=np.int64,
                            count=overflow_count, offset=val_off)
        flat = out.reshape(-1)
        flat[idx] = val.astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Disk loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_image(
    path: str | Path,
    *,
    n_pixels_y: int,
    n_pixels_z: int,
    raw_dtype: Optional[str] = None,
) -> np.ndarray:
    """Load a 2D detector image from disk.

    - .tif / .tiff → tifffile.imread (preserves native dtype)
    - other        → raw binary; ``raw_dtype`` (e.g. 'uint16') is required
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(path))
        if arr.shape != (n_pixels_z, n_pixels_y):
            raise ValueError(
                f"TIFF shape {arr.shape} != expected ({n_pixels_z}, {n_pixels_y})"
            )
        return arr
    if raw_dtype is None:
        raise ValueError("raw_dtype required for non-TIFF inputs")
    npdt = np.dtype(raw_dtype)
    arr = np.fromfile(path, dtype=npdt, count=n_pixels_y * n_pixels_z)
    if arr.size != n_pixels_y * n_pixels_z:
        raise ValueError(
            f"raw file has {arr.size} elements, expected {n_pixels_y * n_pixels_z}"
        )
    return arr.reshape(n_pixels_z, n_pixels_y)


def average_dark_frames(
    path: str | Path,
    *,
    n_pixels_y: int,
    n_pixels_z: int,
    dtype: str = "int64",
) -> np.ndarray:
    """Read a multi-frame dark binary and return the per-pixel mean (float64).

    Mirrors the dark-frame averaging in the C ``main()`` of the GPU stream
    binary: each frame is ``n_pixels * sizeof(int64)`` bytes by default.
    """
    path = Path(path)
    npdt = np.dtype(dtype)
    bytes_per_frame = n_pixels_y * n_pixels_z * npdt.itemsize
    sz = path.stat().st_size
    n_frames = sz // bytes_per_frame
    if n_frames == 0:
        raise ValueError(f"dark file {path} too small for one frame "
                         f"({sz} bytes, frame={bytes_per_frame})")
    raw = np.fromfile(path, dtype=npdt,
                      count=n_pixels_y * n_pixels_z * n_frames)
    raw = raw.reshape(n_frames, n_pixels_z, n_pixels_y)
    return raw.astype(np.float64).mean(axis=0)
