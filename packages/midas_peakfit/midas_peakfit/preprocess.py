"""Per-frame pre-processing: decompress → convert → square-pad → transform →
transpose → dark/flood/threshold/mask.

Order matters and is preserved exactly from ``processImageFrame`` in
``PeaksFittingOMPZarrRefactor.c`` (lines 1380-1440).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


# ─── Image transformations (matching applyImageTransformations_d) ───────────
def apply_image_transformations(
    image: np.ndarray, transform_options: List[int]
) -> np.ndarray:
    """Apply ``transform_options`` in order. Operates on a square (N, N) array.

    Codes (from C source):
        0: no-op
        1: flip-horizontal (along Y / row axis) — ``image[l, m] := image[l, N-m-1]``
        2: flip-vertical   (along Z / col axis) — ``image[l, m] := image[N-l-1, m]``
        3: transpose                            — ``image[l, m] := image[m, l]``

    Operates on a copy; returns the transformed array.
    """
    if not transform_options:
        return image.copy()

    out = image.copy()
    for code in transform_options:
        if code == 1:
            out = out[:, ::-1].copy()
        elif code == 2:
            out = out[::-1, :].copy()
        elif code == 3:
            out = out.T.copy()
        # 0 / unknown → no-op
    return out


def make_square_image(
    img_asym: np.ndarray, NrPixels: int, NrPixelsY: int, NrPixelsZ: int
) -> np.ndarray:
    """Pad an asymmetric (Z, Y) image to a square (NrPixels, NrPixels) image.

    Mirrors ``makeSquareImage_d`` in PeaksFittingOMPZarrRefactor.c. The on-disk
    zarr layout is row-major ``(NrPixelsZ, NrPixelsY)`` (Z slow, Y fast); the
    output square has ``NrPixels = max(NrPixelsZ, NrPixelsY)``. Bytes outside
    ``[:NrPixelsZ, :NrPixelsY]`` are zero.

    Verified against C: in both ``Y > Z`` (single memcpy) and ``Z > Y``
    (line-by-line memcpy) cases, the resulting square's ``[:Z, :Y]`` block
    equals the input.
    """
    if img_asym.shape != (NrPixelsZ, NrPixelsY):
        raise ValueError(
            f"make_square_image expected shape ({NrPixelsZ}, {NrPixelsY}), "
            f"got {img_asym.shape}"
        )
    if NrPixelsY == NrPixelsZ == NrPixels:
        return img_asym.astype(np.float64, copy=True)
    out = np.zeros((NrPixels, NrPixels), dtype=np.float64)
    out[:NrPixelsZ, :NrPixelsY] = img_asym.astype(np.float64, copy=False)
    return out


def transpose_square(image: np.ndarray) -> np.ndarray:
    """Equivalent to ``transposeMatrix`` in C. NumPy's ``.T`` returns a view;
    we materialize a contiguous copy to match C's eager layout.
    """
    return np.ascontiguousarray(image.T)


def prepare_dark(
    raw_dark: np.ndarray,
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],
) -> np.ndarray:
    """Average + square-pad + transform + transpose.

    ``raw_dark`` is shape (nDarks, Y, Z) or (Y, Z). The C tool sums
    each transformed dark and divides by nDarks; since the operations
    (square-pad, flip, transpose, transpose) are linear, averaging
    *first* and then transforming produces identical results.

    Output shape: (NrPixels, NrPixels) float64.
    """
    if raw_dark.ndim == 3:
        avg = raw_dark.mean(axis=0)
    else:
        avg = raw_dark
    sq = make_square_image(avg.astype(np.float64), NrPixels, NrPixelsY, NrPixelsZ)
    transformed = apply_image_transformations(sq, transform_options)
    return transpose_square(transformed)


def prepare_flood(
    raw_flood: Optional[np.ndarray],
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],  # accepted but unused; matches C semantics
) -> np.ndarray:
    """Flood field. The C tool reads the on-disk flood as a *raw*
    ``double[NrPixels × NrPixels]`` block (no square-pad, no transform, no
    transpose) — see PeaksFittingOMPZarrRefactor.c:1311-1322.

    We accept either a pre-cooked (NrPixels, NrPixels) array (used as-is) or
    fall back to ones. Zero entries are replaced with 1.0 to avoid div-by-zero.
    """
    if raw_flood is None:
        return np.ones((NrPixels, NrPixels), dtype=np.float64)
    arr = raw_flood.astype(np.float64, copy=False)
    if arr.shape != (NrPixels, NrPixels):
        # Best-effort pad/crop: assume input is (Z, Y) asymmetric;
        # only [:Z, :Y] of output is populated, rest stays as 1.0.
        out = np.ones((NrPixels, NrPixels), dtype=np.float64)
        zlim = min(NrPixelsZ, arr.shape[0])
        ylim = min(NrPixelsY, arr.shape[1])
        out[:zlim, :ylim] = arr[:zlim, :ylim]
        arr = out
    return np.where(arr == 0, 1.0, arr)


def prepare_mask(
    raw_mask: Optional[np.ndarray],
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],  # accepted but unused; matches C semantics
) -> np.ndarray:
    """Mask. The C tool only square-pads the mask — no transforms, no transpose
    (PeaksFittingOMPZarrRefactor.c:1356-1364). Pixels with value > 0 are masked.

    Stored shape: (NrPixels, NrPixels) with the [:Z, :Y] block populated from
    the asymmetric input; the rest is zero.
    """
    if raw_mask is None:
        return np.zeros((NrPixels, NrPixels), dtype=np.float64)
    return make_square_image(
        raw_mask.astype(np.float64, copy=False), NrPixels, NrPixelsY, NrPixelsZ
    )


# ─── Per-frame pipeline ──────────────────────────────────────────────────────
def preprocess_frame(
    raw_frame: np.ndarray,
    *,
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],
    dark: np.ndarray,
    flood: np.ndarray,
    good_coords: np.ndarray,
    bc: float,
    bad_px_intensity: float,
    make_map: int,
) -> np.ndarray:
    """Replicate the C ``processImageFrame`` corrections (lines 1414-1440).

    Steps:
      1. ``image_d`` = square-padded float64 of ``raw_frame``
      2. (if ``make_map==1``) replace pixels equal to ``bad_px_intensity`` with 0
      3. apply ImTransOpt sequence
      4. transpose to analysis frame
      5. mask via goodCoords; subtract dark, divide by flood, multiply by bc;
         re-threshold against goodCoords

    Returns: ``imgCorrBC`` shape (NrPixels, NrPixels) float64.
    """
    image_d = make_square_image(
        raw_frame.astype(np.float64, copy=False), NrPixels, NrPixelsY, NrPixelsZ
    )

    if make_map == 1 and bad_px_intensity != 0.0:
        image_d = np.where(image_d == bad_px_intensity, 0.0, image_d)

    image_d = apply_image_transformations(image_d, transform_options)
    img = transpose_square(image_d)

    out = np.zeros_like(img)
    keep = good_coords > 0
    if keep.any():
        corr = (img[keep] - dark[keep]) / flood[keep] * bc
        # Re-threshold: if corrected value < goodCoords[i], drop to 0
        corr = np.where(corr < good_coords[keep], 0.0, corr)
        out[keep] = corr
    return out


__all__ = [
    "apply_image_transformations",
    "make_square_image",
    "transpose_square",
    "prepare_dark",
    "prepare_flood",
    "prepare_mask",
    "preprocess_frame",
]
