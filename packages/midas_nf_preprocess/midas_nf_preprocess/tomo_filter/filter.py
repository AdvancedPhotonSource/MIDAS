"""Tomography-based hex-grid masking.

Two modes:

  1. ``filter_grid_by_tomo``  -- C parity: a 2D uint8 image, square dimensions
     inferred from the file size. Grid coordinates are mapped to pixel indices
     with the C convention (Y axis flipped).
  2. ``filter_grid_by_bbox``  -- Python convenience: keep grid points inside a
     ``[x_min, x_max, y_min, y_max]`` rectangle in micrometers. This mirrors
     the ``GridMask`` parameter handled in the Python workflow driver
     (``nf_MIDAS.py:376-386``) without touching disk.

The point-in-image lookup is vectorized; gradient does not flow through this
operation (it returns a boolean mask).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import math

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Tomo image loading
# -----------------------------------------------------------------------------


def load_square_tomo(
    path: Union[str, Path], *, dtype: np.dtype = np.uint8
) -> np.ndarray:
    """Load a square binary tomography image, inferring side length from file size.

    Mirrors filterGridfromTomo.c L13-L21:

        sz = stat.st_size
        nrPxTomo = sqrt(sz)
        imTomo = uint8 buffer of shape (nrPxTomo, nrPxTomo)
    """
    path = Path(path)
    sz = path.stat().st_size
    nr_px = int(math.isqrt(sz)) if dtype == np.uint8 else int(math.sqrt(sz / np.dtype(dtype).itemsize))
    if nr_px * nr_px * np.dtype(dtype).itemsize != sz:
        raise ValueError(
            f"{path}: size {sz} bytes is not a perfect square for dtype {dtype}"
        )
    arr = np.fromfile(path, dtype=dtype, count=nr_px * nr_px)
    return arr.reshape(nr_px, nr_px)


# -----------------------------------------------------------------------------
# Tomo sampling
# -----------------------------------------------------------------------------


def sample_tomo(
    points_xy_um: torch.Tensor,
    tomo: Union[np.ndarray, torch.Tensor],
    px_tomo_um: float,
) -> torch.Tensor:
    """Sample tomography mask values at grid-point locations.

    Coordinate convention (matches filterGridfromTomo.c L39-L43):

        xPos = int(x_um / pxTomo) + nrPxTomo // 2
        yPos = int(y_um / pxTomo) + nrPxTomo // 2
        value = tomo[nrPxTomo - yPos, xPos]      # Y flipped

    Out-of-image points return zero.

    Parameters
    ----------
    points_xy_um : Tensor of shape ``(N, 2)``, columns ``(x, y)`` in um.
    tomo         : 2D uint8 array (numpy or torch).
    px_tomo_um   : pixel size of the tomo image, in um/pixel.

    Returns
    -------
    Int64 Tensor of shape ``(N,)`` with the looked-up mask value for each point.
    Out-of-image points get 0.
    """
    if points_xy_um.ndim != 2 or points_xy_um.shape[1] != 2:
        raise ValueError(
            f"Expected (N, 2) points, got shape {tuple(points_xy_um.shape)}"
        )
    if isinstance(tomo, np.ndarray):
        tomo_t = torch.from_numpy(tomo)
    else:
        tomo_t = tomo
    if tomo_t.ndim != 2 or tomo_t.shape[0] != tomo_t.shape[1]:
        raise ValueError(
            f"Expected square tomo image, got shape {tuple(tomo_t.shape)}"
        )
    nr_px = tomo_t.shape[0]
    device = points_xy_um.device
    tomo_t = tomo_t.to(device=device)

    x_um = points_xy_um[:, 0]
    y_um = points_xy_um[:, 1]
    # C casts to (int) which truncates toward zero for negatives; PyTorch's
    # .to(torch.int64) on a negative float also truncates toward zero. Match.
    x_pos = (x_um / px_tomo_um).to(torch.int64) + nr_px // 2
    y_pos = (y_um / px_tomo_um).to(torch.int64) + nr_px // 2

    in_bounds = (
        (x_pos >= 0) & (x_pos < nr_px) & (y_pos >= 0) & (y_pos < nr_px)
    )
    # Y-flip: row index = nrPxTomo - yPos
    row = nr_px - y_pos
    # Clamp out-of-bounds to a safe index (we'll mask the result).
    row_safe = torch.where(in_bounds, row, torch.zeros_like(row))
    col_safe = torch.where(in_bounds, x_pos, torch.zeros_like(x_pos))
    values = tomo_t[row_safe, col_safe].to(torch.int64)
    return torch.where(in_bounds, values, torch.zeros_like(values))


def filter_grid_by_tomo(
    grid_points: torch.Tensor,
    tomo: Union[np.ndarray, torch.Tensor],
    px_tomo_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep grid points whose tomo lookup is non-zero.

    Parameters
    ----------
    grid_points : Tensor of shape ``(N, 5)`` -- columns ``(dx, dy, x, y, edge_half)``,
        the format from ``hex_grid.make_hex_grid``.
    tomo : square 2D mask (numpy uint8 or torch tensor of any int/bool dtype).
    px_tomo_um : pixel size of ``tomo`` in um/pixel.

    Returns
    -------
    (filtered_points, mask) where:

      - ``filtered_points`` has shape ``(K, 5)`` for the K kept points.
      - ``mask`` has shape ``(N,)`` with ``True`` for kept points.
    """
    if grid_points.ndim != 2 or grid_points.shape[1] != 5:
        raise ValueError(
            f"Expected (N, 5) grid points, got shape {tuple(grid_points.shape)}"
        )
    xy = grid_points[:, [2, 3]]
    values = sample_tomo(xy, tomo, px_tomo_um)
    mask = values != 0
    return grid_points[mask], mask


# -----------------------------------------------------------------------------
# Bounding-box masking (Python convenience, mirrors GridMask in nf_MIDAS.py)
# -----------------------------------------------------------------------------


def bbox_mask(
    grid_points: torch.Tensor,
    bbox_um: Sequence[float],
) -> torch.Tensor:
    """Boolean mask: True where ``(x, y)`` is inside ``bbox_um = [xmin, xmax, ymin, ymax]``."""
    if len(bbox_um) != 4:
        raise ValueError(
            f"bbox_um must be [xmin, xmax, ymin, ymax]; got length {len(bbox_um)}"
        )
    xmin, xmax, ymin, ymax = bbox_um
    if xmax < xmin or ymax < ymin:
        raise ValueError(f"bbox_um corners reversed: {bbox_um}")
    x = grid_points[:, 2]
    y = grid_points[:, 3]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def filter_grid_by_bbox(
    grid_points: torch.Tensor,
    bbox_um: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only grid points inside the rectangular bbox."""
    mask = bbox_mask(grid_points, bbox_um)
    return grid_points[mask], mask
