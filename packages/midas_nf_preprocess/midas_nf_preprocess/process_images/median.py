"""Temporal and spatial median filters.

Both are differentiable in the subgradient sense: ``torch.median`` is piecewise
linear in the input; the gradient flows through whichever element happened to be
selected as the median. That matches the behavior of the C ``quick_select``.

Border handling for the spatial median matches ``ProcessImagesCombined.c`` L957-L990:
pixels within ``radius`` of any edge pass through unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def temporal_median(stack: torch.Tensor) -> torch.Tensor:
    """Per-pixel median across the first (frame) axis.

    Parameters
    ----------
    stack : Tensor of shape ``[N, Z, Y]``

    Returns
    -------
    Tensor of shape ``[Z, Y]`` with dtype matching ``stack``.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected [N, Z, Y], got shape {tuple(stack.shape)}")
    # torch.median.values is the actual median (selected element); torch.median has a
    # well-defined backward that routes the gradient to the selected index.
    return stack.median(dim=0).values


def _unfold_blocks(img: torch.Tensor, k: int) -> torch.Tensor:
    """Extract k x k neighborhoods around every pixel as the last dim.

    Returns a tensor of shape ``[Z - 2r, Y - 2r, k*k]`` where r = (k-1)/2. No padding;
    the caller is responsible for masking the border to match the C semantics.
    """
    z, y = img.shape
    # F.unfold expects [B, C, H, W]; we use [1, 1, Z, Y] and extract sliding windows.
    blocks = F.unfold(img.unsqueeze(0).unsqueeze(0), kernel_size=k, padding=0)
    # blocks: [1, k*k, (Z-2r)*(Y-2r)]
    r = (k - 1) // 2
    return blocks.squeeze(0).T.reshape(z - 2 * r, y - 2 * r, k * k)


def spatial_median(img: torch.Tensor, radius: int) -> torch.Tensor:
    """Replace each interior pixel with the median of its (2r+1) x (2r+1) neighborhood.

    Border behavior matches the C: pixels within ``radius`` of any edge pass through
    unchanged. ``radius=0`` returns ``img`` unchanged.

    Parameters
    ----------
    img : Tensor of shape ``[Z, Y]``
    radius : int, 0 <= radius. Common values: 0 (identity), 1 (3x3), 2 (5x5).

    Returns
    -------
    Tensor of shape ``[Z, Y]``.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected [Z, Y], got shape {tuple(img.shape)}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if radius == 0:
        return img

    k = 2 * radius + 1
    z, y = img.shape
    if z < k or y < k:
        # Window doesn't fit anywhere: pass through unchanged, like the C border path.
        return img

    blocks = _unfold_blocks(img, k)  # [Z-2r, Y-2r, k*k]
    interior_med = blocks.median(dim=-1).values  # [Z-2r, Y-2r]

    # Stitch into a full-size output: edges = original, interior = median.
    out = img.clone()
    out[radius : z - radius, radius : y - radius] = interior_med
    return out
