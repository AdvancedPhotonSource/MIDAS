"""Laplacian-of-Gaussian filter for blob/peak detection.

Two modes:

- ``integer=False`` (default): float kernel, fully differentiable in ``sigma``.
- ``integer=True``: int64 kernel matching the C ``LoGFilt`` quantization in
  ``ProcessImagesCombined.c`` L251-L258 (magic factor 79720). Kept for parity
  diagnostics only -- not differentiable in ``sigma`` and not supported on MPS.

Kernel definition (continuous form):

    k(x, y) = -(1 / (pi * sigma^4))
             * (1 - (x^2 + y^2) / (2 * sigma^2))
             * exp(-(x^2 + y^2) / (2 * sigma^2))

Sampled at integer offsets in [-radius, radius]^2.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F


_C_INT_MAGIC = 79720  # ProcessImagesCombined.c L252


def build_log_kernel(
    radius: int,
    sigma: Union[float, torch.Tensor],
    *,
    integer: bool = False,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build a 2D LoG kernel of shape ``[2r+1, 2r+1]``.

    Parameters
    ----------
    radius : int. Half-extent of the kernel; the result is (2r+1) x (2r+1).
    sigma  : float or 0-d Tensor. Gaussian width.
    integer : if True, returns an int64 kernel matching the C quantization. The
        sigma is then treated as a scalar and the resulting kernel is the same as
        ``round(_C_INT_MAGIC * float_kernel)``. Not differentiable in ``sigma``.
    device, dtype : standard torch construction kwargs. ``dtype`` defaults to the
        dtype of ``sigma`` if it is a tensor, else to ``torch.float64``.

    Returns
    -------
    Tensor of shape ``[2r+1, 2r+1]``.
    """
    if radius < 1:
        raise ValueError(f"LoG radius must be >= 1, got {radius}")
    device = torch.device(device)
    if dtype is None:
        dtype = sigma.dtype if isinstance(sigma, torch.Tensor) else torch.float64

    # Build the (x, y) grid identically to the C double loop:
    #   for i in [-r, r]: for j in [-r, r]: FiltXYs[cr] = (j, i)
    # which is equivalent to a meshgrid with row=i, col=j.
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # yy=row=i, xx=col=j
    r2 = xx * xx + yy * yy

    if not isinstance(sigma, torch.Tensor):
        sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
    else:
        sigma_t = sigma.to(device=device, dtype=dtype)
    s2 = sigma_t * sigma_t
    s4 = s2 * s2
    arg = r2 / (2.0 * s2)
    # k(x,y) = -(1/(pi*s^4)) * (1 - r^2/(2s^2)) * exp(-r^2/(2s^2))
    kernel = -(1.0 / (math.pi * s4)) * (1.0 - arg) * torch.exp(-arg)

    if integer:
        # Match the C casts: kernel = (long) (79720 * raw). The C uses C-style
        # truncation toward zero via `(int)`, not banker's rounding.
        scaled = _C_INT_MAGIC * kernel.detach()
        return scaled.to(torch.int64)
    return kernel


def apply_log(
    img: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Convolve ``img`` with ``kernel`` and zero the border (matches C semantics).

    The C ``FindPeakPositions`` (L260-L277) sets ``Image3[i] = 0`` for any pixel
    within ``radius`` of an edge. We replicate that by applying the convolution with
    ``padding=0`` and zero-padding the result back to the original size.

    Parameters
    ----------
    img    : Tensor of shape ``[Z, Y]``.
    kernel : Tensor of shape ``[2r+1, 2r+1]``. Must match img's dtype unless integer.

    Returns
    -------
    Tensor of shape ``[Z, Y]``. Same dtype as the inputs (for integer kernels, the
    output is int64).
    """
    if img.ndim != 2:
        raise ValueError(f"Expected img [Z, Y], got shape {tuple(img.shape)}")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Expected square kernel, got shape {tuple(kernel.shape)}")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    radius = (kernel.shape[0] - 1) // 2

    if kernel.dtype != img.dtype:
        # int kernels need an int image to keep dtype consistent
        if kernel.dtype.is_floating_point != img.dtype.is_floating_point:
            raise ValueError(
                f"Kernel dtype {kernel.dtype} incompatible with img dtype {img.dtype}"
            )
        kernel = kernel.to(img.dtype)

    out_full = F.conv2d(
        img.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=0,
    ).squeeze(0).squeeze(0)
    # Pad the convolved interior back to the input size with zeros.
    out = torch.zeros_like(img)
    out[radius:-radius, radius:-radius] = out_full
    return out
