"""Family III mitigations — post-process filters on integrated cakes.

Functions here operate on a 2-D cake of shape ``(n_r, n_eta)`` (numpy
or torch) and return a cake of the same shape. The filters are
designed to preserve the per-ring total flux exactly when used with
the area weights from :class:`midas_integrate.kernels.CSRGeometry`.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:                 # pragma: no cover
    _HAS_TORCH = False
    torch = None                    # type: ignore

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_np(arr: ArrayLike) -> tuple[np.ndarray, bool]:
    """Coerce to numpy; remember whether the input was a torch tensor."""
    if _HAS_TORCH and isinstance(arr, torch.Tensor):  # type: ignore[union-attr]
        return arr.detach().cpu().numpy(), True
    return np.asarray(arr), False


def _maybe_to_torch(arr: np.ndarray, was_torch: bool, like: ArrayLike) -> ArrayLike:
    if was_torch and _HAS_TORCH:
        ref: torch.Tensor = like     # type: ignore[assignment]
        return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    return arr


def gauss_smooth_eta(
    cake: ArrayLike,
    sigma_bins: float,
    *,
    area_per_bin: Optional[ArrayLike] = None,
    periodic: bool = True,
) -> ArrayLike:
    """Family III: 1-D Gaussian smoothing along the η axis.

    Args:
        cake: shape ``(n_r, n_eta)``. Numpy or torch.
        sigma_bins: Gaussian width in units of η bins. The implicit FWHM
            in degrees is ``sigma_bins × ΔEta × 2.355``.
        area_per_bin: optional 1-D ``(n_r * n_eta,)`` or 2-D
            ``(n_r, n_eta)`` area weights. When provided, the smoothing
            is area-weighted, which preserves the per-ring total
            flux *exactly* even when adjacent η-bins have unequal area
            (e.g. partially masked pixels). When ``None``, a uniform
            convolution is applied (still conserves total flux when all
            bins have the same area).
        periodic: assume the η axis is periodic ([-180°, 180°)). If
            ``False``, edge bins are reflected instead.

    Returns:
        Smoothed cake, same dtype/device/shape as input.
    """
    cake_np, was_torch = _to_np(cake)
    n_r, n_eta = cake_np.shape

    half = max(1, int(round(4 * sigma_bins)))
    k = np.arange(-half, half + 1, dtype=np.float64)
    w = np.exp(-(k ** 2) / (2.0 * sigma_bins ** 2))
    w /= w.sum()

    if area_per_bin is not None:
        A_np, _ = _to_np(area_per_bin)
        A = A_np.reshape(n_r, n_eta).astype(np.float64)
        IA = cake_np.astype(np.float64) * A
    else:
        A = np.ones_like(cake_np, dtype=np.float64)
        IA = cake_np.astype(np.float64)

    pad = half
    if periodic:
        IA_pad = np.concatenate([IA[:, -pad:], IA, IA[:, :pad]], axis=1)
        A_pad = np.concatenate([A[:, -pad:], A, A[:, :pad]], axis=1)
    else:
        IA_pad = np.pad(IA, ((0, 0), (pad, pad)), mode="reflect")
        A_pad = np.pad(A, ((0, 0), (pad, pad)), mode="reflect")

    out_IA = np.zeros_like(IA)
    out_A = np.zeros_like(A)
    for kk, ww in zip(k.astype(int), w):
        out_IA += ww * IA_pad[:, pad + kk : pad + kk + n_eta]
        out_A += ww * A_pad[:, pad + kk : pad + kk + n_eta]

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(out_A > 0, out_IA / out_A,
                       cake_np.astype(np.float64))
    return _maybe_to_torch(out.astype(cake_np.dtype, copy=False),
                           was_torch, cake)


def median_filter_eta(
    cake: ArrayLike,
    window: int = 3,
    *,
    periodic: bool = True,
    preserve_ring_flux: bool = True,
) -> ArrayLike:
    """Family III: 1-D median filter along the η axis.

    Args:
        cake: shape ``(n_r, n_eta)``.
        window: window size (must be odd).
        periodic: assume periodic η axis.
        preserve_ring_flux: if ``True``, rescale each row so that
            ``Σ_η I_smoothed(R, η) == Σ_η I_input(R, η)``. The median
            itself does not preserve flux; this rescaling is a small
            multiplicative correction.
    """
    cake_np, was_torch = _to_np(cake)
    n_r, n_eta = cake_np.shape
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    half = window // 2

    if periodic:
        cake_pad = np.concatenate([cake_np[:, -half:], cake_np,
                                    cake_np[:, :half]], axis=1)
    else:
        cake_pad = np.pad(cake_np, ((0, 0), (half, half)), mode="reflect")

    out = np.empty_like(cake_np, dtype=cake_np.dtype)
    for j in range(n_eta):
        out[:, j] = np.median(cake_pad[:, j : j + window], axis=1)

    if preserve_ring_flux:
        s_in = cake_np.sum(axis=1, keepdims=True)
        s_out = out.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(s_out != 0, s_in / s_out, 1.0)
        out = out * scale

    return _maybe_to_torch(out, was_torch, cake)
