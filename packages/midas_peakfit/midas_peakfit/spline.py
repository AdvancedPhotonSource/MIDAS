"""Differentiable thin-plate spline (TPS) for Stage 3 residual coupling.

scipy's ``RBFInterpolator`` is fast but not differentiable, so the
v2 4-stage workflow currently fits the spline post-hoc on Stage 2's
residuals.  This module provides a torch-native TPS that lets the
spline weights be refined jointly with geometry — i.e., an LM step
that minimises ``r(geom) - spline(Y, Z)`` over geometry AND ``w``.

The basis is identical to scipy's ``thin_plate_spline``:

    φ(r) = r² log(r)         for r > 0,  φ(0) = 0

with optional polynomial tail (degree 1) so the spline reproduces a
plane exactly.  Smoothing parameter ``s`` adds ``s I`` to the
basis-on-self matrix (matches scipy's convention).

Train (one-shot):

    K = φ(|x_i - x_j|)        # (n, n)
    P = [1, y, z]             # (n, 3) polynomial tail
    Solve [[K + sI, P], [P', 0]] [[w], [c]] = [[dR], [0]]

Predict:

    f(y, z) = Σ w_j φ(|x_q - x_j|) + c₀ + c_y · y + c_z · z

Both train and predict are torch ops, so the spline output is
differentiable in (y, z, w, c) for joint refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TPSpline:
    """Trained thin-plate spline.

    Stored as: control points ``X`` (n, 2), weights ``w`` (n,), tail
    coeffs ``c`` (3,).  All torch tensors.
    """
    X: torch.Tensor      # [n, 2]
    w: torch.Tensor      # [n]
    c: torch.Tensor      # [3]  (constant, dy, dz)
    smoothing: float

    def predict(self, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Evaluate the spline at query points (Y, Z).  Differentiable in
        (Y, Z, w, c) — control points X stay fixed (sample fixed)."""
        # |x_q - x_j| pairwise distance (n_query, n_ctrl).
        Xq = torch.stack([Y, Z], dim=-1)            # [m, 2]
        diff = Xq.unsqueeze(1) - self.X.unsqueeze(0)  # [m, n, 2]
        r = torch.sqrt((diff * diff).sum(-1).clamp(min=1e-30))
        # φ(r) = r² log r  with r=0 → 0.
        phi = torch.where(r > 0, r * r * torch.log(r), torch.zeros_like(r))
        kernel_part = phi @ self.w
        tail = self.c[0] + self.c[1] * Y + self.c[2] * Z
        return kernel_part + tail


def fit_tps(
    Y: torch.Tensor,                    # [n] control y
    Z: torch.Tensor,                    # [n] control z
    dR: torch.Tensor,                   # [n] target values
    *,
    smoothing: float = 0.0,
    dtype=torch.float64,
) -> TPSpline:
    """One-shot TPS solve via the augmented linear system."""
    Y = Y.to(dtype); Z = Z.to(dtype); dR = dR.to(dtype)
    n = Y.numel()
    X = torch.stack([Y, Z], dim=-1)
    diff = X.unsqueeze(1) - X.unsqueeze(0)
    r = torch.sqrt((diff * diff).sum(-1).clamp(min=1e-30))
    K = torch.where(r > 0, r * r * torch.log(r), torch.zeros_like(r))
    if smoothing > 0:
        K = K + smoothing * torch.eye(n, dtype=dtype, device=K.device)
    P = torch.stack([torch.ones(n, dtype=dtype, device=Y.device), Y, Z], dim=-1)  # [n, 3]
    # Augmented system:  [K  P ] [w]   [dR]
    #                    [P' 0 ] [c] = [0 ]
    A = torch.zeros(n + 3, n + 3, dtype=dtype, device=K.device)
    b = torch.zeros(n + 3, dtype=dtype, device=K.device)
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.transpose(0, 1)
    b[:n] = dR
    sol = torch.linalg.solve(A, b)
    w = sol[:n]
    c = sol[n:]
    return TPSpline(X=X.detach(), w=w.detach(), c=c.detach(), smoothing=float(smoothing))


def fit_tps_refinable(
    Y: torch.Tensor,                    # control points (fixed sample)
    Z: torch.Tensor,
    dR: torch.Tensor,                   # initial targets — solved once for init
    *,
    smoothing: float = 0.0,
    dtype=torch.float64,
) -> TPSpline:
    """Same as :func:`fit_tps` but returns a TPSpline whose ``w`` and ``c``
    are leaf tensors with ``requires_grad=True`` (so the LM can refine
    them).  Initial values come from a one-shot solve at the input ``dR``.
    """
    sp = fit_tps(Y, Z, dR, smoothing=smoothing, dtype=dtype)
    sp.w = sp.w.detach().clone().requires_grad_(True)
    sp.c = sp.c.detach().clone().requires_grad_(True)
    return sp


__all__ = ["TPSpline", "fit_tps", "fit_tps_refinable"]
