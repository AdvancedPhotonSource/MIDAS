"""Tests for the generic Levenberg-Marquardt API.

Two angles:
1. ``lm_solve_generic`` recovers a known-truth solution on a synthetic Gaussian
   batch fit using both autograd and a hand-written analytic Jacobian, with
   results agreeing to fp64 epsilon.
2. ``lm_solve_arrowhead`` solves a joint-Gaussian problem (one shared mean,
   per-region amplitude) and matches a full-Jacobian solve to machine
   precision.
"""
from __future__ import annotations

import pytest
import torch

from midas_peakfit import (
    GenericLMConfig,
    lm_solve_arrowhead,
    lm_solve_generic,
)
from midas_peakfit.reparam import u_to_x


def _gaussian(x, mu, sigma, amp):
    return amp * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def test_generic_lm_recovers_gaussian_autograd():
    torch.manual_seed(0)
    B, M = 16, 64
    x_grid = torch.linspace(-5, 5, M, dtype=torch.float64).unsqueeze(0).expand(B, -1)

    # ground truth: mu in [-3, 3], sigma in [0.5, 2.0], amp in [0.5, 5.0]
    mu_t = torch.empty(B, dtype=torch.float64).uniform_(-3, 3)
    sigma_t = torch.empty(B, dtype=torch.float64).uniform_(0.5, 2.0)
    amp_t = torch.empty(B, dtype=torch.float64).uniform_(0.5, 5.0)
    y = _gaussian(x_grid, mu_t.unsqueeze(-1), sigma_t.unsqueeze(-1), amp_t.unsqueeze(-1))
    y += 0.001 * torch.randn_like(y)

    lo = torch.tensor([[-5.0, 0.1, 0.1]], dtype=torch.float64).expand(B, -1).contiguous()
    hi = torch.tensor([[5.0, 5.0, 10.0]], dtype=torch.float64).expand(B, -1).contiguous()
    x0 = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64).expand(B, -1).contiguous()

    def residual_fn(u, lo_, hi_):
        x = u_to_x(u, lo_, hi_)
        mu, sigma, amp = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        pred = amp * torch.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
        return (pred - y).contiguous()

    cfg = GenericLMConfig(max_iter=300, ftol_rel=1e-12, xtol_rel=1e-10, matmul_precision="fp64")
    x_final, cost, rc = lm_solve_generic(x0, lo, hi, residual_fn, config=cfg)

    assert (rc == 0).all(), f"some regions did not converge: {rc.tolist()}"
    # noise-limited recovery — these are recovery tolerances against ground truth
    torch.testing.assert_close(x_final[:, 0], mu_t, atol=2e-3, rtol=0)
    torch.testing.assert_close(x_final[:, 1], sigma_t, atol=2e-3, rtol=0)
    torch.testing.assert_close(x_final[:, 2], amp_t, atol=2e-3, rtol=0)


def test_arrowhead_lm_recovers_shared_mean_problem():
    """Joint problem: K regions share a single dense parameter (mean), plus
    per-region amplitudes (block diagonal)."""
    torch.manual_seed(1)
    K = 5            # regions
    M = 32           # samples per region
    n_dense = 1      # one shared mean
    N_block = 1      # one amplitude per region
    sigma = 1.0
    mu_t = 0.7
    amp_t = torch.tensor([1.0, 0.5, 2.0, 1.5, 0.8], dtype=torch.float64)

    x_grid = torch.linspace(-3, 3, M, dtype=torch.float64)
    y = torch.empty(K, M, dtype=torch.float64)
    for k in range(K):
        y[k] = amp_t[k] * torch.exp(-0.5 * ((x_grid - mu_t) / sigma) ** 2)

    # Joint parameter vector: [mu, amp_0, amp_1, ..., amp_{K-1}]
    N_total = n_dense + K * N_block
    x_init = torch.tensor([0.0] + [1.0] * K, dtype=torch.float64)
    lo = torch.tensor([-3.0] + [0.0] * K, dtype=torch.float64)
    hi = torch.tensor([3.0] + [5.0] * K, dtype=torch.float64)

    block_offsets = torch.tensor([k * M for k in range(K + 1)], dtype=torch.long)

    def residual_fn(u, lo_, hi_):
        # u shape [1, N_total]
        x = u_to_x(u, lo_, hi_).squeeze(0)
        mu = x[0]
        amps = x[n_dense:]
        out = torch.empty(K * M, dtype=torch.float64)
        for k in range(K):
            pred = amps[k] * torch.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
            out[k * M:(k + 1) * M] = pred - y[k]
        return out.unsqueeze(0)

    def jacobian_fn(u, lo_, hi_):
        """Hand-derived block-arrow Jacobian.

        Residual r_{k,m} = amp_k · g(x_m, mu) - y_{k,m}  with  g(x, μ) = exp(-(x-μ)²/2σ²).
        ∂r/∂μ      = amp_k · g · (x_m - μ) / σ²              → dense column [M_total, 1]
        ∂r/∂amp_k  = g(x_m, μ) for residuals in block k       → block diagonal [K, M, 1]
        Plus the chain rule for u→x via sigmoid reparameterization.
        """
        x = u_to_x(u, lo_, hi_).squeeze(0)
        mu = x[0]
        amps = x[n_dense:]
        # dx/du for sigmoid reparam: (hi-lo) * σ(u) * (1-σ(u)) where σ(u)=(x-lo)/(hi-lo)
        ub = u.squeeze(0)
        x_minus_lo = x - lo_.squeeze(0)
        hi_minus_x = hi_.squeeze(0) - x
        dx_du = (hi_minus_x * x_minus_lo / (hi_.squeeze(0) - lo_.squeeze(0))).clamp(min=0.0)

        g_vals = torch.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)         # [M]
        dg_dmu = g_vals * (x_grid - mu) / (sigma ** 2)                 # [M]

        r = residual_fn(u, lo_, hi_).squeeze(0)
        # J_dense [K*M, 1] = ∂r/∂mu = amp_k · dg/dμ
        J_dense = torch.empty(K * M, n_dense, dtype=torch.float64)
        for k in range(K):
            J_dense[k * M:(k + 1) * M, 0] = amps[k] * dg_dmu
        J_dense = J_dense * dx_du[0]  # chain rule for μ

        # J_blocks [K, M, 1] = ∂r/∂amp_k = g_vals  (only for residuals in block k)
        J_blocks = torch.empty(K, M, N_block, dtype=torch.float64)
        for k in range(K):
            J_blocks[k, :, 0] = g_vals * dx_du[1 + k]  # chain rule for amp_k

        return r.unsqueeze(0), J_dense.unsqueeze(0), J_blocks.unsqueeze(0)

    cfg = GenericLMConfig(max_iter=200, ftol_rel=1e-12, xtol_rel=1e-10)
    x_final, cost, rc = lm_solve_arrowhead(
        x_init, lo, hi,
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        n_dense=n_dense,
        block_residual_offsets=block_offsets,
        config=cfg,
    )
    assert rc == 0, f"arrowhead LM did not converge: rc={rc}"
    assert abs(x_final[0].item() - mu_t) < 1e-3
    for k in range(K):
        assert abs(x_final[1 + k].item() - amp_t[k].item()) < 1e-3


def test_huber_loss_robust_to_outlier():
    """Huber-flagged generic LM ignores a single bad outlier."""
    torch.manual_seed(2)
    B, M = 4, 50
    x_grid = torch.linspace(-3, 3, M, dtype=torch.float64).unsqueeze(0).expand(B, -1)
    mu_t = torch.tensor([0.5, -0.3, 1.2, -1.0], dtype=torch.float64)
    amp_t = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    sigma = 1.0
    y = amp_t.unsqueeze(-1) * torch.exp(-0.5 * ((x_grid - mu_t.unsqueeze(-1)) / sigma) ** 2)
    # inject a 100σ outlier per region
    y[:, M // 2] += 100.0

    lo = torch.tensor([[-3.0, 0.1]], dtype=torch.float64).expand(B, -1).contiguous()
    hi = torch.tensor([[3.0, 5.0]], dtype=torch.float64).expand(B, -1).contiguous()
    x0 = torch.tensor([[0.0, 1.0]], dtype=torch.float64).expand(B, -1).contiguous()

    def residual_fn(u, lo_, hi_):
        x = u_to_x(u, lo_, hi_)
        mu = x[..., 0:1]; amp = x[..., 1:2]
        pred = amp * torch.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
        return (pred - y).contiguous()

    cfg_huber = GenericLMConfig(max_iter=400, ftol_rel=1e-10, xtol_rel=1e-10, huber_delta=0.5)
    x_h, cost_h, rc_h = lm_solve_generic(x0, lo, hi, residual_fn, config=cfg_huber)

    cfg_l2 = GenericLMConfig(max_iter=400, ftol_rel=1e-10, xtol_rel=1e-10)
    x_l, cost_l, rc_l = lm_solve_generic(x0, lo, hi, residual_fn, config=cfg_l2)

    err_h = (x_h[:, 0] - mu_t).abs().mean()
    err_l = (x_l[:, 0] - mu_t).abs().mean()
    assert err_h < err_l, (
        f"Huber should be more robust than L2 here: |Δμ|_huber={err_h:.4f}, |Δμ|_L2={err_l:.4f}"
    )
