"""Tests for the ParameterSpec / pack / lm_minimise / laplace / TPSpline
substrate that was promoted from midas_calibrate_v2 in v0.4.0.
"""
from __future__ import annotations

import math

import pytest
import torch

import midas_peakfit as mp


def _line_problem():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 50, dtype=torch.float64)
    y_true = 1.5 * x + 0.7
    y_obs = y_true + 0.01 * torch.randn_like(y_true)
    spec = mp.ParameterSpec()
    spec.add(mp.Parameter("a", init=0.5, bounds=(-3.0, 3.0)))
    spec.add(mp.Parameter("b", init=0.0, bounds=(-3.0, 3.0)))

    def residual(unpacked):
        return unpacked["a"] * x + unpacked["b"] - y_obs

    return spec, residual, y_obs, x


# --------------------------------------------------------- ParameterSpec

def test_parameter_spec_add_and_freeze():
    spec = mp.ParameterSpec()
    spec.add(mp.Parameter("Lsd", init=1000.0))
    spec.add(mp.Parameter("BC_y", init=512.0, refined=False))
    assert "Lsd" in spec
    assert spec.refined_names() == ["Lsd"]
    spec.thaw("BC_y")
    assert set(spec.refined_names()) == {"Lsd", "BC_y"}
    spec.freeze("Lsd")
    assert spec.refined_names() == ["BC_y"]


def test_parameter_spec_duplicate_raises():
    spec = mp.ParameterSpec()
    spec.add(mp.Parameter("Lsd", init=1.0))
    with pytest.raises(ValueError, match="already exists"):
        spec.add(mp.Parameter("Lsd", init=2.0))


def test_calibration_spec_subclasses_parameter_spec():
    """Paper-3's CalibrationSpec must be a true ParameterSpec subclass so the
    pack/unpack/lm/laplace machinery works on it interchangeably."""
    pytest.importorskip("midas_calibrate_v2")
    from midas_calibrate_v2.parameters.spec import CalibrationSpec

    assert issubclass(CalibrationSpec, mp.ParameterSpec)
    cs = CalibrationSpec()
    cs.add(mp.Parameter("Lsd", init=1000.0))
    # Powder-specific extras:
    cs.SpaceGroup = 225
    cs.LatticeConstant = (5.41, 5.41, 5.41, 90, 90, 90)
    assert "Lsd" in cs
    assert cs.SpaceGroup == 225


# --------------------------------------------------------- pack/unpack round-trip

def test_pack_unpack_roundtrip_scalar():
    spec = mp.ParameterSpec()
    spec.add(mp.Parameter("a", init=1.5))
    spec.add(mp.Parameter("b", init=-0.3))
    x, info = mp.pack_spec(spec)
    unpacked = mp.unpack_spec(x, info, spec)
    assert torch.allclose(unpacked["a"], torch.tensor(1.5, dtype=torch.float64))
    assert torch.allclose(unpacked["b"], torch.tensor(-0.3, dtype=torch.float64))


def test_pack_unpack_roundtrip_vector():
    """Vector parameters (e.g. per-panel shifts) must preserve shape."""
    spec = mp.ParameterSpec()
    spec.add(mp.Parameter("panel_delta_yz", init=torch.zeros(48, 2, dtype=torch.float64)))
    spec.add(mp.Parameter("Lsd", init=1000.0))
    x, info = mp.pack_spec(spec)
    assert x.numel() == 48 * 2 + 1
    unpacked = mp.unpack_spec(x, info, spec)
    assert unpacked["panel_delta_yz"].shape == (48, 2)
    assert unpacked["Lsd"].shape == ()


def test_refined_indices_skips_fixed():
    spec = mp.ParameterSpec()
    spec.add(mp.Parameter("a", init=1.0))
    spec.add(mp.Parameter("b", init=2.0, refined=False))
    spec.add(mp.Parameter("c", init=3.0))
    _, info = mp.pack_spec(spec)
    idx = mp.refined_indices(info)
    assert idx.tolist() == [0, 2]


# --------------------------------------------------------- lm_minimise

def test_lm_minimise_recovers_truth():
    spec, residual, _, _ = _line_problem()
    unpacked, cost, rc = mp.lm_minimise(
        spec, residual,
        config=mp.GenericLMConfig(max_iter=100, ftol_rel=1e-12),
        fallback_span=5.0,
    )
    assert rc == 0
    assert abs(float(unpacked["a"]) - 1.5) < 0.05
    assert abs(float(unpacked["b"]) - 0.7) < 0.05
    assert cost < 1.0


def test_lm_minimise_with_frozen_parameter():
    spec, residual, _, _ = _line_problem()
    spec.freeze("a")
    spec.set_init("a", 1.5)   # freeze at the truth
    unpacked, cost, rc = mp.lm_minimise(
        spec, residual,
        config=mp.GenericLMConfig(max_iter=100, ftol_rel=1e-12),
        fallback_span=5.0,
    )
    assert rc == 0
    # 'a' frozen at 1.5; 'b' should still recover.
    assert float(unpacked["a"]) == 1.5
    assert abs(float(unpacked["b"]) - 0.7) < 0.05


# --------------------------------------------------------- Laplace / Fisher

def test_laplace_at_map_well_conditioned():
    spec, residual, _, _ = _line_problem()
    unpacked, _, _ = mp.lm_minimise(
        spec, residual,
        config=mp.GenericLMConfig(max_iter=100, ftol_rel=1e-12),
        fallback_span=5.0,
    )

    def nll(u):
        r = residual(u)
        return 0.5 * (r * r).sum()

    lr = mp.laplace_at_map(spec, nll, unpacked, fallback_span=5.0)
    assert lr.cov.shape == (2, 2)
    assert lr.sigma_per_dim.shape == (2,)
    # Sigmas should be positive and finite.
    assert (lr.sigma_per_dim > 0).all()
    assert torch.isfinite(lr.sigma_per_dim).all()


def test_fisher_at_map_matches_laplace_for_least_squares():
    """For pure least-squares (no prior), Fisher J'J/σ² should give the same
    cov as the full Hessian of 0.5 * Σ r² when residuals are at their
    optimum.  Sigmas should agree to ~3 decimals."""
    spec, residual, y_obs, x = _line_problem()
    unpacked, _, _ = mp.lm_minimise(
        spec, residual,
        config=mp.GenericLMConfig(max_iter=100, ftol_rel=1e-12),
        fallback_span=5.0,
    )

    fr = mp.fisher_at_map(spec, residual, unpacked, sigma_r=1.0, fallback_span=5.0)

    def nll(u):
        r = residual(u)
        return 0.5 * (r * r).sum()
    lr = mp.laplace_at_map(spec, nll, unpacked, fallback_span=5.0)

    # At the optimum, J'J ≈ Hessian for nonlinear-least-squares with sigma_r=1.
    rel = (fr.sigma_per_dim - lr.sigma_per_dim).abs() / lr.sigma_per_dim
    assert (rel < 0.05).all(), f"Fisher and Laplace sigmas differ: rel={rel}"


# --------------------------------------------------------- TPSpline

def test_tpspline_interpolates_control_points():
    Y = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5], dtype=torch.float64)
    Z = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5], dtype=torch.float64)
    dR = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0], dtype=torch.float64)
    sp = mp.fit_tps(Y, Z, dR, smoothing=0.0)
    pred = sp.predict(Y, Z)
    assert torch.allclose(pred, dR, atol=1e-10)


def test_tpspline_refinable_carries_grad():
    Y = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    Z = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    dR = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    sp = mp.fit_tps_refinable(Y, Z, dR)
    assert sp.w.requires_grad
    assert sp.c.requires_grad
    Yq = torch.tensor([0.5, 0.7], dtype=torch.float64)
    Zq = torch.tensor([0.5, 0.3], dtype=torch.float64)
    pred = sp.predict(Yq, Zq)
    pred.sum().backward()
    assert sp.w.grad is not None
    assert sp.c.grad is not None


# --------------------------------------------------------- zero_sum_residual

def test_zero_sum_residual_penalises_drift():
    """Setting non-zero block sums should produce non-zero residual rows."""
    unpacked = {
        "panel_delta_yz": torch.tensor([[1.0, 0.0], [-0.3, 0.0], [0.5, 0.0]], dtype=torch.float64),
        "Lsd": torch.tensor(1000.0, dtype=torch.float64),
    }
    r = mp.zero_sum_residual(unpacked, block_names=["panel_delta_yz"], lambda_zs=1e6)
    # Sum = (1.2, 0); contribution per axis is sqrt(1e6) * sum = 1e3 * 1.2 = 1200 and 0.
    assert r.numel() == 2
    assert abs(float(r[0]) - 1200.0) < 1e-6
    assert abs(float(r[1])) < 1e-6


def test_zero_sum_residual_zero_when_balanced():
    unpacked = {
        "panel_delta_yz": torch.tensor([[1.0, -0.5], [-0.5, 0.5], [-0.5, 0.0]], dtype=torch.float64),
    }
    r = mp.zero_sum_residual(unpacked, block_names=["panel_delta_yz"], lambda_zs=1e6)
    assert r.abs().max() < 1e-9


# --------------------------------------------------------- transforms

def test_logit_inverse_roundtrip():
    t = mp.Logit(lo=-1.0, hi=2.0)
    x = torch.tensor([-0.5, 0.0, 0.5, 1.5], dtype=torch.float64)
    u = t.forward(x)
    x_round = t.inverse(u)
    assert torch.allclose(x_round, x, atol=1e-6)


def test_log_transform():
    t = mp.Log()
    x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    u = t.forward(x)
    x_round = t.inverse(u)
    assert torch.allclose(x_round, x, atol=1e-12)
