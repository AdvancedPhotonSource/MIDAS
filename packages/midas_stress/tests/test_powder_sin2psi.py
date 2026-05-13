"""Item 26 — Powder sin²ψ stress stub."""
from __future__ import annotations

import numpy as np
import pytest

from midas_stress.powder import (
    Sin2PsiResult,
    extract_d_vs_psi,
    fit_sin2psi,
)


def test_extract_d_vs_psi_recovers_planted_pattern():
    n_eta, n_r = 360, 64
    eta = np.linspace(-180.0, 180.0, n_eta, endpoint=False)
    d_axis = np.linspace(1.0, 3.0, n_r)
    d0 = 2.0
    # Plant a planted ψ-dependent shift: d(ψ) = d0 (1 + ε(ψ))
    psi_true = np.abs(eta - 90.0)
    eps_true = 1e-3 * np.sin(np.deg2rad(psi_true)) ** 2
    d_true = d0 * (1.0 + eps_true)
    int2d = np.zeros((n_eta, n_r), dtype=np.float64)
    for i, d_i in enumerate(d_true):
        # Gaussian peak at d_i with FWHM ~ 0.05 Å
        int2d[i] = np.exp(-((d_axis - d_i) / 0.02) ** 2)
    psi_out, d_out = extract_d_vs_psi(
        int2d, eta, d_axis, hkl_d0=d0, capture_radius=0.1,
    )
    np.testing.assert_allclose(psi_out, psi_true, rtol=0, atol=1e-9)
    np.testing.assert_allclose(d_out, d_true, rtol=1e-3)


def test_fit_sin2psi_recovers_planted_strain_slope():
    psi_deg = np.linspace(0.0, 90.0, 91)
    d0 = 2.0
    # ε(ψ) = 1.5e-3 sin²ψ + 1e-5 baseline
    eps = 1.5e-3 * np.sin(np.deg2rad(psi_deg)) ** 2 + 1e-5
    d_spacing = d0 * (1.0 + eps)
    res = fit_sin2psi(psi_deg, d_spacing, d0)
    assert isinstance(res, Sin2PsiResult)
    assert res.epsilon_phi_phi_slope == pytest.approx(1.5e-3, rel=1e-3)
    assert res.intercept == pytest.approx(1e-5, abs=2e-6)
    assert res.rms_residual < 1e-5


def test_fit_sin2psi_with_xecs_returns_stress():
    psi_deg = np.linspace(0.0, 90.0, 91)
    d0 = 2.0
    sigma_planted = 100.0  # MPa
    s2 = 1e-5              # 1/MPa, dummy XEC
    s1 = -0.3 * 0.5 * s2
    eps = (s2 / 2.0) * sigma_planted * np.sin(np.deg2rad(psi_deg)) ** 2
    d_spacing = d0 * (1.0 + eps)
    res = fit_sin2psi(psi_deg, d_spacing, d0, s1=s1, s2=s2)
    assert res.sigma_phi_phi == pytest.approx(sigma_planted, rel=1e-3)


def test_fit_sin2psi_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_sin2psi(np.array([0.0, 30.0]), np.array([2.0, 2.001]), 2.0)
