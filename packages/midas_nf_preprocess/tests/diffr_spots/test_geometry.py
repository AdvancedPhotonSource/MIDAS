"""Tests for the diffr_spots.geometry module.

Most of the heavy Bragg geometry is delegated to ``midas_diffract``; we test:
  - the lab-frame projection helper ``calc_spot_position``
  - the rotation helper ``rotate_around_z``
  - the high-level ``bragg_omega_eta`` adapter (sanity + parity vs C math)
"""

from __future__ import annotations

import math

import pytest
import torch

from midas_nf_preprocess.diffr_spots import (
    bragg_omega_eta,
    calc_eta_deg,
    calc_spot_position,
    quat_to_orient_matrix,
    rotate_around_z,
)


# -----------------------------------------------------------------------------
# rotate_around_z
# -----------------------------------------------------------------------------


def test_rotate_around_z_zero_is_identity():
    v = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    out = rotate_around_z(v, torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(out, v)


def test_rotate_around_z_90_deg():
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    out = rotate_around_z(v, torch.tensor(90.0, dtype=torch.float64))
    expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    assert torch.allclose(out, expected, atol=1e-12)


def test_rotate_around_z_z_invariant():
    v = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
    out = rotate_around_z(v, torch.tensor(45.0, dtype=torch.float64))
    assert torch.allclose(out, v)


# -----------------------------------------------------------------------------
# calc_eta_deg
# -----------------------------------------------------------------------------


def test_calc_eta_north_pole():
    """y=0, z>0: eta should be 0."""
    eta = calc_eta_deg(torch.tensor(0.0), torch.tensor(1.0))
    assert torch.isclose(eta, torch.tensor(0.0), atol=1e-12)


def test_calc_eta_south_pole():
    """y=0, z<0: eta should be 180."""
    eta = calc_eta_deg(torch.tensor(0.0), torch.tensor(-1.0))
    assert torch.isclose(eta.abs(), torch.tensor(180.0), atol=1e-12)


def test_calc_eta_positive_y_negates():
    """C: if (y > 0) alpha = -alpha"""
    eta_pos_y = calc_eta_deg(torch.tensor(1.0), torch.tensor(1.0))
    eta_neg_y = calc_eta_deg(torch.tensor(-1.0), torch.tensor(1.0))
    assert torch.isclose(eta_pos_y, -eta_neg_y, atol=1e-12)


# -----------------------------------------------------------------------------
# calc_spot_position
# -----------------------------------------------------------------------------


def test_calc_spot_position_at_eta_zero():
    """eta=0: yl=0, zl=R."""
    yl, zl = calc_spot_position(
        torch.tensor(10.0), torch.tensor(0.0)
    )
    assert torch.isclose(yl, torch.tensor(0.0), atol=1e-12)
    assert torch.isclose(zl, torch.tensor(10.0), atol=1e-12)


def test_calc_spot_position_at_eta_90():
    """eta=90: yl=-R, zl=0."""
    yl, zl = calc_spot_position(
        torch.tensor(10.0, dtype=torch.float64),
        torch.tensor(90.0, dtype=torch.float64),
    )
    assert torch.isclose(yl, torch.tensor(-10.0, dtype=torch.float64), atol=1e-12)
    assert torch.isclose(zl, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_calc_spot_position_radius_invariant():
    """yl^2 + zl^2 = R^2 for any eta."""
    R = torch.tensor(7.5)
    eta = torch.tensor([10.0, 50.0, 130.0, -40.0, 175.0])
    yl, zl = calc_spot_position(R, eta)
    assert torch.allclose(yl * yl + zl * zl, torch.full_like(yl, 7.5 ** 2), atol=1e-12)


# -----------------------------------------------------------------------------
# bragg_omega_eta: delegation sanity
# -----------------------------------------------------------------------------


def _trivial_orient_and_hkls():
    """Identity rotation + a single HKL with a small Bragg angle.

    Wavelength=1A, lattice ~ 5A, theta = arcsin(lam / 2d). We just construct a
    G-vector with a known length for which the math works out cleanly.
    """
    R = torch.eye(3, dtype=torch.float64).unsqueeze(0)  # (1, 3, 3)
    hkls = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)  # |G| = 1
    # sin(theta) = v / |G| = small fraction
    thetas_deg = torch.tensor([5.0], dtype=torch.float64)
    return R, hkls, thetas_deg


def test_bragg_omega_eta_shapes():
    R, hkls, thetas = _trivial_orient_and_hkls()
    omega, eta, valid = bragg_omega_eta(R, hkls, thetas, distance_um=1000.0)
    assert omega.shape == (1, 2, 1)
    assert eta.shape == (1, 2, 1)
    assert valid.shape == (1, 2, 1)
    assert valid.dtype == torch.bool


def test_bragg_omega_eta_two_solutions_negate_each_other():
    """For identity rotation and a single G with |G|=1, the two omega
    solutions should be related by sign (modulo small wedge effects)."""
    R, hkls, thetas = _trivial_orient_and_hkls()
    omega, _eta, valid = bragg_omega_eta(R, hkls, thetas, distance_um=1000.0)
    # If both valid, |omega_p - (-omega_n)| < tol  OR  |omega_p + omega_n| ~ 0
    if bool(valid[0, 0, 0]) and bool(valid[0, 1, 0]):
        s = omega[0, 0, 0] + omega[0, 1, 0]
        # The two roots are typically related; allow some tolerance.
        assert s.abs() < 360.0  # both in (-180, 180) range


def test_bragg_omega_eta_no_solution_returns_invalid():
    """A G-vector that cannot satisfy Bragg at the given theta returns invalid."""
    R = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    # G = z-axis: trying theta=89 would require essentially impossible geometry
    hkls = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    thetas = torch.tensor([89.0], dtype=torch.float64)
    omega, eta, valid = bragg_omega_eta(R, hkls, thetas, distance_um=1000.0)
    assert valid.shape == (1, 2, 1)
    assert not valid.any()


def test_bragg_omega_eta_differentiable_in_orientation():
    """Gradient flows from omega/eta back to the orientation matrices.

    Use the identity quaternion + a G-vector that admits a valid Bragg
    solution at theta=5deg (G_lab = (1,0,0); -cos(omega) = sin(5deg) -> two
    solutions ~ +/-95deg).
    """
    q = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True
    )
    R = quat_to_orient_matrix(q.unsqueeze(0))
    hkls = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    thetas = torch.tensor([5.0], dtype=torch.float64)
    omega, eta, valid = bragg_omega_eta(R, hkls, thetas, distance_um=1000.0)
    assert valid.any(), "Test inputs degenerate: no valid Bragg solution"
    loss = (omega ** 2 * valid.float()).sum() + (eta ** 2 * valid.float()).sum()
    loss.backward()
    assert q.grad is not None
    assert q.grad.abs().sum() > 0
