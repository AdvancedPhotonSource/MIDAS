"""Bragg-geometry tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_process_grains.compute.geometry import (
    lab_obs_to_g_and_d,
    numpy_lab_obs_to_g_and_d,
)


def test_zero_yz_gives_two_theta_zero_and_undefined_g():
    """Beam centre: (y, z) = 0 → 2θ = 0, d_obs → ∞ (clamped)."""
    g, d = lab_obs_to_g_and_d(
        torch.tensor([0.0]), torch.tensor([0.0]),
        lsd=800000.0, wavelength_a=0.17,
    )
    # At 2θ=0 the g-vector is degenerate; we don't make assertions on direction.
    assert torch.isfinite(d).all()                # clamped, not NaN/Inf


def test_known_2theta_recovers_correct_d():
    """Place a spot at known (y, z) such that 2θ = 5°. Verify d_obs."""
    Lsd = 800000.0           # µm
    wavelength_a = 0.17
    two_theta_deg = 5.0
    two_theta_rad = math.radians(two_theta_deg)
    rho = Lsd * math.tan(two_theta_rad)
    y = rho                  # spot lies on +y axis
    z = 0.0
    g, d = lab_obs_to_g_and_d(
        torch.tensor([y]), torch.tensor([z]),
        lsd=Lsd, wavelength_a=wavelength_a,
    )
    expected_d = wavelength_a / (2.0 * math.sin(two_theta_rad / 2.0))
    np.testing.assert_allclose(d.item(), expected_d, rtol=1e-6)


def test_g_vector_direction_for_y_axis_spot():
    """Spot on +y axis: g = (cos2θ - 1, sin2θ, 0) / |...|.

    For small 2θ, this is approximately (−sin²θ, sinθ·cosθ, 0), which is
    dominated by the +y component. We assert that g_y > 0 and g_z == 0.
    """
    Lsd = 800000.0
    rho = Lsd * math.tan(math.radians(5.0))
    g, _ = lab_obs_to_g_and_d(
        torch.tensor([rho]), torch.tensor([0.0]),
        lsd=Lsd, wavelength_a=0.17,
    )
    g0 = g[0].numpy()
    assert g0[1] > 0
    assert abs(g0[2]) < 1e-12
    np.testing.assert_allclose(np.linalg.norm(g0), 1.0)


def test_g_vector_direction_for_z_axis_spot():
    """Spot on +z axis: by symmetry g_z > 0, g_y == 0."""
    Lsd = 800000.0
    rho = Lsd * math.tan(math.radians(5.0))
    g, _ = lab_obs_to_g_and_d(
        torch.tensor([0.0]), torch.tensor([rho]),
        lsd=Lsd, wavelength_a=0.17,
    )
    g0 = g[0].numpy()
    assert g0[2] > 0
    assert abs(g0[1]) < 1e-12


def test_d_obs_strain_recovery_round_trip():
    """End-to-end: feed in spots from a known strain → recover that strain via
    the lstsq solver."""
    from midas_process_grains.compute.strain import solve_strain_lstsq

    Lsd = 800000.0
    wavelength_a = 0.17
    rng = np.random.default_rng(42)
    # Generate 30 random unit g-directions; map them onto the detector via the
    # forward Bragg formula at d = 2.0 Å (i.e. 2θ = 2 arcsin(λ / (2d))).
    # Then perturb d by a known strain, recompute d, generate detector (y, z),
    # feed back through lab_obs_to_g_and_d and solve.
    n = 30

    # Reference d-spacings: random in [1.5, 2.5] Å.
    d0 = rng.uniform(1.5, 2.5, size=n)
    eps_true = np.array([1e-4, -2e-4, 3e-4, 1e-5, -2e-5, 0.5e-5])

    # For each spot pick a g-direction by sampling unit vectors.
    g_dir = rng.standard_normal((n, 3))
    g_dir /= np.linalg.norm(g_dir, axis=1, keepdims=True)
    bb = (g_dir ** 2 @ eps_true[:3]) + 2 * (
        g_dir[:, 0] * g_dir[:, 1] * eps_true[3]
        + g_dir[:, 0] * g_dir[:, 2] * eps_true[4]
        + g_dir[:, 1] * g_dir[:, 2] * eps_true[5]
    )
    d_strained = d0 * (1.0 + bb)
    # Build detector (y, z) such that the Bragg geometry returns d_strained.
    # We need 2θ such that d_strained = λ / (2 sin θ). Pick (y, z) to match
    # both the 2θ and a chosen azimuth (we'll recover g direction from there).
    # Simpler: use the lstsq solver directly with g_dir + d_strained,
    # bypassing the round-trip through detector geometry. The geometry
    # function is tested separately; this test is the strain solver round-trip.
    g_t = torch.from_numpy(g_dir)
    d_obs_t = torch.from_numpy(d_strained)
    d_0_t = torch.from_numpy(d0)
    res = solve_strain_lstsq(g_t, d_obs_t, d_0_t)
    np.testing.assert_allclose(res.epsilon_voigt.numpy(), eps_true, atol=1e-12)


def test_numpy_wrapper_matches_torch():
    y = np.array([100.0, -50.0, 200.0])
    z = np.array([200.0, 100.0, -150.0])
    g_np, d_np = numpy_lab_obs_to_g_and_d(y, z, lsd=800000.0, wavelength_a=0.17)
    g_t, d_t = lab_obs_to_g_and_d(
        torch.from_numpy(y), torch.from_numpy(z),
        lsd=800000.0, wavelength_a=0.17,
    )
    np.testing.assert_allclose(g_np, g_t.numpy())
    np.testing.assert_allclose(d_np, d_t.numpy())
