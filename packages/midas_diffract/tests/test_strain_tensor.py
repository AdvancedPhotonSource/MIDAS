"""Verify the direct 6-component strain-tensor input to HEDMForwardModel.

The model exposes ``strain=`` on both ``correct_hkls_latc`` and ``forward()``,
implementing the same Voigt-layout (I + eps)^{-1} B0 transformation as
``CorrectHKLsLatCEpsilon`` in
``FF_HEDM/src/ForwardSimulationCompressed.c:423``. This test verifies:

  1. ``strain=zeros(6)`` reproduces the no-strain path bitwise.
  2. For diagonal cubic strain, the strain= path is mathematically equivalent
     to a lattice-parameter perturbation [a(1+e11), b(1+e22), c(1+e33), ...],
     so both paths must produce identical hkls_cart and Bragg theta.
  3. Off-diagonal strain affects hkls_cart in a way that is differentiable
     and produces gradients that pass a finite-difference check.
"""
import math

import numpy as np
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry

DEG2RAD = math.pi / 180.0


def _make_model():
    # Small cubic-Au-like setup; geometry doesn't matter for the
    # correct_hkls_latc test, but is needed to construct the model.
    hkls_int = torch.tensor(
        [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1], [2, 2, 2]],
        dtype=torch.float64,
    )
    # Dummy nominal G + theta; correct_hkls_latc rebuilds these from
    # lattice_params anyway.
    g_dummy = torch.zeros((hkls_int.shape[0], 3), dtype=torch.float64)
    th_dummy = torch.zeros(hkls_int.shape[0], dtype=torch.float64)
    geom = HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048,
        min_eta=6.0, wavelength=0.172979,
    )
    return HEDMForwardModel(
        hkls=g_dummy, thetas=th_dummy, geometry=geom, hkls_int=hkls_int,
    )


def test_zero_strain_matches_no_strain():
    model = _make_model()
    latc = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
                         dtype=torch.float64)

    g_no, th_no = model.correct_hkls_latc(latc, strain=None)
    g_zero, th_zero = model.correct_hkls_latc(
        latc, strain=torch.zeros(6, dtype=torch.float64),
    )

    assert torch.allclose(g_no, g_zero, atol=1e-12), \
        f"strain=zeros should be identical to no-strain; max diff "\
        f"{(g_no - g_zero).abs().max().item():.2e}"
    assert torch.allclose(th_no, th_zero, atol=1e-12)


def test_diagonal_strain_matches_lattice_perturbation():
    """For diagonal cubic strain, eps = (e11, 0, 0, e22, 0, e33) and
    latc_strained = [a*(1+e11), a*(1+e22), a*(1+e33), 90, 90, 90] must
    produce the same B matrix and hence identical G-vectors / Bragg angles.
    """
    model = _make_model()
    a = 4.08
    e11, e22, e33 = 1.5e-3, -2.0e-3, 0.8e-3

    # Path A: lattice perturbation via real-space lattice constants
    latc_A = torch.tensor(
        [a * (1 + e11), a * (1 + e22), a * (1 + e33), 90.0, 90.0, 90.0],
        dtype=torch.float64,
    )
    g_A, th_A = model.correct_hkls_latc(latc_A, strain=None)

    # Path B: direct strain tensor on nominal lattice
    latc_B = torch.tensor([a, a, a, 90.0, 90.0, 90.0], dtype=torch.float64)
    eps = torch.tensor([e11, 0.0, 0.0, e22, 0.0, e33], dtype=torch.float64)
    g_B, th_B = model.correct_hkls_latc(latc_B, strain=eps)

    # Both must agree to floating-point precision.
    max_g = (g_A - g_B).abs().max().item()
    max_th = (th_A - th_B).abs().max().item()
    assert max_g < 1e-12, f"diagonal strain != lattice path on G: max diff {max_g:.2e}"
    assert max_th < 1e-12, f"diagonal strain != lattice path on theta: max diff {max_th:.2e}"


def test_offdiagonal_strain_changes_g_and_is_differentiable():
    model = _make_model()
    latc = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
                         dtype=torch.float64)
    eps_diag = torch.zeros(6, dtype=torch.float64)
    eps_offdiag = torch.tensor(
        [0.0, 1e-3, 5e-4, 0.0, 8e-4, 0.0], dtype=torch.float64,
    )

    g_zero, _ = model.correct_hkls_latc(latc, strain=eps_diag)
    g_off, _ = model.correct_hkls_latc(latc, strain=eps_offdiag)
    diff = (g_off - g_zero).abs().max().item()
    assert diff > 1e-6, \
        f"off-diagonal strain should produce a measurable G change; got {diff:.2e}"

    # Differentiability + gradient finite-difference check
    eps_var = eps_offdiag.clone().requires_grad_(True)
    g_var, _ = model.correct_hkls_latc(latc, strain=eps_var)
    loss = (g_var ** 2).sum()
    loss.backward()
    grad_auto = eps_var.grad.detach().clone()

    # Central-difference reference at h=1e-6
    h = 1e-6
    grad_fd = torch.zeros_like(eps_var)
    with torch.no_grad():
        for k in range(6):
            eps_p = eps_offdiag.clone()
            eps_m = eps_offdiag.clone()
            eps_p[k] += h
            eps_m[k] -= h
            g_p, _ = model.correct_hkls_latc(latc, strain=eps_p)
            g_m, _ = model.correct_hkls_latc(latc, strain=eps_m)
            grad_fd[k] = ((g_p ** 2).sum() - (g_m ** 2).sum()) / (2 * h)

    rel_err = (grad_auto - grad_fd).abs().max() / (grad_fd.abs().max() + 1e-12)
    assert rel_err < 1e-5, \
        f"autograd-vs-finite-difference relative error too large: {rel_err:.2e}"


def test_forward_pass_accepts_strain_kwarg():
    """End-to-end forward() must accept strain= and produce a SpotDescriptors
    that depends on the strain tensor (i.e. backprop through the full pass)."""
    model = _make_model()
    euler = torch.tensor(
        [[45.0 * DEG2RAD, 30.0 * DEG2RAD, 60.0 * DEG2RAD]],
        dtype=torch.float64,
    )
    pos = torch.zeros((1, 3), dtype=torch.float64)
    latc = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
                         dtype=torch.float64)
    eps = torch.tensor(
        [1e-3, 5e-4, 0.0, -8e-4, 2e-4, 7e-4], dtype=torch.float64,
    )
    eps.requires_grad_(True)

    spots = model(euler, pos, lattice_params=latc, strain=eps)
    coords, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
    loss = (coords[valid > 0.5] ** 2).sum()
    loss.backward()
    assert eps.grad is not None
    assert torch.isfinite(eps.grad).all()
    assert eps.grad.abs().max().item() > 0.0
