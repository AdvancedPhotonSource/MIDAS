"""Differentiability + correctness tests for the tilt+distortion transform."""

import math

import numpy as np
import pytest
import torch

from midas_transforms.fit_setup.transform import (
    apply_tilt_distortion, calc_eta_angle_local, correct_wedge_no_op,
)


def _default_params(device, dtype):
    return dict(
        Lsd=torch.tensor(1_000_000.0, device=device, dtype=dtype, requires_grad=True),
        BC_y=torch.tensor(1024.0, device=device, dtype=dtype, requires_grad=True),
        BC_z=torch.tensor(1024.0, device=device, dtype=dtype, requires_grad=True),
        tx=torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True),
        ty=torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True),
        tz=torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True),
        p_coeffs=torch.zeros(15, device=device, dtype=dtype, requires_grad=True),
        px=torch.tensor(200.0, device=device, dtype=dtype),
        rho_d=torch.tensor(1500.0, device=device, dtype=dtype),
    )


def test_apply_tilt_distortion_no_tilt_no_distortion_recovers_input():
    """With zero tilts and zero distortion, the transform should reproduce the
    raw lab-frame coordinates from pixel inputs (via the standard
    Yc=-(Y-BC)*px, Zc=(Z-BC)*px formula)."""
    dev = torch.device("cpu")
    dt = torch.float64
    params = _default_params(dev, dt)
    # 5 sample pixels along a known ring.
    Y_pix = torch.tensor([800.0, 900.0, 1000.0, 1100.0, 1200.0], device=dev, dtype=dt)
    Z_pix = torch.tensor([1024.0, 1024.0, 1024.0, 1024.0, 1024.0], device=dev, dtype=dt)
    Y_lab, Z_lab = apply_tilt_distortion(Y_pix, Z_pix, **params)
    # With zero distortion / tilts, R = sqrt(Yc² + Zc²); Eta = atan2(-Yc, Zc)
    # Yc = -(Y-BC)*px = -(800-1024)*200 = 44800, etc.
    expected_Yc = -(Y_pix - params["BC_y"]) * params["px"]
    expected_Zc = (Z_pix - params["BC_z"]) * params["px"]
    expected_R = torch.sqrt(expected_Yc * expected_Yc + expected_Zc * expected_Zc)
    np.testing.assert_allclose(
        torch.sqrt(Y_lab ** 2 + Z_lab ** 2).detach().numpy(),
        expected_R.detach().numpy(),
        rtol=1e-10, atol=1e-6,
    )


def test_apply_tilt_distortion_grad_flows_to_all_geometry_params():
    """Backward should produce non-zero gradients on every active geometry leaf.

    With zero tilts AND zero distortion, the projection collapses to pure
    pixel-to-µm scaling and Lsd has no effect on the output (analytic). To
    exercise the full surface, we set non-zero ty/tz and a non-zero p2.
    """
    dev = torch.device("cpu")
    dt = torch.float64
    params = _default_params(dev, dt)
    # Bump tilts off zero so Lsd matters.
    params["ty"] = torch.tensor(0.1, device=dev, dtype=dt, requires_grad=True)
    params["tz"] = torch.tensor(0.05, device=dev, dtype=dt, requires_grad=True)
    # Non-trivial distortion polynomial.
    pc = torch.zeros(15, device=dev, dtype=dt)
    pc[2] = 0.001  # p2 — radial term, multiplicative on Rad
    params["p_coeffs"] = pc.clone().detach().requires_grad_(True)

    Y_pix = torch.tensor([800.0, 1100.0, 1200.0], device=dev, dtype=dt)
    Z_pix = torch.tensor([900.0, 1024.0, 1100.0], device=dev, dtype=dt)
    Y_lab, Z_lab = apply_tilt_distortion(Y_pix, Z_pix, **params)
    loss = (Y_lab.sum() + Z_lab.sum())
    loss.backward()
    for name in ("Lsd", "BC_y", "BC_z", "ty", "tz"):
        g = params[name].grad
        assert g is not None and g.abs().item() > 0, f"{name} grad must be non-zero"
    # p_coeffs as a vector — at least one component should be non-zero (p2 is active).
    assert params["p_coeffs"].grad is not None
    assert params["p_coeffs"].grad.abs().sum().item() > 0


def test_calc_eta_angle_local_quadrants():
    # y > 0, z > 0  → eta < 0 (top-right quadrant): C convention flips the sign
    y = torch.tensor([1.0, -1.0, 0.0, 0.0])
    z = torch.tensor([0.0, 0.0, 1.0, -1.0])
    eta = calc_eta_angle_local(y, z).numpy()
    # y>0, z=0  →  acos(0) = 90°, with sign flip → -90°
    assert abs(eta[0] - (-90.0)) < 1e-6
    # y<0, z=0  →  acos(0) = 90°, no flip → +90°
    assert abs(eta[1] - 90.0) < 1e-6
    # y=0, z>0  →  acos(1) = 0
    assert abs(eta[2] - 0.0) < 1e-6
    # y=0, z<0  →  acos(-1) = 180°
    assert abs(eta[3] - 180.0) < 1e-6


def test_correct_wedge_no_op_returns_identity():
    dev = torch.device("cpu")
    dt = torch.float64
    Y = torch.tensor([1000.0, -2000.0], dtype=dt)
    Z = torch.tensor([3000.0, -1000.0], dtype=dt)
    Lsd = torch.tensor(1_000_000.0, dtype=dt)
    om = torch.tensor([10.0, 20.0], dtype=dt)
    y_o, z_o, om_o, eta_o, tth_o = correct_wedge_no_op(Y, Z, Lsd, om)
    np.testing.assert_array_equal(y_o, Y)
    np.testing.assert_array_equal(z_o, Z)
    np.testing.assert_array_equal(om_o, om)
    # eta and tth should be derived
    assert eta_o.shape == Y.shape
    assert tth_o.shape == Y.shape
