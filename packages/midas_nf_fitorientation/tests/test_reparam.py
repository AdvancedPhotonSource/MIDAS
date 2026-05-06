"""Tests for tanh-box reparameterisation, ΔLsd encoding, and quaternion
misorientation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.reparam import (
    LsdEncoding,
    TanhBox,
    misorientation_deg_symmetric,
    normalize_orient_mat,
    quaternion_from_euler_zxz,
)


def test_tanh_box_x_at_seed_when_u_zero():
    x0 = torch.tensor([1.0, 2.0, 3.0])
    box = TanhBox(x0, 0.5)
    assert torch.allclose(box.x, x0)
    # u is the autograd leaf; x is a derived tensor
    assert box.u.requires_grad
    assert box.u.grad is None


def test_tanh_box_respects_bounds_under_extreme_u():
    x0 = torch.tensor([0.0])
    tol = 1.5
    box = TanhBox(x0, tol)
    with torch.no_grad():
        box.u.copy_(torch.tensor([1e9]))
    # tanh(1e9) → 1, so x should hit the upper bound exactly.
    assert box.x.item() == pytest.approx(tol, abs=1e-12)
    with torch.no_grad():
        box.u.copy_(torch.tensor([-1e9]))
    assert box.x.item() == pytest.approx(-tol, abs=1e-12)


def test_tanh_box_gradient_flows_through_x():
    x0 = torch.tensor([0.0])
    box = TanhBox(x0, 1.0)
    y = (box.x ** 2).sum()
    y.backward()
    # dx/du = tol * sech^2(u); at u=0 sech^2(0)=1, so dy/du = 2*x*1 = 0.
    assert box.u.grad is not None


def test_tanh_box_perturb_changes_x():
    x0 = torch.tensor([0.0, 0.0])
    box = TanhBox(x0, 1.0)
    g = torch.Generator().manual_seed(123)
    box.perturb(0.5, generator=g)
    assert not torch.allclose(box.x, x0)
    # Reset returns to seed
    box.reset_to_seed()
    assert torch.allclose(box.x, x0)


def test_tikhonov_zero_at_seed():
    x0 = torch.tensor([1.0, 2.0])
    box = TanhBox(x0, 1.0)
    pen = box.tikhonov(sigma=1.0, lam=1.0)
    assert pen.item() == pytest.approx(0.0, abs=1e-12)


def test_tikhonov_quadratic_in_deviation():
    x0 = torch.tensor([0.0])
    box = TanhBox(x0, 10.0)
    with torch.no_grad():
        box.u.copy_(torch.tensor([math.atanh(0.4)]))  # x ≈ 4
    pen = box.tikhonov(sigma=2.0, lam=1.0)
    # ((4-0)/2)^2 = 4
    assert pen.item() == pytest.approx(4.0, rel=1e-6)


def test_lsd_encoding_round_trip():
    Lsds = torch.tensor([1000.0, 1500.0, 2200.0, 3000.0])
    enc = LsdEncoding.from_lsds(Lsds)
    assert enc.Lsd0.item() == pytest.approx(1000.0)
    assert torch.allclose(enc.deltas, torch.tensor([500.0, 700.0, 800.0]))
    decoded = enc.decode()
    assert torch.allclose(decoded, Lsds)


def test_lsd_encoding_single_layer():
    Lsds = torch.tensor([1234.0])
    enc = LsdEncoding.from_lsds(Lsds)
    assert enc.deltas.numel() == 0
    assert torch.allclose(enc.decode(), Lsds)


def test_quaternion_from_zero_euler_is_identity():
    eul = torch.tensor([0.0, 0.0, 0.0])
    q = quaternion_from_euler_zxz(eul)
    assert torch.allclose(q, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_misorientation_zero_for_same_orientation_cubic():
    """Identical Eulers → zero miso, regardless of space group."""
    eul = (0.5, 0.7, 0.9)
    miso = misorientation_deg_symmetric(eul, eul, space_group=225)  # FCC
    assert miso == pytest.approx(0.0, abs=1e-6)


def test_misorientation_cubic_collapses_90deg_rotation():
    """A 90° rotation about the z-axis is a cubic symmetry op
    (m-3m point group), so the miso between an orientation and its
    90°-z-rotated cousin must be zero under SG 225."""
    eul1 = [0.0, 0.0, 0.0]
    eul2 = [math.pi / 2, 0.0, 0.0]   # 90° about z (Bunge ZXZ)
    miso = misorientation_deg_symmetric(eul1, eul2, space_group=225)
    assert miso == pytest.approx(0.0, abs=1e-3)


def test_normalize_orient_mat_round_trip():
    """det^(-1/3) scaling brings det back to 1."""
    R = np.eye(3) * 1.05  # det = 1.05^3
    Rn = normalize_orient_mat(R)
    assert np.linalg.det(Rn) == pytest.approx(1.0, abs=1e-12)


def test_normalize_orient_mat_batched():
    """Batched (N, 3, 3) input is scaled per-matrix."""
    R = np.stack([np.eye(3) * 0.95, np.eye(3) * 1.10])
    Rn = normalize_orient_mat(R)
    dets = np.linalg.det(Rn)
    assert np.allclose(dets, 1.0, atol=1e-12)
