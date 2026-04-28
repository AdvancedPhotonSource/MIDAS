"""Tests for compute.rotation."""

import math

import numpy as np
import pytest
import torch

from midas_index.compute.rotation import axis_angle_batch, calc_rotation_angle


# --------------------------------------------------------------------------- calc_rotation_angle


@pytest.mark.parametrize("ring_nr,sg,hkl,expected", [
    (1, 225, (1, 1, 1), 120.0),     # Cubic, n_zeros=0 with h==k==l → 120
    (2, 225, (2, 0, 0), 90.0),      # Cubic, n_zeros=2 → 90
    (3, 225, (2, 2, 0), 180.0),     # Cubic, n_zeros=1, h==k → 180
    (4, 225, (3, 2, 1), 360.0),     # Cubic, all distinct → 360
    (1, 1, (1, 1, 1), 360.0),       # Triclinic
    (1, 16, (1, 0, 0), 180.0),      # Orthorhombic, n_zeros=2
    (1, 75, (0, 0, 1), 90.0),       # Tetragonal, n_zeros=2 with l!=0
    (1, 168, (1, 0, 0), 360.0),     # Hexagonal, n_zeros=2 but l==0
    (1, 168, (0, 0, 1), 60.0),      # Hexagonal, n_zeros=2 with l!=0
    (1, 225, (0, 0, 0), 0.0),       # Forbidden (all zero)
])
def test_calc_rotation_angle(ring_nr, sg, hkl, expected):
    assert calc_rotation_angle(ring_nr, sg, hkl) == pytest.approx(expected)


# --------------------------------------------------------------------------- axis_angle_batch


def test_axis_angle_batch_single_pair():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    angle = torch.tensor(90.0, dtype=torch.float64)
    R = axis_angle_batch(axis, angle)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    np.testing.assert_allclose(R.numpy(), expected, atol=1e-12)


def test_axis_angle_batch_batched():
    axes = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    angles = torch.tensor([90.0, 180.0], dtype=torch.float64)
    R = axis_angle_batch(axes, angles)
    assert R.shape == (2, 3, 3)
    np.testing.assert_allclose(R[0].numpy() @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R[1].numpy() @ np.array([0.0, 1.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-12)


def test_axis_angle_batch_zero_angle_returns_identity():
    axis = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    angle = torch.tensor(0.0, dtype=torch.float64)
    R = axis_angle_batch(axis, angle)
    np.testing.assert_allclose(R.numpy(), np.eye(3), atol=1e-12)
