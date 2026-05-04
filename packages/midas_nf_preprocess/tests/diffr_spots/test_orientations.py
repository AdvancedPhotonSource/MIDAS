"""Tests for the orientations module: quaternion <-> matrix, Euler -> matrix."""

from __future__ import annotations

import math

import pytest
import torch

from midas_nf_preprocess.diffr_spots import (
    euler_to_orient_matrix,
    orient_matrix_to_quat,
    quat_to_orient_matrix,
)


# -----------------------------------------------------------------------------
# quat_to_orient_matrix: structural & known cases
# -----------------------------------------------------------------------------


def test_identity_quaternion_gives_identity_matrix():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    R = quat_to_orient_matrix(q)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64).unsqueeze(0))


def test_180_around_z_gives_diag_minus1_minus1_plus1():
    """q = (0, 0, 0, 1) is a 180-degree rotation about z."""
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    R = quat_to_orient_matrix(q)
    expected = torch.tensor(
        [[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    assert torch.allclose(R, expected, atol=1e-12)


def test_quat_matrix_orthonormal():
    """For unit quaternions, the resulting matrix is orthonormal."""
    q = torch.tensor(
        [[0.7071067811865476, 0.0, 0.7071067811865476, 0.0]], dtype=torch.float64
    )
    R = quat_to_orient_matrix(q).squeeze(0)
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.isclose(torch.det(R), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)


def test_quat_matrix_batched():
    """A batch of (N, 4) quaternions yields (N, 3, 3) matrices."""
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    R = quat_to_orient_matrix(q)
    assert R.shape == (4, 3, 3)


def test_quat_to_orient_wrong_shape_raises():
    with pytest.raises(ValueError, match="last dim = 4"):
        quat_to_orient_matrix(torch.zeros(3, 3))


# -----------------------------------------------------------------------------
# Quaternion roundtrip
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.7071067811865476, 0.7071067811865476, 0.0, 0.0]),
        torch.tensor([0.5, 0.5, 0.5, 0.5]),
        torch.tensor([0.6324555320336759, -0.31622776601683794, 0.6324555320336759, 0.31622776601683794]),
    ],
)
def test_quat_to_matrix_to_quat_roundtrip(q):
    q = q.to(torch.float64).unsqueeze(0)
    q = q / torch.norm(q, dim=-1, keepdim=True)
    R = quat_to_orient_matrix(q)
    q2 = orient_matrix_to_quat(R)
    # Check matrices match (quats may differ by global sign).
    R2 = quat_to_orient_matrix(q2)
    assert torch.allclose(R, R2, atol=1e-12)


# -----------------------------------------------------------------------------
# Euler ZXZ
# -----------------------------------------------------------------------------


def test_euler_zero_is_identity():
    e = torch.zeros((1, 3), dtype=torch.float64)
    R = euler_to_orient_matrix(e)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64).unsqueeze(0))


def test_euler_orthonormal():
    e = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float64)
    R = euler_to_orient_matrix(e).squeeze(0)
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.isclose(torch.det(R), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)


def test_euler_unsupported_convention_raises():
    with pytest.raises(NotImplementedError):
        euler_to_orient_matrix(torch.zeros((1, 3)), convention="XYZ")


# -----------------------------------------------------------------------------
# Differentiability
# -----------------------------------------------------------------------------


def test_quat_to_matrix_differentiable():
    q = torch.tensor([0.7, 0.1, 0.7, 0.1], dtype=torch.float64, requires_grad=True)
    R = quat_to_orient_matrix(q.unsqueeze(0))
    R.sum().backward()
    assert q.grad is not None
    assert q.grad.abs().sum() > 0


def test_euler_to_matrix_differentiable():
    e = torch.tensor([15.0, 25.0, 35.0], dtype=torch.float64, requires_grad=True)
    R = euler_to_orient_matrix(e.unsqueeze(0))
    R.sum().backward()
    assert e.grad is not None
    assert e.grad.abs().sum() > 0
