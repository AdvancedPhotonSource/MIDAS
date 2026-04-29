"""Tests for compute.orientation_grid."""

import numpy as np
import torch

from midas_index.compute.orientation_grid import (
    generate_candidate_orientations,
    generate_candidate_orientations_batched,
)


def test_generate_candidate_orientations_cubic_111():
    hkl = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    plane_normal = torch.tensor([0.5, 0.7, 0.3], dtype=torch.float64)
    plane_normal = plane_normal / torch.linalg.vector_norm(plane_normal)

    R = generate_candidate_orientations(
        hkl, plane_normal, stepsize_orient_deg=10.0,
        ring_nr=1, space_group=225, hkl_int=(1, 1, 1),
    )
    # Cubic h==k==l → 120 deg sweep, 12 steps
    assert R.shape == (12, 3, 3)
    # Each is a valid rotation
    for i in range(R.shape[0]):
        np.testing.assert_allclose(
            (R[i] @ R[i].T).numpy(), np.eye(3), atol=1e-9
        )
        np.testing.assert_allclose(torch.det(R[i]).item(), 1.0, atol=1e-9)


def test_generate_candidate_orientations_first_step_aligns_hkl_with_normal():
    hkl = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    R = generate_candidate_orientations(
        hkl, plane_normal, stepsize_orient_deg=30.0,
        ring_nr=1, space_group=225, hkl_int=(2, 0, 0),
    )
    # Step 0: pure pre-rotation — R @ hkl/|hkl| should align with plane_normal
    aligned = (R[0] @ hkl).numpy()
    aligned = aligned / np.linalg.norm(aligned)
    np.testing.assert_allclose(aligned, plane_normal.numpy(), atol=1e-9)


def test_generate_candidate_orientations_zero_step_returns_empty():
    """If MaxAngle is 0 (forbidden HKL), returns empty tensor."""
    hkl = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    R = generate_candidate_orientations(
        hkl, plane_normal, stepsize_orient_deg=10.0,
        ring_nr=1, space_group=225, hkl_int=(0, 0, 0),
    )
    assert R.shape == (0, 3, 3)


def test_generate_candidate_orientations_batched_matches_scalar():
    """Batched variant must produce bit-identical Rs as the scalar variant
    when called per-candidate. This is the parity contract that lets the
    pipeline replace the per-(y0, z0) inner loop without changing results."""
    hkl = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    rng = np.random.default_rng(42)
    plane_normals = torch.tensor(
        rng.uniform(-1.0, 1.0, (5, 3)), dtype=torch.float64
    )
    plane_normals = plane_normals / torch.linalg.vector_norm(
        plane_normals, dim=-1, keepdim=True
    )

    Rs_batched = generate_candidate_orientations_batched(
        hkl, plane_normals, stepsize_orient_deg=10.0,
        ring_nr=1, space_group=225, hkl_int=(1, 1, 1),
    )
    assert Rs_batched.shape == (5, 12, 3, 3)
    for b in range(5):
        Rs_scalar = generate_candidate_orientations(
            hkl, plane_normals[b], stepsize_orient_deg=10.0,
            ring_nr=1, space_group=225, hkl_int=(1, 1, 1),
        )
        np.testing.assert_allclose(
            Rs_batched[b].numpy(), Rs_scalar.numpy(), atol=1e-12
        )


def test_generate_candidate_orientations_batched_zero_steps():
    """Forbidden HKL → (B, 0, 3, 3) for any B."""
    hkl = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    plane_normals = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    R = generate_candidate_orientations_batched(
        hkl, plane_normals, stepsize_orient_deg=10.0,
        ring_nr=1, space_group=225, hkl_int=(0, 0, 0),
    )
    assert R.shape == (2, 0, 3, 3)
