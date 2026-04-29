"""Torch-backend parity tests for Phase E (elastic_inverse).

Covers the building-block functions used to compose Hill / Voigt / Reuss
stage matrices for elastic-constant fitting. The big iterative
`fit_single_crystal_stiffness` is left NumPy-only (scipy least-squares).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_stress.elastic_inverse import (
    build_stage_matrix,
    build_stage_matrix_reuss,
    build_stage_matrix_voigt,
    stiffness_from_cij,
    symmetry_parameterisation,
)


def _eq(a, b, atol=1e-12):
    a_np = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    b_np = np.asarray(b, dtype=np.float64)
    np.testing.assert_allclose(a_np, b_np, atol=atol)


# ---------------------------------------------------------------------------
# stiffness_from_cij
# ---------------------------------------------------------------------------


def test_stiffness_from_cij_torch_vector_matches_numpy():
    cij = np.array([168.4, 121.4, 75.4])
    np_C = stiffness_from_cij(cij, "cubic")
    t_C = stiffness_from_cij(torch.tensor(cij, dtype=torch.float64), "cubic")
    assert isinstance(t_C, torch.Tensor)
    _eq(t_C, np_C)


def test_stiffness_from_cij_torch_dict_matches_numpy():
    cij_np = {"C11": 168.4, "C12": 121.4, "C44": 75.4}
    cij_torch = {
        "C11": torch.tensor(168.4, dtype=torch.float64),
        "C12": torch.tensor(121.4, dtype=torch.float64),
        "C44": torch.tensor(75.4, dtype=torch.float64),
    }
    np_C = stiffness_from_cij(cij_np, "cubic")
    t_C = stiffness_from_cij(cij_torch, "cubic")
    assert isinstance(t_C, torch.Tensor)
    _eq(t_C, np_C)


def test_stiffness_from_cij_is_differentiable():
    cij = torch.tensor([168.4, 121.4, 75.4], dtype=torch.float64, requires_grad=True)
    C = stiffness_from_cij(cij, "cubic")
    C.sum().backward()
    assert cij.grad is not None
    assert torch.isfinite(cij.grad).all()


# ---------------------------------------------------------------------------
# build_stage_matrix (Hill)
# ---------------------------------------------------------------------------


def _toy_stage(rng=None):
    rng = rng or np.random.default_rng(0)
    n = 5
    orientations = np.tile(np.eye(3), (n, 1, 1))
    for i in range(n):
        a = (i + 1) * 0.05
        orientations[i] = np.array([[np.cos(a), -np.sin(a), 0],
                                    [np.sin(a),  np.cos(a), 0],
                                    [0, 0, 1]])
    strains = rng.uniform(-1e-3, 1e-3, (n, 3, 3))
    strains = (strains + np.transpose(strains, (0, 2, 1))) / 2
    weights = rng.uniform(0.5, 2.0, n)
    weights = weights / weights.sum()
    _, P = symmetry_parameterisation("cubic")
    return orientations, strains, weights, P


def test_build_stage_matrix_torch_matches_numpy():
    orient, strains, weights, P = _toy_stage()
    np_A = build_stage_matrix(orient, strains, weights, P)
    t_A = build_stage_matrix(
        torch.tensor(orient, dtype=torch.float64),
        torch.tensor(strains, dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(P, dtype=torch.float64),
    )
    assert isinstance(t_A, torch.Tensor)
    _eq(t_A, np_A, atol=1e-10)


def test_build_stage_matrix_voigt_torch_matches_numpy():
    orient, _, weights, P = _toy_stage()
    rng = np.random.default_rng(1)
    mean_strain = rng.uniform(-1e-3, 1e-3, (3, 3))
    mean_strain = (mean_strain + mean_strain.T) / 2
    np_A = build_stage_matrix_voigt(orient, mean_strain, weights, P)
    t_A = build_stage_matrix_voigt(
        torch.tensor(orient, dtype=torch.float64),
        torch.tensor(mean_strain, dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(P, dtype=torch.float64),
    )
    assert isinstance(t_A, torch.Tensor)
    _eq(t_A, np_A, atol=1e-10)


def test_build_stage_matrix_reuss_torch_matches_numpy():
    orient, _, weights, P = _toy_stage()
    rng = np.random.default_rng(2)
    applied = rng.uniform(-100, 100, (3, 3))
    applied = (applied + applied.T) / 2
    np_A = build_stage_matrix_reuss(orient, applied, weights, P)
    t_A = build_stage_matrix_reuss(
        torch.tensor(orient, dtype=torch.float64),
        torch.tensor(applied, dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(P, dtype=torch.float64),
    )
    assert isinstance(t_A, torch.Tensor)
    _eq(t_A, np_A, atol=1e-10)


def test_build_stage_matrix_is_differentiable():
    orient, strains, weights, P = _toy_stage()
    cij = torch.tensor([168.4, 121.4, 75.4], dtype=torch.float64, requires_grad=True)
    A = build_stage_matrix(
        torch.tensor(orient, dtype=torch.float64),
        torch.tensor(strains, dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(P, dtype=torch.float64),
    )
    sigma_lab = A @ cij    # (6,) — predicted volume-averaged stress
    sigma_lab.sum().backward()
    assert torch.isfinite(cij.grad).all()
