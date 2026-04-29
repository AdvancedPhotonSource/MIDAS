"""Torch-backend tests for midas_stress.tensor.

Verifies torch path agrees with NumPy path for the full tensor module
(Voigt-Mandel conversion, A-matrix, lattice strain, frame transforms,
6x6 Mandel rotation, scalar invariants).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_stress.tensor import (
    deviatoric,
    hydrostatic,
    lattice_params_to_A_matrix,
    lattice_params_to_strain,
    rotation_voigt_mandel,
    strain_grain_to_lab,
    strain_lab_to_grain,
    tensor_to_voigt,
    tensor_to_voigt_engineering,
    voigt_to_tensor,
    von_mises,
)


def _eq(a, b, atol=1e-12):
    a_np = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    b_np = np.asarray(b, dtype=np.float64)
    np.testing.assert_allclose(a_np, b_np, atol=atol)


def _sym(rng, *shape):
    M = rng.uniform(-1.0, 1.0, shape + (3, 3))
    return (M + np.swapaxes(M, -1, -2)) / 2


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Voigt-Mandel
# ---------------------------------------------------------------------------


def test_tensor_to_voigt_torch_matches_numpy(rng):
    T = _sym(rng, 4)
    np_v = tensor_to_voigt(T)
    t_v = tensor_to_voigt(torch.tensor(T, dtype=torch.float64))
    assert isinstance(t_v, torch.Tensor)
    _eq(t_v, np_v)


def test_voigt_to_tensor_torch_matches_numpy(rng):
    v = rng.uniform(-1, 1, (3, 6))
    np_T = voigt_to_tensor(v)
    t_T = voigt_to_tensor(torch.tensor(v, dtype=torch.float64))
    assert isinstance(t_T, torch.Tensor)
    _eq(t_T, np_T)


def test_voigt_roundtrip_torch_preserves_frobenius_norm(rng):
    T = _sym(rng, 5)
    Tt = torch.tensor(T, dtype=torch.float64)
    v = tensor_to_voigt(Tt)
    fro = torch.linalg.norm(Tt.reshape(5, 9), dim=-1)
    l2 = torch.linalg.norm(v, dim=-1)
    np.testing.assert_allclose(fro.numpy(), l2.numpy(), atol=1e-12)


def test_tensor_to_voigt_engineering_torch_matches_numpy(rng):
    T = _sym(rng, 2)
    _eq(
        tensor_to_voigt_engineering(torch.tensor(T, dtype=torch.float64)),
        tensor_to_voigt_engineering(T),
    )


# ---------------------------------------------------------------------------
# A-matrix and lattice strain
# ---------------------------------------------------------------------------


def test_lattice_params_to_A_matrix_torch_matches_numpy():
    latc = np.array([4.08, 4.08, 4.08, 90.0, 90.0, 90.0])
    np_A = lattice_params_to_A_matrix(latc)
    t_A = lattice_params_to_A_matrix(torch.tensor(latc, dtype=torch.float64))
    assert isinstance(t_A, torch.Tensor)
    _eq(t_A, np_A)


def test_lattice_params_to_A_matrix_batched_torch():
    latc = torch.tensor([
        [4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
        [3.6, 3.6, 5.4, 90.0, 90.0, 120.0],
    ], dtype=torch.float64)
    A = lattice_params_to_A_matrix(latc)
    assert A.shape == (2, 3, 3)
    # cubic A should be 4.08 * I (with the convention used)
    _eq(A[0], 4.08 * np.eye(3))


def test_lattice_params_to_strain_torch_matches_numpy():
    latc0 = np.array([4.08, 4.08, 4.08, 90.0, 90.0, 90.0])
    latc1 = np.array([4.0808, 4.0801, 4.0810, 89.9999, 90.0002, 90.0001])
    np_e = lattice_params_to_strain(latc1, latc0)
    t_e = lattice_params_to_strain(
        torch.tensor(latc1, dtype=torch.float64),
        torch.tensor(latc0, dtype=torch.float64),
    )
    assert isinstance(t_e, torch.Tensor)
    _eq(t_e, np_e, atol=1e-10)


def test_lattice_params_to_strain_zero_at_unstrained():
    latc = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0], dtype=torch.float64)
    eps = lattice_params_to_strain(latc, latc)
    _eq(eps, np.zeros((3, 3)), atol=1e-12)


# ---------------------------------------------------------------------------
# Frame transforms (similarity)
# ---------------------------------------------------------------------------


def test_strain_grain_to_lab_torch_matches_numpy(rng):
    eps_g = _sym(rng, 3)
    U = np.eye(3)
    np_lab = strain_grain_to_lab(eps_g, U)
    t_lab = strain_grain_to_lab(
        torch.tensor(eps_g, dtype=torch.float64),
        torch.tensor(U, dtype=torch.float64),
    )
    _eq(t_lab, np_lab)


def test_strain_roundtrip_torch(rng):
    eps_g = _sym(rng, 4)
    U = np.eye(3)
    eps_g_t = torch.tensor(eps_g, dtype=torch.float64)
    U_t = torch.tensor(U, dtype=torch.float64)
    eps_back = strain_lab_to_grain(strain_grain_to_lab(eps_g_t, U_t), U_t)
    _eq(eps_back, eps_g, atol=1e-12)


# ---------------------------------------------------------------------------
# 6x6 Mandel rotation
# ---------------------------------------------------------------------------


def test_rotation_voigt_mandel_torch_matches_numpy(rng):
    # Random rotation matrix via QR
    M = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    np_R6 = rotation_voigt_mandel(Q)
    t_R6 = rotation_voigt_mandel(torch.tensor(Q, dtype=torch.float64))
    assert isinstance(t_R6, torch.Tensor)
    _eq(t_R6, np_R6, atol=1e-12)


# ---------------------------------------------------------------------------
# Scalar invariants
# ---------------------------------------------------------------------------


def test_hydrostatic_torch_matches_numpy(rng):
    T = _sym(rng, 5)
    _eq(
        hydrostatic(torch.tensor(T, dtype=torch.float64)),
        hydrostatic(T),
    )


def test_deviatoric_torch_matches_numpy(rng):
    T = _sym(rng, 4)
    _eq(
        deviatoric(torch.tensor(T, dtype=torch.float64)),
        deviatoric(T),
    )


def test_von_mises_torch_matches_numpy(rng):
    T = _sym(rng, 4)
    _eq(
        von_mises(torch.tensor(T, dtype=torch.float64)),
        von_mises(T),
    )


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------


def test_lattice_params_to_strain_is_differentiable():
    latc1 = torch.tensor([4.0808, 4.08, 4.08, 90.0, 90.0, 90.0],
                         dtype=torch.float64, requires_grad=True)
    latc0 = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
                         dtype=torch.float64)
    eps = lattice_params_to_strain(latc1, latc0)
    eps.sum().backward()
    assert latc1.grad is not None
    assert torch.isfinite(latc1.grad).all()


def test_von_mises_is_differentiable(rng):
    T = torch.tensor(_sym(rng, 1), dtype=torch.float64, requires_grad=True)
    vm = von_mises(T)
    vm.sum().backward()
    assert torch.isfinite(T.grad).all()
