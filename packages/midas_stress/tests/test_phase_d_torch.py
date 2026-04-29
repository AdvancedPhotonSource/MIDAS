"""Torch-backend parity tests for Phase D modules:
materials.py, equilibrium.py (core constraints), plasticity.py.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_stress.equilibrium import (
    hydrostatic_deviatoric_decomposition,
    volume_average_stress_constraint,
)
from midas_stress.materials import (
    cubic_stiffness,
    get_stiffness,
    hexagonal_stiffness,
)
from midas_stress.plasticity import (
    resolved_shear_stress,
    schmid_factor,
    slip_systems_to_lab,
)


def _eq(a, b, atol=1e-12):
    a_np = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    b_np = np.asarray(b, dtype=np.float64)
    np.testing.assert_allclose(a_np, b_np, atol=atol)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------


def test_cubic_stiffness_torch_matches_numpy():
    np_C = cubic_stiffness(168.4, 121.4, 75.4)
    t_C = cubic_stiffness(
        torch.tensor(168.4, dtype=torch.float64),
        torch.tensor(121.4, dtype=torch.float64),
        torch.tensor(75.4, dtype=torch.float64),
    )
    assert isinstance(t_C, torch.Tensor)
    _eq(t_C, np_C)


def test_cubic_stiffness_explicit_torch_dtype():
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=torch.float64)
    assert isinstance(C, torch.Tensor)
    assert C.dtype == torch.float64
    assert C.shape == (6, 6)


def test_hexagonal_stiffness_torch_matches_numpy():
    args = (162.4, 92.0, 69.0, 180.7, 46.7)
    np_C = hexagonal_stiffness(*args)
    t_C = hexagonal_stiffness(*args, dtype=torch.float64)
    assert isinstance(t_C, torch.Tensor)
    _eq(t_C, np_C)


def test_get_stiffness_torch_dtype():
    C = get_stiffness("Cu", dtype=torch.float64)
    assert isinstance(C, torch.Tensor)
    np_C = get_stiffness("Cu")
    _eq(C, np_C)


# ---------------------------------------------------------------------------
# equilibrium — core constraints
# ---------------------------------------------------------------------------


def test_volume_average_stress_constraint_torch_matches_numpy():
    rng = np.random.default_rng(0)
    stresses = rng.uniform(-100, 100, (5, 3, 3))
    stresses = (stresses + np.transpose(stresses, (0, 2, 1))) / 2
    volumes = rng.uniform(0.5, 2.0, 5)

    np_out = volume_average_stress_constraint(stresses, volumes)
    t_out = volume_average_stress_constraint(
        torch.tensor(stresses, dtype=torch.float64),
        torch.tensor(volumes, dtype=torch.float64),
    )
    assert isinstance(t_out, torch.Tensor)
    _eq(t_out, np_out, atol=1e-10)


def test_volume_average_voigt_input_torch():
    rng = np.random.default_rng(0)
    voigt = rng.uniform(-100, 100, (5, 6))
    volumes = rng.uniform(0.5, 2.0, 5)
    np_out = volume_average_stress_constraint(voigt, volumes)
    t_out = volume_average_stress_constraint(
        torch.tensor(voigt, dtype=torch.float64),
        torch.tensor(volumes, dtype=torch.float64),
    )
    assert t_out.shape == (5, 6)
    _eq(t_out, np_out, atol=1e-10)


def test_hydrostatic_deviatoric_decomposition_torch_matches_numpy():
    rng = np.random.default_rng(0)
    stresses = rng.uniform(-100, 100, (5, 3, 3))
    stresses = (stresses + np.transpose(stresses, (0, 2, 1))) / 2
    volumes = rng.uniform(0.5, 2.0, 5)

    np_h, np_d, np_c = hydrostatic_deviatoric_decomposition(stresses, volumes)
    t_h, t_d, t_c = hydrostatic_deviatoric_decomposition(
        torch.tensor(stresses, dtype=torch.float64),
        torch.tensor(volumes, dtype=torch.float64),
    )
    assert isinstance(t_h, torch.Tensor)
    _eq(t_h, np_h, atol=1e-10)
    _eq(t_d, np_d, atol=1e-10)
    _eq(t_c, np_c, atol=1e-10)


# ---------------------------------------------------------------------------
# plasticity
# ---------------------------------------------------------------------------


def _toy_orient_and_systems():
    rng = np.random.default_rng(0)
    n = 4
    orient = np.tile(np.eye(3), (n, 1, 1))
    # Simulate a slight rotation per grain
    for i in range(n):
        a = (i + 1) * 0.1
        orient[i] = np.array([[np.cos(a), -np.sin(a), 0],
                              [np.sin(a),  np.cos(a), 0],
                              [0, 0, 1]])
    # FCC octahedral: 4 planes × 3 directions
    n_crystal = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]],
                         dtype=np.float64)
    n_crystal = n_crystal / np.linalg.norm(n_crystal, axis=-1, keepdims=True)
    b_crystal = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1], [1, 1, 0]],
                         dtype=np.float64)
    b_crystal = b_crystal / np.linalg.norm(b_crystal, axis=-1, keepdims=True)
    return orient, n_crystal, b_crystal


def test_slip_systems_to_lab_torch_matches_numpy():
    orient, n_crystal, b_crystal = _toy_orient_and_systems()
    np_n, np_b = slip_systems_to_lab(orient, n_crystal, b_crystal)
    t_n, t_b = slip_systems_to_lab(
        torch.tensor(orient, dtype=torch.float64),
        torch.tensor(n_crystal, dtype=torch.float64),
        torch.tensor(b_crystal, dtype=torch.float64),
    )
    assert isinstance(t_n, torch.Tensor)
    _eq(t_n, np_n)
    _eq(t_b, np_b)


def test_schmid_factor_torch_matches_numpy():
    orient, n_crystal, b_crystal = _toy_orient_and_systems()
    load_dir = np.array([0.0, 0.0, 1.0])
    np_m = schmid_factor(orient, load_dir, n_crystal, b_crystal)
    t_m = schmid_factor(
        torch.tensor(orient, dtype=torch.float64),
        torch.tensor(load_dir, dtype=torch.float64),
        torch.tensor(n_crystal, dtype=torch.float64),
        torch.tensor(b_crystal, dtype=torch.float64),
    )
    assert isinstance(t_m, torch.Tensor)
    _eq(t_m, np_m, atol=1e-12)


def test_resolved_shear_stress_torch_matches_numpy():
    orient, n_crystal, b_crystal = _toy_orient_and_systems()
    rng = np.random.default_rng(1)
    stress = rng.uniform(-100, 100, (orient.shape[0], 3, 3))
    stress = (stress + np.transpose(stress, (0, 2, 1))) / 2
    np_tau = resolved_shear_stress(stress, orient, n_crystal, b_crystal)
    t_tau = resolved_shear_stress(
        torch.tensor(stress, dtype=torch.float64),
        torch.tensor(orient, dtype=torch.float64),
        torch.tensor(n_crystal, dtype=torch.float64),
        torch.tensor(b_crystal, dtype=torch.float64),
    )
    assert isinstance(t_tau, torch.Tensor)
    _eq(t_tau, np_tau, atol=1e-10)


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------


def test_schmid_factor_is_differentiable():
    orient, n_crystal, b_crystal = _toy_orient_and_systems()
    load = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    m = schmid_factor(
        torch.tensor(orient, dtype=torch.float64),
        load,
        torch.tensor(n_crystal, dtype=torch.float64),
        torch.tensor(b_crystal, dtype=torch.float64),
    )
    m.sum().backward()
    assert load.grad is not None
    assert torch.isfinite(load.grad).all()
