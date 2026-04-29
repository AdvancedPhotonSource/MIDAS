"""Torch-backend tests for midas_stress.frames.

Verifies that:
  1. Torch.Tensor inputs return torch.Tensor outputs.
  2. Torch path agrees numerically with the NumPy path to float64 EPS.
  3. Outputs preserve input dtype/device.
  4. autograd flows through.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_stress.frames import (
    R_APS_TO_MIDAS,
    R_MIDAS_TO_APS,
    grains_midas_to_sample,
    lab_to_sample_rotation,
    orient_aps_to_midas,
    orient_midas_to_aps,
    tensor_aps_to_midas,
    tensor_lab_to_sample,
    tensor_midas_to_aps,
    vector_aps_to_midas,
    vector_midas_to_aps,
)


def _eq(a, b, atol=1e-12):
    a_np = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    b_np = np.asarray(b, dtype=np.float64)
    np.testing.assert_allclose(a_np, b_np, atol=atol)


# ---------------------------------------------------------------------------
# lab_to_sample_rotation
# ---------------------------------------------------------------------------


def test_lab_to_sample_rotation_torch_matches_numpy_midas():
    omega = 30.0
    np_R = lab_to_sample_rotation(omega, "midas")
    t_R = lab_to_sample_rotation(torch.tensor(omega, dtype=torch.float64), "midas")
    assert isinstance(t_R, torch.Tensor)
    assert t_R.shape == (3, 3)
    _eq(t_R, np_R)


def test_lab_to_sample_rotation_torch_matches_numpy_aps():
    omega = 47.5
    np_R = lab_to_sample_rotation(omega, "aps")
    t_R = lab_to_sample_rotation(torch.tensor(omega, dtype=torch.float64), "aps")
    _eq(t_R, np_R)


def test_lab_to_sample_rotation_torch_zero_omega_is_identity():
    R = lab_to_sample_rotation(torch.tensor(0.0, dtype=torch.float64), "midas")
    _eq(R, np.eye(3))


def test_lab_to_sample_rotation_torch_unknown_frame_raises():
    with pytest.raises(ValueError, match="Unknown frame"):
        lab_to_sample_rotation(torch.tensor(0.0, dtype=torch.float64), "bogus")


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------


def test_vector_midas_to_aps_torch_matches_numpy():
    v = np.array([1.0, 2.0, 3.0])
    np_out = vector_midas_to_aps(v)
    t_out = vector_midas_to_aps(torch.tensor(v, dtype=torch.float64))
    assert isinstance(t_out, torch.Tensor)
    _eq(t_out, np_out)


def test_vector_aps_to_midas_torch_matches_numpy():
    v = np.array([4.0, -2.0, 0.5])
    np_out = vector_aps_to_midas(v)
    t_out = vector_aps_to_midas(torch.tensor(v, dtype=torch.float64))
    _eq(t_out, np_out)


def test_vector_midas_to_aps_torch_batched():
    v = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    out = vector_midas_to_aps(v)
    assert out.shape == (2, 3)
    # MIDAS->APS: (Y, Z, X)
    _eq(out, [[2.0, 3.0, 1.0], [5.0, 6.0, 4.0]])


def test_vector_roundtrip_torch():
    v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    v_back = vector_aps_to_midas(vector_midas_to_aps(v))
    _eq(v_back, v.numpy())


# ---------------------------------------------------------------------------
# orientation matrices
# ---------------------------------------------------------------------------


def test_orient_midas_to_aps_torch_matches_numpy():
    U = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    np_U = orient_midas_to_aps(U)
    t_U = orient_midas_to_aps(torch.tensor(U, dtype=torch.float64))
    assert isinstance(t_U, torch.Tensor)
    _eq(t_U, np_U)


def test_orient_aps_to_midas_torch_matches_numpy():
    U = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    np_U = orient_aps_to_midas(U)
    t_U = orient_aps_to_midas(torch.tensor(U, dtype=torch.float64))
    _eq(t_U, np_U)


# ---------------------------------------------------------------------------
# tensors (similarity transform)
# ---------------------------------------------------------------------------


def test_tensor_midas_to_aps_torch_matches_numpy():
    T = np.array([[1.0, 0.5, 0.3], [0.5, 2.0, 0.1], [0.3, 0.1, 1.5]])
    _eq(
        tensor_midas_to_aps(torch.tensor(T, dtype=torch.float64)),
        tensor_midas_to_aps(T),
    )


def test_tensor_lab_to_sample_torch_matches_numpy():
    T = np.array([[1.0, 0.5, 0.3], [0.5, 2.0, 0.1], [0.3, 0.1, 1.5]])
    omega = 25.0
    np_out = tensor_lab_to_sample(T, omega, "midas")
    t_out = tensor_lab_to_sample(
        torch.tensor(T, dtype=torch.float64),
        torch.tensor(omega, dtype=torch.float64),
        "midas",
    )
    assert isinstance(t_out, torch.Tensor)
    _eq(t_out, np_out)


# ---------------------------------------------------------------------------
# grains_midas_to_sample
# ---------------------------------------------------------------------------


def test_grains_midas_to_sample_torch_matches_numpy():
    rng = np.random.default_rng(0)
    n = 4
    orient = np.tile(np.eye(3), (n, 1, 1)).astype(np.float64)
    pos = rng.uniform(-100, 100, (n, 3))
    strains = rng.uniform(-1e-3, 1e-3, (n, 3, 3))
    strains = (strains + np.transpose(strains, (0, 2, 1))) / 2

    np_out = grains_midas_to_sample(orient, pos, strains, omega_deg=15.0, target_frame="aps")
    t_out = grains_midas_to_sample(
        torch.tensor(orient, dtype=torch.float64),
        torch.tensor(pos, dtype=torch.float64),
        torch.tensor(strains, dtype=torch.float64),
        omega_deg=15.0,
        target_frame="aps",
    )
    assert isinstance(t_out["orientations"], torch.Tensor)
    _eq(t_out["orientations"], np_out["orientations"])
    _eq(t_out["positions"], np_out["positions"])
    _eq(t_out["strains"], np_out["strains"])


# ---------------------------------------------------------------------------
# dtype + autograd
# ---------------------------------------------------------------------------


def test_torch_outputs_preserve_dtype_and_device():
    v32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    out = vector_midas_to_aps(v32)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


def test_lab_to_sample_rotation_is_differentiable():
    omega = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    R = lab_to_sample_rotation(omega, "midas")
    R.sum().backward()
    assert omega.grad is not None
    assert torch.isfinite(omega.grad)
