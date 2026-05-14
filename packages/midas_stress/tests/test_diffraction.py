"""Tests for midas_stress.diffraction — eta-angle helper.

Parity reference is utils/calcMiso.CalcEtaAngleAll inlined to avoid a
sys.path dependency on /utils. Tested for numpy + torch backends.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from midas_stress.diffraction import calc_eta_angle_all

torch = pytest.importorskip("torch")


_RAD2DEG = 180.0 / math.pi


def _calcmiso_reference(y, z):
    """Inlined utils/calcMiso.CalcEtaAngleAll for parity gate."""
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    alpha = _RAD2DEG * np.arccos(z / np.linalg.norm(np.array([y, z]), axis=0))
    alpha[y > 0] *= -1
    return alpha


class TestCalcEtaAngleAllNumpy:
    def test_z_up(self):
        """y=0, z>0 → eta = 0°."""
        eta = calc_eta_angle_all(np.array([0.0]), np.array([1.0]))
        np.testing.assert_allclose(eta, [0.0], atol=1e-14)

    def test_z_down(self):
        """y=0, z<0 → eta = 180°."""
        eta = calc_eta_angle_all(np.array([0.0]), np.array([-1.0]))
        np.testing.assert_allclose(eta, [180.0], atol=1e-12)

    def test_y_positive(self):
        """y>0, z=0 → eta = -90° (sign flipped)."""
        eta = calc_eta_angle_all(np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(eta, [-90.0], atol=1e-12)

    def test_y_negative(self):
        """y<0, z=0 → eta = +90°."""
        eta = calc_eta_angle_all(np.array([-1.0]), np.array([0.0]))
        np.testing.assert_allclose(eta, [90.0], atol=1e-12)

    def test_matches_calcmiso_reference(self):
        rng = np.random.default_rng(42)
        n = 100
        y = rng.uniform(-5.0, 5.0, n)
        z = rng.uniform(-5.0, 5.0, n)
        # Drop the singular point at the origin.
        ok = (y * y + z * z) > 1e-6
        y = y[ok]; z = z[ok]
        out = calc_eta_angle_all(y, z)
        ref = _calcmiso_reference(y, z)
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_scalar_input_returns_scalar(self):
        eta_pos = calc_eta_angle_all(1.0, 0.0)
        eta_neg = calc_eta_angle_all(-1.0, 0.0)
        assert isinstance(eta_pos, float)
        assert isinstance(eta_neg, float)
        assert math.isclose(eta_pos, -90.0, abs_tol=1e-12)
        assert math.isclose(eta_neg, 90.0, abs_tol=1e-12)

    def test_array_input_preserves_shape(self):
        y = np.array([0.0, 1.0, -1.0, 0.5])
        z = np.array([1.0, 0.0, 0.0, 0.5])
        eta = calc_eta_angle_all(y, z)
        assert eta.shape == y.shape

    def test_2d_array_input(self):
        y = np.array([[0.0, 1.0], [-1.0, 0.5]])
        z = np.array([[1.0, 0.0], [0.0, 0.5]])
        eta = calc_eta_angle_all(y, z)
        assert eta.shape == (2, 2)
        # Spot-check one element: y=-1, z=0 → +90°.
        np.testing.assert_allclose(eta[1, 0], 90.0, atol=1e-12)


class TestCalcEtaAngleAllTorch:
    def test_torch_returns_tensor_and_matches_numpy(self):
        rng = np.random.default_rng(7)
        y_np = rng.uniform(-5.0, 5.0, 50)
        z_np = rng.uniform(-5.0, 5.0, 50)
        ok = (y_np * y_np + z_np * z_np) > 1e-6
        y_np = y_np[ok]; z_np = z_np[ok]
        eta_np = calc_eta_angle_all(y_np, z_np)
        eta_t = calc_eta_angle_all(torch.tensor(y_np), torch.tensor(z_np))
        assert isinstance(eta_t, torch.Tensor)
        np.testing.assert_allclose(eta_t.numpy(), eta_np, atol=1e-12)

    def test_torch_dtype_device_follow_input(self):
        y = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
        z = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        eta = calc_eta_angle_all(y, z)
        assert eta.dtype == torch.float32
        assert eta.device == y.device

    def test_torch_is_differentiable(self):
        """Smooth where for sign flip should not break autograd."""
        y = torch.tensor([0.3, -0.4, 0.6], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.5, 0.7, -0.2], dtype=torch.float64)
        eta = calc_eta_angle_all(y, z)
        eta.sum().backward()
        assert y.grad is not None
        assert not torch.isnan(y.grad).any()

    def test_torch_only_y_tensor(self):
        """Mixed input: y as tensor, z as float — output should be a tensor."""
        y = torch.tensor([1.0, -1.0])
        eta = calc_eta_angle_all(y, 0.0)
        assert isinstance(eta, torch.Tensor)
        np.testing.assert_allclose(eta.numpy(), np.array([-90.0, 90.0]), atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_calc_eta_angle_all_runs_on_cuda():
    y = torch.tensor([0.3, -0.5, 0.7], dtype=torch.float64, device="cuda")
    z = torch.tensor([0.5, 0.4, -0.2], dtype=torch.float64, device="cuda")
    eta = calc_eta_angle_all(y, z)
    assert eta.device.type == "cuda"
    np.testing.assert_allclose(eta.cpu().numpy(),
                               calc_eta_angle_all(y.cpu().numpy(), z.cpu().numpy()),
                               atol=1e-12)


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                    reason="MPS not available")
def test_calc_eta_angle_all_runs_on_mps():
    # MPS does not support fp64; use fp32.
    y = torch.tensor([0.3, -0.5, 0.7], dtype=torch.float32, device="mps")
    z = torch.tensor([0.5, 0.4, -0.2], dtype=torch.float32, device="mps")
    eta = calc_eta_angle_all(y, z)
    assert eta.device.type == "mps"
    np.testing.assert_allclose(eta.cpu().numpy(),
                               calc_eta_angle_all(y.cpu().numpy().astype(np.float64),
                                                  z.cpu().numpy().astype(np.float64)),
                               atol=1e-5)
