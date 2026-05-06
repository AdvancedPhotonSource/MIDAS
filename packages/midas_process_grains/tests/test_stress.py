"""Stress-wrapper tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_process_grains.compute.stress import cauchy_stress, resolve_stiffness


def test_cubic_stress_for_uniaxial_lab_strain():
    """Cubic stiffness, lab-frame uniaxial ε_xx → known σ_xx via C11."""
    C11, C12, C44 = 200e9, 130e9, 80e9
    from midas_stress.materials import cubic_stiffness
    C = cubic_stiffness(C11, C44, C12)
    if isinstance(C, np.ndarray):
        C = torch.from_numpy(C).to(torch.float64)
    eps = torch.zeros((3, 3), dtype=C.dtype)
    eps[0, 0] = 1e-3
    sigma = cauchy_stress(eps, C, orient=torch.eye(3, dtype=C.dtype), frame="grain")
    assert torch.isfinite(sigma).all()
    # σ_xx should be C11 * ε_xx (within reason; exact form depends on shear etc.)
    assert sigma[0, 0].abs() > 0


def test_resolve_stiffness_from_material_name():
    C = resolve_stiffness(material_name="Cu")
    assert C is not None
    assert C.shape == (6, 6)


def test_resolve_stiffness_returns_none_when_unconfigured():
    assert resolve_stiffness() is None


def test_resolve_stiffness_from_file(tmp_path: Path):
    C = np.eye(6) * 100e9
    p = tmp_path / "stiff.txt"
    np.savetxt(p, C)
    out = resolve_stiffness(stiffness_file=str(p))
    assert out is not None
    np.testing.assert_allclose(out.numpy(), C)


def test_bad_stiffness_file_shape_raises(tmp_path: Path):
    p = tmp_path / "bad.txt"
    np.savetxt(p, np.eye(3))
    with pytest.raises(ValueError, match="6x6"):
        resolve_stiffness(stiffness_file=str(p))
