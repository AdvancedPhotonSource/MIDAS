"""Tests for the SpotsInfo bitmap loader and overlap kernels."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.obs_volume import ObsVolume, _unpack_bits


def test_obs_volume_default_dtype_is_uint8():
    """v0.3 default storage is uint8 — 4× smaller than float32 for
    the 24 GB Au-example obs volume, and ``hard_fraction`` only needs
    0/1 values. The L-BFGS soft-overlap path explicitly upcasts to
    float by passing ``dtype=torch.float32`` at construction."""
    arr = np.zeros((1, 4, 8, 8), dtype=np.float32)
    arr[0, 2, 3, 4] = 1.0
    ov = ObsVolume.from_dense_array(arr)        # default dtype
    assert ov.dense.dtype == torch.uint8

    # hard_fraction works fine with uint8 obs.
    frame = torch.tensor([2.0])
    y = torch.tensor([3.0])
    z = torch.tensor([4.0])
    valid = torch.tensor([1.0])
    frac = ov.hard_fraction(frame, y, z, valid)
    assert frac.item() == pytest.approx(1.0)


def test_obs_volume_soft_fraction_rejects_uint8():
    """``soft_fraction`` needs grid_sample which requires float; the
    error message should point users to the float32 escape hatch."""
    import math
    arr = np.zeros((1, 4, 8, 8), dtype=np.float32)
    ov = ObsVolume.from_dense_array(arr)        # uint8
    with pytest.raises(TypeError, match="floating-point"):
        ov.soft_fraction(
            torch.tensor([2.0]), torch.tensor([3.0]),
            torch.tensor([4.0]), torch.tensor([1.0]),
            sigma_px=1.0,
        )


def test_unpack_bits_lsb_first():
    # uint32 with bits 0, 5, 31 set
    word = (1 << 0) | (1 << 5) | (1 << 31)
    arr = np.array([word], dtype=np.uint32)
    bits = _unpack_bits(arr, total_bits=32)
    assert bits[0] == 1
    assert bits[5] == 1
    assert bits[31] == 1
    # all other bits zero
    bits[0] = bits[5] = bits[31] = 0
    assert bits.sum() == 0


def test_obs_volume_from_dense_array():
    arr = np.zeros((1, 4, 8, 8), dtype=np.float32)
    arr[0, 2, 3, 4] = 1.0
    ov = ObsVolume.from_dense_array(arr)
    assert ov.dense.shape == (1, 4, 8, 8)
    assert ov.dense[0, 2, 3, 4].item() == 1.0


def test_hard_fraction_full_match():
    arr = np.zeros((1, 4, 8, 8), dtype=np.float32)
    arr[0, 2, 3, 4] = 1.0
    arr[0, 2, 5, 5] = 1.0
    ov = ObsVolume.from_dense_array(arr)
    frame = torch.tensor([2.0, 2.0])
    y = torch.tensor([3.0, 5.0])
    z = torch.tensor([4.0, 5.0])
    valid = torch.tensor([1.0, 1.0])
    frac = ov.hard_fraction(frame, y, z, valid)
    assert frac.item() == pytest.approx(1.0)


def test_hard_fraction_half_match():
    arr = np.zeros((1, 4, 8, 8), dtype=np.float32)
    arr[0, 2, 3, 4] = 1.0
    ov = ObsVolume.from_dense_array(arr)
    frame = torch.tensor([2.0, 2.0])
    y = torch.tensor([3.0, 7.0])
    z = torch.tensor([4.0, 7.0])
    valid = torch.tensor([1.0, 1.0])
    frac = ov.hard_fraction(frame, y, z, valid)
    assert frac.item() == pytest.approx(0.5)


def test_soft_fraction_differentiable():
    arr = np.zeros((1, 4, 8, 8), dtype=np.float32)
    arr[0, 2, 3, 4] = 1.0
    ov = ObsVolume.from_dense_array(arr, dtype=torch.float64)
    frame = torch.tensor([2.0, 2.0], requires_grad=False)
    y = torch.tensor([3.5, 5.0], requires_grad=True, dtype=torch.float64)
    z = torch.tensor([4.0, 5.0], requires_grad=False, dtype=torch.float64)
    valid = torch.tensor([1.0, 1.0])
    frac = ov.soft_fraction(frame, y, z, valid, sigma_px=0.5)
    assert frac.requires_grad
    frac.backward()
    # Pulling the first spot back toward y=3 (the lit pixel) should
    # raise the overlap, so dfrac/dy[0] should be negative.
    assert y.grad[0].item() < 0


def test_multi_distance_layered_requires_all_hits():
    # Two distances. Spot lit on distance 0 only.
    arr = np.zeros((2, 4, 8, 8), dtype=np.float32)
    arr[0, 2, 3, 4] = 1.0  # distance 0 only
    ov = ObsVolume.from_dense_array(arr)
    frame = torch.tensor([2.0])
    y = torch.tensor([[3.0], [3.0]])  # (D=2, M=1) per-distance
    z = torch.tensor([[4.0], [4.0]])
    valid = torch.tensor([1.0])
    frac = ov.hard_fraction(frame, y, z, valid)
    # Only one of the two distances has a hit, so prod across distances
    # = 0, fraction = 0.
    assert frac.item() == pytest.approx(0.0)
