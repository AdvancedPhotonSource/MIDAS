"""Tests for the packed-bit obs storage (1 bit / pixel, the C
``SpotsInfo.bin`` layout).

Two parity targets:

1. Packed and dense ``uint8`` modes return **identical** hard
   fractions for every input.
2. The bit ordering matches the C ``TestBit`` macro on a
   little-endian host: bit ``k`` is at byte ``k // 8``, bit ``k % 8``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.obs_volume import ObsVolume


def _random_dense(rng, shape, density=0.05):
    """Synthetic obs bitmap with ~``density`` lit pixels."""
    arr = (rng.random(shape) < density).astype(np.float32)
    return arr


def test_packed_and_dense_agree_on_hard_fraction():
    """For random spots over a random obs, packed and dense storage
    must return bit-identical hard fractions."""
    rng = np.random.default_rng(0)
    arr = _random_dense(rng, (2, 32, 64, 64), density=0.05)
    ov_dense = ObsVolume.from_dense_array(arr, packed=False)
    ov_packed = ObsVolume.from_dense_array(arr, packed=True)

    M = 50
    frame = torch.tensor(rng.integers(0, 32, size=M).astype(np.float64))
    y_pix = torch.tensor(rng.integers(0, 64, size=(2, M)).astype(np.float64))
    z_pix = torch.tensor(rng.integers(0, 64, size=(2, M)).astype(np.float64))
    valid = torch.ones(M, dtype=torch.float64)

    f_dense = ov_dense.hard_fraction(frame, y_pix, z_pix, valid)
    f_packed = ov_packed.hard_fraction(frame, y_pix, z_pix, valid)
    assert torch.allclose(f_dense.float(), f_packed.float(), atol=1e-9)


def test_packed_handles_batched_inputs():
    """Multi-grain batched lookup (shape (B, K, M)) must also agree."""
    rng = np.random.default_rng(1)
    arr = _random_dense(rng, (2, 16, 32, 32), density=0.10)
    ov_dense = ObsVolume.from_dense_array(arr, packed=False)
    ov_packed = ObsVolume.from_dense_array(arr, packed=True)

    B, K, M = 4, 2, 12
    frame = torch.tensor(rng.integers(0, 16, size=(B, K, M)).astype(np.float64))
    y_pix = torch.tensor(
        rng.integers(0, 32, size=(2, B, K, M)).astype(np.float64),
    )
    z_pix = torch.tensor(
        rng.integers(0, 32, size=(2, B, K, M)).astype(np.float64),
    )
    valid = torch.ones(B, K, M, dtype=torch.float64)

    f_dense = ov_dense.hard_fraction(frame, y_pix, z_pix, valid)
    f_packed = ov_packed.hard_fraction(frame, y_pix, z_pix, valid)
    assert f_dense.shape == (B,)
    assert f_packed.shape == (B,)
    assert torch.allclose(f_dense.float(), f_packed.float(), atol=1e-9)


def test_packed_storage_is_1_bit_per_pixel():
    """The packed buffer must be ``ceil(n_pixels / 8)`` bytes."""
    arr = np.zeros((1, 16, 32, 32), dtype=np.float32)
    ov = ObsVolume.from_dense_array(arr, packed=True)
    n_pixels = 1 * 16 * 32 * 32
    expected_bytes = (n_pixels + 7) // 8
    assert ov.packed.numel() == expected_bytes


def test_packed_soft_fraction_raises():
    """Soft path needs a dense float volume; the error must point
    users to the right escape hatch."""
    arr = np.zeros((1, 8, 16, 16), dtype=np.float32)
    ov = ObsVolume.from_dense_array(arr, packed=True)
    with pytest.raises(TypeError, match="packed-bit"):
        ov.soft_fraction(
            torch.tensor([2.0]), torch.tensor([3.0]),
            torch.tensor([4.0]), torch.tensor([1.0]),
            sigma_px=1.0,
        )


def test_packed_construction_validates_shape():
    """Construction guards against missing dimensions."""
    bytes_t = torch.zeros(16, dtype=torch.uint8)
    with pytest.raises(ValueError, match="n_distances"):
        ObsVolume(packed=bytes_t)


def test_packed_bit_layout_matches_c_testbit():
    """Mirror the C ``TestBit`` macro on a little-endian host:
    bit_idx = (((d * F + f) * H + y) * W + z); byte_idx = bit_idx // 8;
    bit_pos = bit_idx % 8; byte[byte_idx] >> bit_pos & 1.
    """
    # Lit one specific pixel; the packed storage must encode it at
    # the byte/bit position the formula predicts.
    D, F_, H, W = 1, 4, 8, 8
    arr = np.zeros((D, F_, H, W), dtype=np.float32)
    # Pick (d=0, f=2, y=3, z=5).
    arr[0, 2, 3, 5] = 1.0
    ov = ObsVolume.from_dense_array(arr, packed=True)
    bit_idx = (((0 * F_ + 2) * H + 3) * W + 5)
    byte_idx = bit_idx // 8
    bit_pos = bit_idx % 8
    assert int(ov.packed[byte_idx]) == (1 << bit_pos)
