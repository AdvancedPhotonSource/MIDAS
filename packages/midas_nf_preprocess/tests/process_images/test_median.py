"""Tests for temporal_median and spatial_median."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_preprocess.process_images import spatial_median, temporal_median

try:
    from scipy.ndimage import median_filter as scipy_median
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# -----------------------------------------------------------------------------
# Temporal median
# -----------------------------------------------------------------------------


def test_temporal_median_constant_stack():
    stack = torch.ones(7, 4, 5) * 42.0
    med = temporal_median(stack)
    assert med.shape == (4, 5)
    assert torch.all(med == 42.0)


def test_temporal_median_matches_numpy():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(size=(11, 6, 8)).astype(np.float64)
    stack = torch.from_numpy(arr)
    med = temporal_median(stack)
    # numpy uses average of two middle values; torch.median picks the lower one.
    # For odd N (=11) they agree exactly.
    np_med = np.median(arr, axis=0)
    np.testing.assert_allclose(med.numpy(), np_med, atol=1e-10)


def test_temporal_median_recovers_background(noisy_blob_stack):
    stack, centers, background = noisy_blob_stack(N=11, H=32, W=32, seed=1)
    med = temporal_median(stack)
    # The blob moves through positions; at any one pixel, the median should be
    # close to the background (since the blob touches each pixel only briefly).
    # Test on a corner pixel that the blob never reaches:
    assert torch.allclose(med[0, 0], background[0, 0], atol=1.0)


def test_temporal_median_dtype_preserved():
    stack = torch.zeros(3, 2, 2, dtype=torch.float32)
    med = temporal_median(stack)
    assert med.dtype == torch.float32


def test_temporal_median_wrong_ndim_raises():
    with pytest.raises(ValueError, match="\\[N, Z, Y\\]"):
        temporal_median(torch.zeros(4, 4))


# -----------------------------------------------------------------------------
# Spatial median
# -----------------------------------------------------------------------------


def test_spatial_median_radius_zero_is_identity():
    img = torch.randn(8, 8, dtype=torch.float64)
    out = spatial_median(img, radius=0)
    assert torch.equal(out, img)


def test_spatial_median_constant_image():
    img = torch.ones(10, 10, dtype=torch.float64) * 5.0
    out = spatial_median(img, radius=1)
    assert torch.allclose(out, img)


def test_spatial_median_3x3_removes_isolated_spike():
    img = torch.zeros(7, 7, dtype=torch.float64)
    img[3, 3] = 1000.0
    out = spatial_median(img, radius=1)
    # Spike's neighborhood is 8 zeros + 1 spike; median of 9 values is 0.
    assert out[3, 3] == 0.0
    # Border passes through unchanged: edge pixels stay zero (and they were zero).
    assert torch.all(out[0, :] == 0.0)
    assert torch.all(out[-1, :] == 0.0)


def test_spatial_median_5x5_removes_2x2_spike():
    img = torch.zeros(9, 9, dtype=torch.float64)
    img[4:6, 4:6] = 1000.0
    out = spatial_median(img, radius=2)
    # 2x2 of spikes inside a 5x5 of zeros: spike pixels see 21 zeros + 4 spikes;
    # median of 25 is 0.
    assert out[4, 4] == 0.0
    assert out[5, 5] == 0.0


def test_spatial_median_borders_unchanged():
    rng = np.random.default_rng(42)
    img = torch.from_numpy(rng.standard_normal((20, 20)))
    out = spatial_median(img, radius=1)
    # Top/bottom row + left/right column: untouched
    np.testing.assert_array_equal(out[0, :].numpy(), img[0, :].numpy())
    np.testing.assert_array_equal(out[-1, :].numpy(), img[-1, :].numpy())
    np.testing.assert_array_equal(out[:, 0].numpy(), img[:, 0].numpy())
    np.testing.assert_array_equal(out[:, -1].numpy(), img[:, -1].numpy())


@pytest.mark.parity
@pytest.mark.skipif(not HAVE_SCIPY, reason="needs scipy")
def test_spatial_median_3x3_matches_scipy_interior():
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 100, size=(20, 20)).astype(np.float64)
    out = spatial_median(torch.from_numpy(arr), radius=1).numpy()
    # scipy reflects at borders; we pass through. So compare interior only.
    scipy_out = scipy_median(arr, size=3, mode="reflect")
    np.testing.assert_allclose(out[1:-1, 1:-1], scipy_out[1:-1, 1:-1])


@pytest.mark.parity
@pytest.mark.skipif(not HAVE_SCIPY, reason="needs scipy")
def test_spatial_median_5x5_matches_scipy_interior():
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 100, size=(20, 20)).astype(np.float64)
    out = spatial_median(torch.from_numpy(arr), radius=2).numpy()
    scipy_out = scipy_median(arr, size=5, mode="reflect")
    np.testing.assert_allclose(out[2:-2, 2:-2], scipy_out[2:-2, 2:-2])


def test_spatial_median_small_image():
    """Image smaller than the kernel should pass through unchanged."""
    img = torch.arange(9.0).reshape(3, 3)
    out = spatial_median(img, radius=2)  # k=5, but image is 3x3
    assert torch.equal(out, img)


def test_spatial_median_negative_radius_raises():
    with pytest.raises(ValueError, match="radius must be"):
        spatial_median(torch.zeros(4, 4), radius=-1)


def test_spatial_median_wrong_ndim_raises():
    with pytest.raises(ValueError, match="\\[Z, Y\\]"):
        spatial_median(torch.zeros(2, 3, 4), radius=1)
