"""Tests for the LoG kernel and convolution."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_nf_preprocess.process_images import apply_log, build_log_kernel
from midas_nf_preprocess.process_images.log_filter import _C_INT_MAGIC


# -----------------------------------------------------------------------------
# Kernel construction
# -----------------------------------------------------------------------------


def test_log_kernel_shape():
    k = build_log_kernel(radius=4, sigma=1.0)
    assert k.shape == (9, 9)


def test_log_kernel_radius_1():
    k = build_log_kernel(radius=1, sigma=0.5)
    assert k.shape == (3, 3)


def test_log_kernel_dtype_default_float64():
    k = build_log_kernel(radius=3, sigma=1.0)
    assert k.dtype == torch.float64


def test_log_kernel_invalid_radius():
    with pytest.raises(ValueError, match="LoG radius"):
        build_log_kernel(radius=0, sigma=1.0)


def test_log_kernel_center_value_matches_formula():
    """At (x=0, y=0): k = -(1/(pi*sigma^4)) * (1 - 0) * exp(0) = -1/(pi*sigma^4)."""
    sigma = 1.5
    k = build_log_kernel(radius=4, sigma=sigma)
    expected = -1.0 / (math.pi * sigma ** 4)
    assert torch.isclose(k[4, 4], torch.tensor(expected, dtype=torch.float64))


def test_log_kernel_symmetry():
    """LoG kernel is rotationally symmetric (and 4-fold for unit pixels)."""
    k = build_log_kernel(radius=3, sigma=1.0)
    # Up-down symmetry
    assert torch.allclose(k, k.flip(0))
    # Left-right symmetry
    assert torch.allclose(k, k.flip(1))
    # Diagonal symmetry (k[i,j] == k[j,i])
    assert torch.allclose(k, k.T)


def test_log_kernel_integer_mode_matches_c_quantization():
    """integer=True applies the magic 79720 factor (C ProcessImagesCombined.c L252)."""
    k_float = build_log_kernel(radius=4, sigma=1.0, integer=False)
    k_int = build_log_kernel(radius=4, sigma=1.0, integer=True)
    assert k_int.dtype == torch.int64
    # Should be approximately 79720 * float kernel (truncation toward zero).
    expected = (_C_INT_MAGIC * k_float).to(torch.int64)
    assert torch.equal(k_int, expected)


def test_log_kernel_sigma_as_tensor_keeps_grad():
    sigma = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    k = build_log_kernel(radius=4, sigma=sigma)
    assert k.requires_grad
    loss = k.sum()
    loss.backward()
    assert sigma.grad is not None


# -----------------------------------------------------------------------------
# Convolution
# -----------------------------------------------------------------------------


def test_apply_log_zero_input_zero_output():
    img = torch.zeros(20, 20, dtype=torch.float64)
    k = build_log_kernel(radius=2, sigma=1.0)
    out = apply_log(img, k)
    assert torch.allclose(out, torch.zeros_like(out))


def test_apply_log_borders_zeroed():
    """Pixels within ``radius`` of an edge are zeroed (matches C semantics)."""
    img = torch.ones(20, 20, dtype=torch.float64)
    k = build_log_kernel(radius=4, sigma=1.0)
    out = apply_log(img, k)
    # Top/bottom 4 rows, left/right 4 columns must be exactly zero.
    assert torch.all(out[:4, :] == 0)
    assert torch.all(out[-4:, :] == 0)
    assert torch.all(out[:, :4] == 0)
    assert torch.all(out[:, -4:] == 0)


def test_apply_log_blob_response_negative_inside():
    """The LoG kernel is negative-center; convolving with a positive blob gives
    a *negative* response at the blob center (standard Mexican-hat convention)."""
    H = 33
    sigma = 2.0
    z = torch.arange(H, dtype=torch.float64).view(-1, 1) - H // 2
    y = torch.arange(H, dtype=torch.float64).view(1, -1) - H // 2
    blob = torch.exp(-(z * z + y * y) / (2 * sigma ** 2))
    k = build_log_kernel(radius=int(2 * sigma), sigma=sigma)
    out = apply_log(blob, k)
    center = H // 2
    # Center should be a local *min* of the LoG response and negative.
    assert out[center, center] < 0
    # Off-center (but still inside the non-zero LoG output region): less negative.
    edge_offset = int(2 * sigma) + 1
    assert out[center + edge_offset, center] > out[center, center]


def test_apply_log_output_shape_preserved():
    img = torch.randn(40, 30, dtype=torch.float64)
    k = build_log_kernel(radius=3, sigma=1.0)
    out = apply_log(img, k)
    assert out.shape == img.shape


def test_apply_log_invalid_kernel_size_raises():
    img = torch.zeros(8, 8, dtype=torch.float64)
    even_kernel = torch.zeros(4, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="odd"):
        apply_log(img, even_kernel)


def test_apply_log_non_square_kernel_raises():
    img = torch.zeros(8, 8, dtype=torch.float64)
    bad = torch.zeros(3, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="square"):
        apply_log(img, bad)


def test_apply_log_dtype_mismatch_raises():
    img = torch.zeros(20, 20, dtype=torch.float64)
    k = build_log_kernel(radius=2, sigma=1.0, integer=True)  # int64
    with pytest.raises(ValueError, match="incompatible"):
        apply_log(img, k)


def test_apply_log_integer_mode_with_int_image():
    """Integer kernel with integer image should produce integer output."""
    img = torch.zeros(20, 20, dtype=torch.int64)
    img[10, 10] = 100
    k = build_log_kernel(radius=2, sigma=1.0, integer=True)
    out = apply_log(img, k)
    assert out.dtype == torch.int64
