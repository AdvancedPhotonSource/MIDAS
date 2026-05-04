"""Differentiability tests: gradcheck on primitives, end-to-end loss.backward.

The pipeline is differentiable along the path:

    image  ->  median-subtract  ->  spatial median  ->  LoG conv  ->  spot_prob

and (for soft kernels) along:

    sigma  ->  LoG kernel  ->  LoG response  ->  spot_prob

The detached path (labels, SpotsBitMask) is not.
"""

from __future__ import annotations

import pytest
import torch

from midas_nf_preprocess.process_images import (
    ProcessImagesPipeline,
    ProcessParams,
    apply_log,
    build_log_kernel,
    find_peaks,
    spatial_median,
    temporal_median,
)


# Use float64 throughout for gradcheck precision.
DTYPE = torch.float64


# -----------------------------------------------------------------------------
# gradcheck on primitives
# -----------------------------------------------------------------------------


def test_temporal_median_gradcheck():
    # Use unique values so the median selection is unambiguous (avoids ties
    # which would give an undefined subgradient).
    stack = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda x: temporal_median(x).sum(), (stack,), eps=1e-6, atol=1e-5
    )


def test_spatial_median_gradcheck_radius_1():
    # 5x5 with unique values, no ties.
    img = torch.arange(25, dtype=DTYPE).reshape(5, 5).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda x: spatial_median(x, radius=1).sum(), (img,), eps=1e-6, atol=1e-5
    )


def test_spatial_median_gradcheck_radius_0_identity():
    img = torch.randn(4, 4, dtype=DTYPE, requires_grad=True)
    out = spatial_median(img, radius=0)
    assert out is img  # no-op


def test_apply_log_gradcheck_in_image():
    img = torch.randn(10, 10, dtype=DTYPE, requires_grad=True)
    k = build_log_kernel(radius=2, sigma=1.0, dtype=DTYPE)
    assert torch.autograd.gradcheck(
        lambda x: apply_log(x, k).sum(), (img,), eps=1e-6, atol=1e-5
    )


def test_apply_log_gradcheck_in_sigma():
    img = torch.randn(10, 10, dtype=DTYPE)
    sigma = torch.tensor(1.5, dtype=DTYPE, requires_grad=True)
    def f(s):
        k = build_log_kernel(radius=2, sigma=s, dtype=DTYPE)
        return apply_log(img, k).sum()
    assert torch.autograd.gradcheck(f, (sigma,), eps=1e-6, atol=1e-4)


def test_log_kernel_gradcheck_in_sigma():
    sigma = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda s: build_log_kernel(radius=2, sigma=s, dtype=DTYPE).sum(),
        (sigma,),
        eps=1e-6,
        atol=1e-5,
    )


# -----------------------------------------------------------------------------
# Soft-surrogate gradient flow
# -----------------------------------------------------------------------------


def test_find_peaks_spot_prob_grad_to_image():
    """spot_prob should backprop to the input image."""
    img = torch.empty(20, 20, dtype=DTYPE).uniform_(0, 100).requires_grad_(True)
    k = build_log_kernel(radius=3, sigma=1.5, dtype=DTYPE)
    out = find_peaks(img, [k], soft_temperature=1.0)
    loss = out.spot_prob.sum()
    loss.backward()
    assert img.grad is not None
    assert img.grad.abs().sum() > 0


def test_find_peaks_log_response_grad_to_sigma():
    """log_response (autograd path) should backprop to sigma."""
    img = torch.rand(20, 20, dtype=DTYPE) * 100
    sigma = torch.tensor(1.5, dtype=DTYPE, requires_grad=True)
    k = build_log_kernel(radius=3, sigma=sigma, dtype=DTYPE)
    out = find_peaks(img, [k])
    loss = out.log_response.sum()
    loss.backward()
    assert sigma.grad is not None
    assert sigma.grad.abs() > 0


def test_find_peaks_labels_have_no_grad():
    img = torch.rand(20, 20, dtype=DTYPE, requires_grad=True)
    k = build_log_kernel(radius=3, sigma=1.5, dtype=DTYPE)
    out = find_peaks(img, [k])
    assert not out.labels.requires_grad
    # Trying to backprop through labels should not affect grad of img.
    img.grad = None
    # labels is int64 and detached -> can't even .backward() on it, that's the point.
    assert out.labels.dtype == torch.int64


# -----------------------------------------------------------------------------
# End-to-end pipeline differentiability
# -----------------------------------------------------------------------------


def test_pipeline_loss_backward_to_image():
    """A loss on spot_prob should backprop through the entire pipeline."""
    p = ProcessParams(
        nr_pixels_y=24, nr_pixels_z=24, log_mask_radius=3,
        sigma=1.5, mean_filt_radius=1,
    )
    pipe = ProcessImagesPipeline(p, device="cpu", dtype="fp64")

    frame = torch.empty(24, 24, dtype=DTYPE).uniform_(0, 200).requires_grad_(True)
    median = torch.zeros(24, 24, dtype=DTYPE)
    out = pipe.process_frame(0, frame, median, layer_nr=1)
    loss = out.spot_prob.sum()
    loss.backward()

    assert frame.grad is not None
    assert frame.grad.abs().sum() > 0


def test_pipeline_loss_backward_to_blanket_subtraction():
    """Treat blanket_subtraction as a tensor and backprop through it."""
    # Blanket isn't a tensor in ProcessParams (it's an int), so we test the
    # mathematical path directly: median-subtract is `frame - median - blanket`,
    # which is linear in blanket. We construct the same expression manually.
    frame = torch.rand(20, 20, dtype=DTYPE) * 200
    median = torch.zeros(20, 20, dtype=DTYPE)
    blanket = torch.tensor(5.0, dtype=DTYPE, requires_grad=True)
    img = torch.clamp(frame - median - blanket, min=0)
    k = build_log_kernel(radius=3, sigma=1.5, dtype=DTYPE)
    img = spatial_median(img, radius=1)
    out = find_peaks(img, [k])
    loss = out.spot_prob.sum()
    loss.backward()
    # Blanket affects every above-clip pixel, so the grad must be nonzero.
    assert blanket.grad is not None
    assert blanket.grad.abs() > 0


def test_pipeline_filtered_grad_zero_when_input_far_below_median():
    """If frame << median, filtered is clamped at 0 everywhere -> zero gradient."""
    frame = torch.zeros(16, 16, dtype=DTYPE, requires_grad=True)
    median = 1000 * torch.ones(16, 16, dtype=DTYPE)
    img = torch.clamp(frame - median, min=0)
    img.sum().backward()
    # All entries clamped to zero; gradient through clamp is also zero in that region.
    assert torch.all(frame.grad == 0)
