"""Device-coverage tests: parametrize core ops over cpu/cuda/mps.

These tests are skipped automatically on hosts without the relevant backend.
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
    label_components,
    spatial_median,
    temporal_median,
)


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append("mps")


# Device-appropriate dtype: float32 on accelerators, float64 on CPU.
def _device_dtype(d: str) -> torch.dtype:
    return torch.float64 if d == "cpu" else torch.float32


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


# -----------------------------------------------------------------------------
# Per-op smoke tests on each device
# -----------------------------------------------------------------------------


def test_temporal_median_on_device(device):
    dtype = _device_dtype(device.type)
    stack = torch.rand(5, 8, 8, device=device, dtype=dtype)
    med = temporal_median(stack)
    assert med.device.type == device.type
    assert med.shape == (8, 8)


def test_spatial_median_on_device(device):
    dtype = _device_dtype(device.type)
    img = torch.rand(16, 16, device=device, dtype=dtype)
    out = spatial_median(img, radius=1)
    assert out.device.type == device.type
    assert out.shape == img.shape


def test_log_kernel_on_device(device):
    dtype = _device_dtype(device.type)
    k = build_log_kernel(radius=3, sigma=1.5, device=device, dtype=dtype)
    assert k.device.type == device.type
    assert k.dtype == dtype


def test_apply_log_on_device(device):
    dtype = _device_dtype(device.type)
    img = torch.rand(20, 20, device=device, dtype=dtype)
    k = build_log_kernel(radius=3, sigma=1.5, device=device, dtype=dtype)
    out = apply_log(img, k)
    assert out.device.type == device.type
    assert out.shape == img.shape


def test_label_components_on_device(device):
    mask = torch.zeros(10, 10, dtype=torch.bool, device=device)
    mask[2:5, 2:5] = True
    mask[7:9, 7:9] = True
    labels, n = label_components(mask, return_n=True)
    assert labels.device.type == device.type
    assert n == 2


def test_find_peaks_on_device(device):
    dtype = _device_dtype(device.type)
    z = torch.arange(32, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(32, device=device, dtype=dtype).view(1, -1)
    blob = 1000 * torch.exp(-((z - 16) ** 2 + (y - 16) ** 2) / (2 * 2.0 ** 2))
    k = build_log_kernel(radius=4, sigma=2.0, device=device, dtype=dtype)
    out = find_peaks(blob, [k])
    assert out.labels.device.type == device.type
    assert out.spot_prob.device.type == device.type
    assert out.n_components >= 1


def test_pipeline_process_frame_on_device(device):
    dtype = _device_dtype(device.type)
    p = ProcessParams(nr_pixels_y=24, nr_pixels_z=24, log_mask_radius=3, sigma=1.5)
    pipe = ProcessImagesPipeline(p, device=device, dtype=dtype)
    frame = torch.rand(24, 24, device=device, dtype=dtype) * 200
    median = torch.zeros(24, 24, device=device, dtype=dtype)
    out = pipe.process_frame(0, frame, median, layer_nr=1)
    assert out.filtered.device.type == device.type
    assert out.labels.device.type == device.type


def test_pipeline_process_layer_on_device(device):
    dtype = _device_dtype(device.type)
    p = ProcessParams(
        nr_pixels_y=20, nr_pixels_z=20, nr_files_per_distance=3,
        n_distances=1, log_mask_radius=3, sigma=1.5, mean_filt_radius=0,
    )
    pipe = ProcessImagesPipeline(p, device=device, dtype=dtype)
    z = torch.arange(20, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(20, device=device, dtype=dtype).view(1, -1)
    stack = torch.zeros(3, 20, 20, device=device, dtype=dtype)
    for j in range(3):
        cz, cy = 5 + j * 4, 6 + j * 3
        stack[j] = 100 + 800 * torch.exp(-((z - cz) ** 2 + (y - cy) ** 2) / (2 * 2.0 ** 2))
    bm = pipe.process_layer(1, stack=stack)
    assert bm.count_bits() > 0


# -----------------------------------------------------------------------------
# Cross-device parity (CPU is reference)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(len(DEVICES) < 2, reason="needs an accelerator")
def test_temporal_median_cross_device_parity():
    rng = torch.Generator(device="cpu").manual_seed(0)
    stack_cpu = torch.rand(5, 16, 16, generator=rng, dtype=torch.float64)
    med_cpu = temporal_median(stack_cpu)
    for d in DEVICES[1:]:
        med_dev = temporal_median(stack_cpu.to(device=d, dtype=torch.float32))
        assert torch.allclose(
            med_dev.cpu().to(torch.float64), med_cpu, atol=1e-4
        ), f"temporal_median diverged on {d}"


@pytest.mark.skipif(len(DEVICES) < 2, reason="needs an accelerator")
def test_label_components_cross_device_parity():
    """Component partition should match across devices (label IDs may differ)."""
    rng = torch.Generator(device="cpu").manual_seed(7)
    mask_cpu = torch.rand(20, 20, generator=rng) > 0.6
    labels_cpu, n_cpu = label_components(mask_cpu, return_n=True)
    for d in DEVICES[1:]:
        labels_dev, n_dev = label_components(mask_cpu.to(d), return_n=True)
        assert n_dev == n_cpu, f"component count differs on {d}"
