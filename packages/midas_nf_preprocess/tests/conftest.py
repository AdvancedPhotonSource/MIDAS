"""Shared pytest fixtures for midas_nf_preprocess (used across all submodules).

Pins the default device to CPU during testing so float64 is available
everywhere (MPS does not support float64). Tests that want to exercise an
accelerator pass ``device=...`` explicitly or use the ``device`` fixture
parametrized over ``ALL_DEVICES``.
"""

from __future__ import annotations

import os

# Force CPU as the default device for tests; individual tests can still
# override with explicit device kwargs or the ``device`` fixture.
os.environ.setdefault("MIDAS_NF_PREPROCESS_DEVICE", "cpu")

import pytest
import torch


# --- Device fixtures ---------------------------------------------------------

ALL_DEVICES = ["cpu"]
if torch.cuda.is_available():
    ALL_DEVICES.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    ALL_DEVICES.append("mps")


@pytest.fixture(params=ALL_DEVICES)
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


# --- Synthetic data fixtures -------------------------------------------------


@pytest.fixture
def gaussian_blob_image():
    """Single 64x64 image with two well-separated Gaussian blobs.

    Returns ``(img, expected_centers)`` where ``expected_centers`` is a list of
    ``(z, y)`` tuples.
    """

    def _make(device="cpu", dtype=torch.float64):
        H, W = 64, 64
        z = torch.arange(H, device=device, dtype=dtype).view(-1, 1)
        y = torch.arange(W, device=device, dtype=dtype).view(1, -1)
        centers = [(15, 20), (45, 50)]
        img = torch.zeros(H, W, device=device, dtype=dtype)
        for cz, cy in centers:
            r2 = (z - cz) ** 2 + (y - cy) ** 2
            img = img + 1000.0 * torch.exp(-r2 / (2 * 2.5 ** 2))
        return img, centers

    return _make


@pytest.fixture
def noisy_blob_stack():
    """Stack of N frames, each with a moving Gaussian blob + uniform background.

    The temporal median should recover the (frame-mean) background; per-frame
    median-subtract should reveal the blob.
    """

    def _make(N=11, H=64, W=64, device="cpu", dtype=torch.float64, seed=0):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.arange(H, dtype=dtype).view(-1, 1)
        y = torch.arange(W, dtype=dtype).view(1, -1)
        background = 100.0 * torch.ones(H, W, dtype=dtype)
        stack = torch.zeros(N, H, W, dtype=dtype)
        centers = []
        for n in range(N):
            cz = 10 + n * 4
            cy = 12 + n * 3
            centers.append((cz, cy))
            r2 = (z - cz) ** 2 + (y - cy) ** 2
            blob = 800.0 * torch.exp(-r2 / (2 * 2.0 ** 2))
            stack[n] = background + blob
        # Add a small constant noise so quick_select selects deterministically.
        stack = stack + 0.01 * torch.randn(stack.shape, generator=gen, dtype=dtype)
        return stack.to(device=device), centers, background.to(device=device)

    return _make


# --- Skip helpers ------------------------------------------------------------


def needs_cuda():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA"
    )


def needs_mps():
    return pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="requires Apple MPS",
    )
