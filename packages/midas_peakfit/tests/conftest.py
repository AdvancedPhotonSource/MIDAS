"""Shared fixtures for midas_peakfit tests.

Generates a tiny synthetic Zarr archive with known peak positions, used by
multiple test modules.
"""
from __future__ import annotations

import os

# Workaround: PyTorch and SciPy can ship duplicate libomp on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import zarr


@pytest.fixture(scope="session")
def synthetic_zarr() -> Iterator[Path]:
    """Build a small Zarr archive with 2 well-separated Pseudo-Voigt peaks
    in 2 frames. Yields the path; cleans up at session end.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="midas_peakfit_test_"))
    zip_path = tmpdir / "synthetic.MIDAS.zip"
    nFrames, NrPixelsZ, NrPixelsY = 3, 256, 256

    Y = np.arange(NrPixelsY, dtype=np.float64)
    Z = np.arange(NrPixelsZ, dtype=np.float64)
    Yg, Zg = np.meshgrid(Y, Z, indexing="xy")  # shapes (Z, Y)

    def gauss2d(y0, z0, amp, sig):
        return amp * np.exp(-((Yg - y0) ** 2 + (Zg - z0) ** 2) / (2 * sig * sig))

    data = np.zeros((nFrames, NrPixelsZ, NrPixelsY), dtype=np.uint16)
    # Frame 0: 2 peaks
    data[0] = (
        gauss2d(60, 70, 1500, 4) + gauss2d(180, 200, 2200, 4) + 5
    ).astype(np.uint16)
    # Frame 1: 1 peak
    data[1] = (gauss2d(128, 128, 3000, 4) + 5).astype(np.uint16)
    # Frame 2: empty (just background)
    data[2] = np.full_like(data[2], 5)

    with zarr.ZipStore(str(zip_path), mode="w") as store:
        root = zarr.open_group(store=store, mode="w")
        root.create_dataset(
            "exchange/data", data=data, chunks=(1, NrPixelsZ, NrPixelsY)
        )
        ap = root.require_group("analysis/process/analysis_parameters")
        sp = root.require_group("measurement/process/scan_parameters")

        ap.create_dataset("YCen", data=np.array([128.0]))
        ap.create_dataset("ZCen", data=np.array([128.0]))
        ap.create_dataset("PixelSize", data=np.array([200.0]))
        ap.create_dataset("Lsd", data=np.array([1e6]))
        ap.create_dataset("Wavelength", data=np.array([0.18]))
        ap.create_dataset("RhoD", data=np.array([NrPixelsY * 200.0]))
        ap.create_dataset("Width", data=np.array([10000.0]))
        ap.create_dataset("DoFullImage", data=np.array([1], dtype=np.int32))
        ap.create_dataset("RingThresh", data=np.array([[1, 50.0]]))
        ap.create_dataset("MinNrPx", data=np.array([3], dtype=np.int32))
        ap.create_dataset("MaxNrPx", data=np.array([10000], dtype=np.int32))
        ap.create_dataset("MaxNPeaks", data=np.array([20], dtype=np.int32))
        ap.create_dataset("UpperBoundThreshold", data=np.array([14000.0]))
        ap.create_dataset("ResultFolder", data=np.bytes_(str(tmpdir).encode()))

        sp.create_dataset("start", data=np.array([0.0]))
        sp.create_dataset("step", data=np.array([1.0]))
        sp.create_dataset("doPeakFit", data=np.array([1], dtype=np.int32))

    try:
        yield zip_path
    finally:
        import shutil
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


@pytest.fixture
def out_tmpdir(tmp_path) -> Path:
    """Per-test scratch directory."""
    return tmp_path
