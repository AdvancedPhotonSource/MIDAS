"""Unit tests for midas_pipeline.recon.fbp.

We test the FBP wrapper end-to-end by:
1. Generating a disk phantom + sinogram via the torch-free forward_project.
2. Running ``fbp_recon`` (shells out to MIDAS_TOMO).
3. Asserting RMSE recovery is within a generous threshold.

The MIDAS_TOMO binary must be available on disk; we skip the test if
the importer cannot locate it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.recon import forward_project
from midas_pipeline.recon.fbp import _load_run_tomo_from_sinos, fbp_recon


def _tomo_binary_available() -> bool:
    try:
        _load_run_tomo_from_sinos()
    except ImportError:
        return False
    candidates = [
        Path(os.path.expanduser("~/opt/MIDAS/TOMO/bin/MIDAS_TOMO")),
        Path("/Users/hsharma/opt/MIDAS/build/bin/MIDAS_TOMO"),
    ]
    return any(c.exists() for c in candidates)


pytestmark = pytest.mark.skipif(
    not _tomo_binary_available(),
    reason="MIDAS_TOMO binary not available on this machine",
)


def _disk_phantom(N: int, radius: float = 0.3) -> np.ndarray:
    coord = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(coord, coord)
    return ((xx ** 2 + yy ** 2) < radius ** 2).astype(np.float64)


def test_fbp_recon_disk_phantom(tmp_path):
    N = 16
    phantom = _disk_phantom(N)
    angles = np.linspace(0.0, 180.0, 60, endpoint=False)
    sino = forward_project(phantom, angles)
    recon = fbp_recon(sino, angles, tmp_path / "tomo", n_scans=N)
    assert recon.shape == (N, N)

    # Normalize and compare. FBP gives signed values, so positive part only.
    p = phantom / max(phantom.max(), 1e-12)
    r_pos = np.maximum(recon, 0)
    r = r_pos / max(r_pos.max(), 1e-12)
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    # FBP on a small grid is approximate; allow generous tolerance.
    assert rmse < 0.5, f"FBP RMSE too high: {rmse:.3f}"
