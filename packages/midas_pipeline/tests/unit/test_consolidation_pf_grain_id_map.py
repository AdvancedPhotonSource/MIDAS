"""Grain-ID map (``Full_recon_max_project_grID.tif``) wiring test.

The PF reconstruct stage produces ``Recons/Full_recon_max_project_grID.tif``
upstream of consolidation. Consolidation consumes it (does not regenerate
it) and surfaces its path in the result. We assert the path is reported
and the file shape/dtype are preserved when present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from midas_pipeline.stages.consolidation_pf import consolidate_pf

from .test_consolidation_pf_synthetic import _setup_fixture


def test_grain_id_map_path_reported_when_present(tmp_path):
    layer_dir = _setup_fixture(tmp_path, n_scans=2)
    # Synthesize the grain-ID map produced by the reconstruct stage.
    n_scans = 2
    grain_map = np.array([[0, 0], [1, 1]], dtype=np.int32)
    recons = layer_dir / "Recons"
    recons.mkdir(exist_ok=True)
    Image.fromarray(grain_map).save(recons / "Full_recon_max_project_grID.tif")

    result = consolidate_pf(layer_dir, n_grains=2,
                            n_scans=n_scans, space_group=225)
    assert result.full_recon_max_project_grid_tif != ""
    tif_path = Path(result.full_recon_max_project_grid_tif)
    assert tif_path.exists()

    # Reload and verify shape + dtype preserved.
    with Image.open(tif_path) as im:
        arr = np.array(im)
    assert arr.shape == (n_scans, n_scans)
    assert arr.dtype == np.int32
    np.testing.assert_array_equal(arr, grain_map)


def test_grain_id_map_absent_returns_empty_path(tmp_path):
    """When the reconstruct stage didn't run, the tif path should be ''."""
    layer_dir = _setup_fixture(tmp_path, n_scans=2)
    # Intentionally do NOT create the tif.
    result = consolidate_pf(layer_dir, n_grains=2,
                            n_scans=2, space_group=225)
    assert result.full_recon_max_project_grid_tif == ""
