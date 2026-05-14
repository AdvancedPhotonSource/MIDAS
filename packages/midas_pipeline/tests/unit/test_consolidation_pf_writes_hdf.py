"""``microstructure.hdf`` structural test.

Verifies the consolidated H5 file has the legacy two datasets (microstr
and images) with the expected shapes, dtypes, and Header attributes.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from midas_pipeline.stages.consolidation_pf import (
    IMAGES_HEADER,
    MICROSTR_HEADER,
    N_COLS,
    consolidate_pf,
)

from .test_consolidation_pf_synthetic import _setup_fixture


def test_microstructure_hdf_has_microstr_dataset(tmp_path):
    layer_dir = _setup_fixture(tmp_path)
    result = consolidate_pf(layer_dir, n_grains=2, n_scans=2, space_group=225)
    h5_path = Path(result.microstructure_hdf)
    assert h5_path.exists()
    with h5py.File(h5_path, "r") as f:
        assert "microstr" in f
        assert "images" in f
        micstr = f["microstr"][...]
        assert micstr.shape == (4, N_COLS)
        assert micstr.dtype == np.float64
        # Header attribute should match the byte-canonical legacy header.
        header = f["microstr"].attrs["Header"]
        if isinstance(header, bytes):
            header = header.decode()
        assert header == MICROSTR_HEADER


def test_microstructure_hdf_images_shape(tmp_path):
    layer_dir = _setup_fixture(tmp_path, n_scans=2)
    result = consolidate_pf(layer_dir, n_grains=2, n_scans=2, space_group=225)
    with h5py.File(Path(result.microstructure_hdf), "r") as f:
        imgs = f["images"][...]
        assert imgs.shape == (23, 2, 2)
        assert imgs.dtype == np.float64
        header = f["images"].attrs["Header"]
        if isinstance(header, bytes):
            header = header.decode()
        assert header == IMAGES_HEADER


def test_microstructure_hdf_images_values_not_all_nan(tmp_path):
    """All 4 voxels populated → none of the images cells should be NaN."""
    layer_dir = _setup_fixture(tmp_path, n_scans=2)
    result = consolidate_pf(layer_dir, n_grains=2, n_scans=2, space_group=225)
    with h5py.File(Path(result.microstructure_hdf), "r") as f:
        imgs = f["images"][...]
    assert np.isfinite(imgs).all()
