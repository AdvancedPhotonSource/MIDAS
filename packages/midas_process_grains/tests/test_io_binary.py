"""IO smoke tests on the synthetic ``tiny_run_dir`` fixture."""

from __future__ import annotations

import numpy as np

from midas_process_grains.io import (
    BinaryInputs,
    read_all,
    read_fit_best,
    read_index_best,
    read_index_best_full,
    read_key,
    read_orient_pos_fit,
    read_process_key,
)
from midas_process_grains.io.binary import ORIENT_POS_FIT_LAYOUT


def test_read_orient_pos_fit_shape(tiny_run_dir):
    arr = read_orient_pos_fit(tiny_run_dir)
    assert arr.shape == (3, 27)
    assert arr.dtype == np.float64


def test_orient_pos_fit_layout_extracts_orient_mat(tiny_run_dir):
    arr = read_orient_pos_fit(tiny_run_dir)
    # Identity OM was written
    om = arr[0, ORIENT_POS_FIT_LAYOUT["orient_mat"]].reshape(3, 3)
    np.testing.assert_allclose(om, np.eye(3))


def test_orient_pos_fit_layout_position(tiny_run_dir):
    arr = read_orient_pos_fit(tiny_run_dir)
    pos = arr[1, ORIENT_POS_FIT_LAYOUT["position"]]
    np.testing.assert_allclose(pos, [10.0, 20.0, 30.0])


def test_orient_pos_fit_layout_lattice(tiny_run_dir):
    arr = read_orient_pos_fit(tiny_run_dir)
    lat = arr[2, ORIENT_POS_FIT_LAYOUT["lattice"]]
    np.testing.assert_allclose(lat, [3.6, 3.6, 3.6, 90.0, 90.0, 90.0])


def test_orient_pos_fit_layout_internal_angle(tiny_run_dir):
    arr = read_orient_pos_fit(tiny_run_dir)
    ia = arr[2, ORIENT_POS_FIT_LAYOUT["internal_ang"]]
    assert ia == 0.03


def test_read_key(tiny_run_dir):
    key = read_key(tiny_run_dir)
    assert key.shape == (3, 2)
    assert key.dtype == np.int32
    np.testing.assert_array_equal(key[:, 0], 1)             # all alive
    np.testing.assert_array_equal(key[:, 1], [4, 5, 3])     # NrIDsPerID


def test_read_process_key_returns_spot_ids(tiny_run_dir):
    pk = read_process_key(tiny_run_dir)
    assert pk.shape == (3, 5000)
    np.testing.assert_array_equal(pk[0, :4], [101, 102, 103, 104])
    np.testing.assert_array_equal(pk[1, :5], [201, 202, 203, 204, 205])
    assert pk[0, 4] == 0      # zero-padded after the matched SpotIDs


def test_read_index_best(tiny_run_dir):
    ib = read_index_best(tiny_run_dir)
    assert ib.shape == (3, 15)
    assert ib[0, 14] == 4.0   # n_matches matches Key.bin
    assert ib[1, 14] == 5.0


def test_read_index_best_full_spotid_column(tiny_run_dir):
    ibf = read_index_best_full(tiny_run_dir)
    assert ibf.shape == (3, 5000, 2)
    np.testing.assert_array_equal(ibf[0, :4, 0], [101, 102, 103, 104])
    np.testing.assert_array_equal(ibf[1, :5, 0], [201, 202, 203, 204, 205])


def test_read_fit_best_spotid_column(tiny_run_dir):
    fb = read_fit_best(tiny_run_dir)
    assert fb.shape == (3, 5000, 22)
    np.testing.assert_array_equal(fb[0, :4, 0], [101, 102, 103, 104])


def test_read_all_bundles_everything(tiny_run_dir):
    inputs = read_all(tiny_run_dir)
    assert isinstance(inputs, BinaryInputs)
    assert inputs.n_seeds == 3
    assert inputs.orient_pos_fit is not None
    assert inputs.key is not None
    assert inputs.process_key is not None
    assert inputs.index_best is not None
    assert inputs.index_best_full is not None
    assert inputs.fit_best is not None


def test_read_all_can_skip_optional_files(tiny_run_dir):
    inputs = read_all(
        tiny_run_dir,
        require_fit_best=False,
        require_index_best_full=False,
    )
    assert inputs.fit_best is None
    assert inputs.index_best_full is None
