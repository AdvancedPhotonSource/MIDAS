"""Tests for io.bins_builder."""

import math

import numpy as np

from midas_index.io import build_bin_index


def _make_obs(rows):
    return np.asarray(rows, dtype=np.float64)


def test_build_bin_index_single_spot_offsets_to_zero():
    obs = _make_obs([
        # y, z, omega, ring_rad, spot_id, ring_nr, eta, ttheta, rad_diff
        (0.0, 0.0, 0.0, 0.0, 0, 1, 0.0, 0.0, 0.0),
    ])
    data, ndata = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1,
    )
    n_eta = 360
    n_ome = 360
    pos = 180 * n_ome + 180     # ring=1, eta=0 -> i_eta=180, omega=0 -> i_ome=180
    assert data.tolist() == [0]
    assert int(ndata[2 * pos]) == 1     # count
    assert int(ndata[2 * pos + 1]) == 0  # offset


def test_build_bin_index_two_spots_same_bin():
    obs = _make_obs([
        (0.0, 0.0, 0.0, 0.0, 100, 1, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 200, 1, 0.0, 0.0, 0.0),
    ])
    data, ndata = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1,
    )
    n_ome = 360
    pos = 180 * n_ome + 180
    assert int(ndata[2 * pos]) == 2
    # Spots 0 and 1 in the bin (both observed-spot row indices)
    offset = int(ndata[2 * pos + 1])
    assert sorted(data[offset:offset + 2].tolist()) == [0, 1]


def test_build_bin_index_two_spots_different_bins():
    obs = _make_obs([
        (0.0, 0.0,   0.0, 0.0, 100, 1,  0.0, 0.0, 0.0),
        (0.0, 0.0, 90.0, 0.0, 200, 2, 45.0, 0.0, 0.0),
    ])
    data, ndata = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=2,
    )
    n_eta, n_ome = 360, 360
    pos1 = 0 * n_eta * n_ome + 180 * n_ome + 180    # ring 1, eta 0, omega 0
    pos2 = 1 * n_eta * n_ome + 225 * n_ome + 270    # ring 2, eta 45, omega 90
    assert int(ndata[2 * pos1]) == 1
    assert int(ndata[2 * pos2]) == 1
    # Each spot lands in its own bin
    off1 = int(ndata[2 * pos1 + 1])
    off2 = int(ndata[2 * pos2 + 1])
    assert int(data[off1]) == 0
    assert int(data[off2]) == 1


def test_build_bin_index_drops_out_of_range_rings():
    obs = _make_obs([
        (0.0, 0.0, 0.0, 0.0, 1, 99, 0.0, 0.0, 0.0),  # ring 99 > n_rings=1
    ])
    data, ndata = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1,
    )
    assert data.size == 0
    assert ndata.sum() == 0


def test_build_bin_index_spreading_replicates_across_margin_bins():
    """With margins set, a single spot should appear in multiple bins."""
    # Spot at (eta=0, omega=0). With omemargin=2.0 and bin size=1.0, the spot
    # should land in iOme = {178, 179, 180, 181, 182} (from omega in [-2, 2]).
    obs = _make_obs([
        (0.0, 56000.0, 0.0, 56000.0, 1, 1, 0.0, 1.5, 0.0),
    ])
    data, ndata = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1,
        margin_eta=10.0, margin_ome=2.0, stepsize_orient=0.0,
        ring_radii={1: 56000.0},
    )
    # Total counts should exceed 1 due to replication.
    assert int(ndata[0::2].sum()) > 1
    # The data array contains repeated row indices.
    assert (data == 0).all()


def test_build_bin_index_no_margins_falls_back_to_single_bin():
    """With margin_eta=margin_ome=stepsize_orient=0, behavior matches the
    non-spreading reference: each spot lands in exactly one bin."""
    obs = _make_obs([
        (0.0, 56000.0, 0.0, 56000.0, 1, 1, 0.0, 1.5, 0.0),
    ])
    data, ndata = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1,
        margin_eta=0.0, margin_ome=0.0, stepsize_orient=0.0,
        ring_radii={1: 56000.0},
    )
    # Single spot in a single bin -> data has one element.
    assert data.size == 1
    assert int(ndata[0::2].sum()) == 1


def test_build_bin_index_offsets_are_cumulative():
    """If bins have varying counts, the offsets must be the running sum."""
    obs = _make_obs([
        (0.0, 0.0,  0.0, 0.0, 1, 1,   0.0, 0.0, 0.0),    # bin A
        (0.0, 0.0,  5.0, 0.0, 2, 1,   0.0, 0.0, 0.0),    # bin B
        (0.0, 0.0,  5.0, 0.0, 3, 1,   0.0, 0.0, 0.0),    # bin B (same)
        (0.0, 0.0, 10.0, 0.0, 4, 1,   0.0, 0.0, 0.0),    # bin C
    ])
    data, ndata = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1,
    )
    assert int(ndata.sum()) > 0
    # Total stored rows must equal total kept obs (4)
    assert data.size == 4
    # And every (count, offset) pair must satisfy offset+count <= data.size
    counts = ndata[0::2]
    offsets = ndata[1::2]
    nonempty = counts > 0
    assert ((offsets[nonempty] + counts[nonempty]) <= data.size).all()
