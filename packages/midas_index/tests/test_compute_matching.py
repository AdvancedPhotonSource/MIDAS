"""Tests for compute.matching."""

import numpy as np
import pytest
import torch

from midas_index.compute.matching import (
    build_eta_margins,
    build_ome_margins,
    compare_spots,
)


def _make_obs(rows):
    """rows: list of (y, z, omega, ring_radius, spot_id, ring_nr, eta, ttheta, rad_diff)"""
    return torch.tensor(rows, dtype=torch.float64)


def _make_theor(rows):
    """rows: list of 14-element TheorSpots."""
    return torch.tensor(rows, dtype=torch.float64).unsqueeze(0)  # (1, T, 14)


def test_compare_spots_perfect_match():
    # 1 evaluation tuple with 1 theoretical spot, 1 matching observed spot.
    obs = _make_obs([
        # y, z, omega, ring_rad, spot_id, ring_nr, eta, ttheta, rad_diff
        (10.0, 5.0, 1.0, 30000.0, 17, 1, 12.0, 1.5, 0.0),
    ])
    # ndata: 1 bin, count=1, offset=0
    # data: [0]
    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size, ome_bin_size = 0.1, 0.1

    # Compute the bin index for the observed spot — theor must hash to same bin.
    obs_ring_nr = 1
    obs_eta = 12.0
    obs_ome = 1.0
    pos = (
        (obs_ring_nr - 1) * n_eta_bins * n_ome_bins
        + int(np.floor((180 + obs_eta) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + obs_ome) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 1), dtype=torch.int32)
    ndata[2 * pos] = 1
    ndata[2 * pos + 1] = 0
    bin_data = torch.tensor([0], dtype=torch.int32)

    theor = _make_theor([
        # 0 1 2 3 4 5 6:omega 7 8 9:ringnr 10:y 11:z 12:eta 13:rad_diff
        [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1, 10.0, 5.0, 12.0, 0.0],
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)
    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
    )
    assert int(res.n_matches.item()) == 1
    assert int(res.matched_obs_id[0, 0].item()) == 17
    assert res.frac_matches[0].item() == 1.0


def test_compare_spots_no_match_when_far():
    # Theoretical spot far from observed.
    obs = _make_obs([
        (10.0, 5.0, 1.0, 30000.0, 17, 1, 12.0, 1.5, 0.0),
    ])
    # Theor at eta=180 (different bin)
    theor = _make_theor([
        [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1, 10.0, 5.0, 180.0, 0.0],
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)
    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size, ome_bin_size = 0.1, 0.1
    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    # ndata: empty bins
    ndata = torch.zeros(2 * (n_eta_bins * n_ome_bins), dtype=torch.int32)
    bin_data = torch.zeros(0, dtype=torch.int32)

    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
    )
    assert int(res.n_matches.item()) == 0


def test_compare_spots_tie_break_picks_smallest_delta_omega():
    # Two observed spots in the same bin; theor must match the one with
    # smaller |delta omega|.
    obs = _make_obs([
        (10.0, 5.0, 0.6, 30000.0, 100, 1, 12.0, 1.5, 0.0),  # Δω=0.4
        (10.0, 5.0, 1.1, 30000.0, 200, 1, 12.0, 1.5, 0.0),  # Δω=0.1  ← winner
        (10.0, 5.0, 1.5, 30000.0, 300, 1, 12.0, 1.5, 0.0),  # Δω=0.5
    ])
    theor = _make_theor([
        [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1, 10.0, 5.0, 12.0, 0.0],
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)

    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size, ome_bin_size = 0.1, 0.1
    pos = (
        0 * n_eta_bins * n_ome_bins
        + int(np.floor((180 + 12.0) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + 1.0) / ome_bin_size))
    )
    # Make both eta bins resolve to same bin: ome bin width is 0.1, and the
    # observed omegas (0.6, 1.1, 1.5) fall in different ome-bins. Force them
    # into the same bin by using a wider OmeBinSize for this test.
    ome_bin_size = 5.0
    n_ome_bins = 72
    pos = (
        0 * n_eta_bins * n_ome_bins
        + int(np.floor((180 + 12.0) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + 1.0) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 10), dtype=torch.int32)
    ndata[2 * pos] = 3
    ndata[2 * pos + 1] = 0
    bin_data = torch.tensor([0, 1, 2], dtype=torch.int32)

    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=10.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
    )
    assert int(res.matched_obs_id[0, 0].item()) == 200


def test_build_eta_margins_shapes_and_zeros_outside_rings():
    eta = build_eta_margins(
        ring_radii={1: 30000.0, 3: 50000.0},
        margin_eta=5.0, stepsize_orient_deg=1.0,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    assert eta.shape == (500,)
    assert eta[2].item() == 0.0       # ring 2 has no radius
    assert eta[1].item() > 0.0
    assert eta[3].item() > 0.0


def test_build_ome_margins_lut():
    ome = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    assert ome.shape == (181,)
    # endpoints repeat the i==1 value
    assert ome[0].item() == ome[1].item()
    assert ome[180].item() == ome[1].item()


def test_compare_spots_avg_ia_zero_when_perfect_match():
    """A theor and obs spot at exactly the same lab-frame location must yield IA=0."""
    # 1 evaluation tuple, 1 theor spot, 1 obs spot, identical y/z/omega → IA=0.
    obs = _make_obs([
        (10.0, 5.0, 1.0, 30000.0, 17, 1, 12.0, 1.5, 0.0),
    ])
    theor = _make_theor([
        # Theor cols 10/11 = 10/5 (same as obs). col 6 = 1.0 (same omega).
        [0, 0, 0, 0, 10.0, 5.0, 1.0, 0, 0, 1, 10.0, 5.0, 12.0, 0.0],
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)
    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size, ome_bin_size = 0.1, 0.1
    pos = (
        0 * n_eta_bins * n_ome_bins
        + int(np.floor((180 + 12.0) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + 1.0) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 1), dtype=torch.int32)
    ndata[2 * pos] = 1
    ndata[2 * pos + 1] = 0
    bin_data = torch.tensor([0], dtype=torch.int32)

    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
        distance=1_000_000.0,
        pos=torch.zeros((1, 3), dtype=torch.float64),
    )
    assert res.avg_ia[0].item() == pytest.approx(0.0, abs=1e-6)


def test_compare_spots_avg_ia_zero_when_no_matches():
    """No matches -> avg_ia is 0 (no spots contribute)."""
    obs = _make_obs([
        (0.0, 0.0, 0.0, 0.0, 17, 1, 0.0, 0.0, 0.0),
    ])
    theor = _make_theor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 99.0, 0],   # eta=99 → won't match
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)
    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    n_eta_bins, n_ome_bins = 3600, 3600
    ndata = torch.zeros(2 * n_eta_bins * n_ome_bins, dtype=torch.int32)
    bin_data = torch.zeros(0, dtype=torch.int32)
    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=0.1, ome_bin_size=0.1,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
        distance=1_000_000.0,
        pos=torch.zeros((1, 3), dtype=torch.float64),
    )
    assert int(res.n_matches.item()) == 0
    assert res.avg_ia[0].item() == 0.0
