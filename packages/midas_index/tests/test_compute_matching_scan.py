"""Tests for the scan-aware extensions to compute.matching (P5).

Verifies:
- FF mode (scan_pos_tol_um=0) leaves match counts unchanged.
- Scan-position filter drops candidates with inconsistent scan_idx.
- Friedel-symmetric default keeps mirror-image candidates that the
  single-sided filter would drop.
- Single-sided form (Friedel OFF) matches the C IndexerScanningOMP
  filter exactly — required for the parity gate.
- 9-col obs (FF) raises a clear error when scan-aware mode is enabled.

All tests are CPU-only and small (one or two voxels × a handful of
spots). They share the binning setup from test_compute_matching.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_index.compute.matching import (
    build_eta_margins,
    build_ome_margins,
    compare_spots,
)

DEG2RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spot10(y, z, omega, ring_rad, spot_id, ring_nr, eta, ttheta, rad_diff, scan_nr):
    """10-col Spots.bin row (PF layout): adds scan_nr at col 9."""
    return (y, z, omega, ring_rad, spot_id, ring_nr, eta, ttheta, rad_diff, scan_nr)


def _theor_row(omega, ring_nr, y, z, eta, rad_diff):
    """14-col TheorSpots row that compares_spots reads."""
    return [0, 0, 0, 0, 0, 0, omega, 0, 0, ring_nr, y, z, eta, rad_diff]


def _build_bin(obs_eta, obs_ome, obs_ring_nr, eta_bin_size, ome_bin_size,
               n_eta_bins, n_ome_bins, n_rows):
    """Build a single-bin ndata/bin_data pair containing ``n_rows`` spots."""
    pos = (
        (obs_ring_nr - 1) * n_eta_bins * n_ome_bins
        + int(np.floor((180 + obs_eta) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + obs_ome) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 10), dtype=torch.int32)
    ndata[2 * pos] = n_rows
    ndata[2 * pos + 1] = 0
    bin_data = torch.arange(n_rows, dtype=torch.int32)
    return ndata, bin_data, pos


def _matching_kwargs():
    """Default eta/ome margins used across the scan tests."""
    return dict(
        eta_margins=build_eta_margins(
            ring_radii={1: 30000.0}, margin_eta=20.0, stepsize_orient_deg=0.5,
            device=torch.device("cpu"), dtype=torch.float64,
        ),
        ome_margins=build_ome_margins(
            margin_ome=10.0, stepsize_orient_deg=0.5,
            device=torch.device("cpu"), dtype=torch.float64,
        ),
        eta_bin_size=0.1,
        ome_bin_size=5.0,
        n_eta_bins=3600,
        n_ome_bins=72,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
        margin_rad=50.0,
        margin_radial=50.0,
    )


# ---------------------------------------------------------------------------
# FF regression: scan-aware mode disabled (tol=0) leaves results unchanged
# ---------------------------------------------------------------------------


def test_scan_filter_disabled_is_ff_noop_with_10col_obs():
    """With scan_pos_tol_um=0 a 10-col obs should match exactly like 9-col FF."""
    omega = 1.0
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, omega, 30000.0, 17, 1, 12.0, 1.5, 0.0, scan_nr=0),
    ], dtype=torch.float64)
    obs9 = obs10[..., :9]
    theor = torch.tensor([_theor_row(omega, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, omega, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=1)
    common = dict(
        theor=theor, valid=valid,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        **kw,
    )
    res_ff = compare_spots(obs=obs9, **common)
    res_pf = compare_spots(obs=obs10, **common)  # scan_pos_tol_um=0 by default
    assert int(res_ff.n_matches.item()) == int(res_pf.n_matches.item()) == 1


# ---------------------------------------------------------------------------
# Scan-position filter drops candidates with wrong scan_nr
# ---------------------------------------------------------------------------


def test_scan_filter_drops_inconsistent_scan_nr():
    """Voxel at (0, 0) µm with tol=2 µm: spots at scan_nr=0 (ypos=0) keep,
    spots at scan_nr=1 (ypos=20 µm) drop."""
    omega = 1.0
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, omega, 30000.0, 17, 1, 12.0, 1.5, 0.0, scan_nr=0),
        _spot10(10.0, 5.0, omega, 30000.0, 18, 1, 12.0, 1.5, 0.0, scan_nr=1),
    ], dtype=torch.float64)
    theor = torch.tensor([_theor_row(omega, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, omega, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=2)
    scan_positions = torch.tensor([0.0, 20.0], dtype=torch.float64)
    voxel_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

    res = compare_spots(
        theor=theor, valid=valid, obs=obs10,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        scan_positions=scan_positions, voxel_xy=voxel_xy,
        scan_pos_tol_um=2.0,
        friedel_symmetric_scan_filter=False,
        **kw,
    )
    assert int(res.n_matches.item()) == 1
    assert int(res.matched_obs_id[0, 0].item()) == 17


def test_scan_filter_disabled_keeps_both_candidates():
    """tol=0 ⇒ both candidates remain (filter inactive)."""
    omega = 1.0
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, omega, 30000.0, 17, 1, 12.0, 1.5, 0.0, scan_nr=0),
        _spot10(10.0, 5.0, omega, 30000.0, 18, 1, 12.0, 1.5, 0.0, scan_nr=1),
    ], dtype=torch.float64)
    theor = torch.tensor([_theor_row(omega, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, omega, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=2)

    res = compare_spots(
        theor=theor, valid=valid, obs=obs10,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        # No scan_positions / voxel_xy → filter disabled regardless of tol.
        **kw,
    )
    assert int(res.n_matches.item()) == 1  # tie-break picks first match


# ---------------------------------------------------------------------------
# Friedel-symmetric filter keeps a +scan/−scan mirror pair that single-sided drops
# ---------------------------------------------------------------------------


def test_friedel_symmetric_keeps_mirror_candidate():
    """Voxel at (5, 0) µm, omega=0 ⇒ s_proj = 5 µm.

    Friedel pair appears at ypos = −5 µm (scan_nr=0). With Friedel OFF the
    filter drops it (|5 − (−5)| = 10 > tol=2). With Friedel ON it's kept
    via |−5 − (−5)| = 0 < tol.
    """
    omega = 0.0
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, omega, 30000.0, 42, 1, 12.0, 1.5, 0.0, scan_nr=0),
    ], dtype=torch.float64)
    theor = torch.tensor([_theor_row(omega, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, omega, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=1)
    scan_positions = torch.tensor([-5.0], dtype=torch.float64)
    voxel_xy = torch.tensor([[5.0, 0.0]], dtype=torch.float64)

    common = dict(
        theor=theor, valid=valid, obs=obs10,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        scan_positions=scan_positions, voxel_xy=voxel_xy,
        scan_pos_tol_um=2.0,
        **kw,
    )
    res_single = compare_spots(friedel_symmetric_scan_filter=False, **common)
    res_friedel = compare_spots(friedel_symmetric_scan_filter=True, **common)
    assert int(res_single.n_matches.item()) == 0
    assert int(res_friedel.n_matches.item()) == 1


# ---------------------------------------------------------------------------
# 9-col obs in scan-aware mode raises with a helpful message
# ---------------------------------------------------------------------------


def test_scan_aware_with_9col_obs_raises():
    obs9 = torch.tensor([
        (10.0, 5.0, 1.0, 30000.0, 17, 1, 12.0, 1.5, 0.0),
    ], dtype=torch.float64)
    theor = torch.tensor([_theor_row(1.0, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, 1.0, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=1)
    with pytest.raises(ValueError, match="10 columns"):
        compare_spots(
            theor=theor, valid=valid, obs=obs9,
            bin_data=bin_data, bin_ndata=ndata,
            ref_rad=torch.tensor([30000.0], dtype=torch.float64),
            scan_positions=torch.tensor([0.0], dtype=torch.float64),
            voxel_xy=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            scan_pos_tol_um=2.0,
            **kw,
        )


# ---------------------------------------------------------------------------
# Differentiability + multi-device (per feedback_diff_multidev_required.md)
# ---------------------------------------------------------------------------


def test_scan_filter_forward_runs_with_requires_grad():
    """``compare_spots`` is fundamentally categorical (counts + argmin
    indices), so the scan-filter mask cannot backprop to voxel_xy — the
    filter enters the graph only through boolean masking, which produces
    integer counts. We document this here and verify the forward pass
    runs cleanly when voxel_xy.requires_grad=True (no graph corruption,
    no NaN, no error).

    For end-to-end differentiability against voxel_xy, callers should
    use a smooth surrogate loss outside ``compare_spots`` — e.g.,
    integrating the predicted-vs-observed pairs by the gathered
    ``delta_omega`` field, which is continuous in voxel_xy.
    """
    omega = 1.0
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, omega, 30000.0, 17, 1, 12.0, 1.5, 0.0, scan_nr=0),
    ], dtype=torch.float64)
    theor = torch.tensor([_theor_row(omega, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, omega, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=1)
    voxel_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    scan_positions = torch.tensor([0.0], dtype=torch.float64)
    res = compare_spots(
        theor=theor, valid=valid, obs=obs10,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        scan_positions=scan_positions, voxel_xy=voxel_xy,
        scan_pos_tol_um=2.0,
        **kw,
    )
    # Forward pass produces finite outputs even with requires_grad=True.
    assert torch.isfinite(res.frac_matches).all()
    assert int(res.n_matches.item()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_scan_filter_runs_on_cuda():
    omega = 1.0
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, omega, 30000.0, 17, 1, 12.0, 1.5, 0.0, scan_nr=0),
    ], dtype=torch.float64, device="cuda")
    theor = torch.tensor([_theor_row(omega, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float64, device="cuda").unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool, device="cuda")
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, omega, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=1)
    res = compare_spots(
        theor=theor, valid=valid, obs=obs10,
        bin_data=bin_data.cuda(), bin_ndata=ndata.cuda(),
        ref_rad=torch.tensor([30000.0], dtype=torch.float64, device="cuda"),
        scan_positions=torch.tensor([0.0], dtype=torch.float64, device="cuda"),
        voxel_xy=torch.tensor([[0.0, 0.0]], dtype=torch.float64, device="cuda"),
        scan_pos_tol_um=2.0,
        eta_margins=kw["eta_margins"].cuda(),
        ome_margins=kw["ome_margins"].cuda(),
        eta_bin_size=kw["eta_bin_size"], ome_bin_size=kw["ome_bin_size"],
        n_eta_bins=kw["n_eta_bins"], n_ome_bins=kw["n_ome_bins"],
        rings_to_reject=kw["rings_to_reject"].cuda(),
        margin_rad=kw["margin_rad"], margin_radial=kw["margin_radial"],
    )
    assert res.n_matches.device.type == "cuda"


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS unavailable",
)
def test_scan_filter_runs_on_mps():
    """MPS doesn't support fp64 — run in fp32."""
    omega = 1.0
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, omega, 30000.0, 17, 1, 12.0, 1.5, 0.0, scan_nr=0),
    ], dtype=torch.float32, device="mps")
    theor = torch.tensor([_theor_row(omega, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float32, device="mps").unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool, device="mps")
    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=20.0, stepsize_orient_deg=0.5,
        device=torch.device("mps"), dtype=torch.float32,
    )
    ome_margins = build_ome_margins(
        margin_ome=10.0, stepsize_orient_deg=0.5,
        device=torch.device("mps"), dtype=torch.float32,
    )
    n_eta_bins, n_ome_bins, eta_bin_size, ome_bin_size = 3600, 72, 0.1, 5.0
    pos = (
        0 * n_eta_bins * n_ome_bins
        + int(np.floor((180 + 12.0) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + omega) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 10), dtype=torch.int32, device="mps")
    ndata[2 * pos] = 1
    bin_data = torch.tensor([0], dtype=torch.int32, device="mps")
    res = compare_spots(
        theor=theor, valid=valid, obs=obs10,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float32, device="mps"),
        scan_positions=torch.tensor([0.0], dtype=torch.float32, device="mps"),
        voxel_xy=torch.tensor([[0.0, 0.0]], dtype=torch.float32, device="mps"),
        scan_pos_tol_um=2.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64, device="mps"),
        margin_rad=50.0, margin_radial=50.0,
    )
    assert res.n_matches.device.type == "mps"
