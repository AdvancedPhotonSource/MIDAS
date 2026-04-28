"""Tests for compute.forward_adapter (delegates to midas_diffract.calc_bragg_geometry)."""

import math

import numpy as np
import pytest
import torch

from midas_index.compute.forward_adapter import IndexerForwardAdapter
from midas_index.params import IndexerParams


def _toy_params() -> IndexerParams:
    p = IndexerParams()
    p.Distance = 1_000_000.0
    p.Wavelength = 0.172979
    p.Rsample = 200.0
    p.Hbeam = 200.0
    p.px = 200.0
    p.SpaceGroup = 225
    p.LatticeConstant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.StepsizePos = 5.0
    p.StepsizeOrient = 0.5
    p.MarginOme = 0.5
    p.MarginRad = 200.0
    p.MarginRadial = 200.0
    p.MarginEta = 1.0
    p.EtaBinSize = 0.1
    p.OmeBinSize = 0.1
    p.ExcludePoleAngle = 1.0
    p.MinMatchesToAcceptFrac = 0.6
    p.RingNumbers = [1, 2]
    p.RingRadii = {1: 56000.0, 2: 81000.0}
    p.OmegaRanges = [(-180.0, 180.0)]
    p.BoxSizes = [(-1_500_000.0, 1_500_000.0, -1_500_000.0, 1_500_000.0)]
    return p


def _toy_hkls():
    # Two HKLs on rings 1 and 2 of a fictitious cubic crystal.
    # Layout: [g1, g2, g3, ring_nr, d_spacing, theta_rad, radius]
    hkls_real = torch.tensor(
        [[0.184, 0.184, 0.184, 1.0, 3.124, 0.05, 56000.0],
         [0.213, 0.0,   0.0,   2.0, 2.706, 0.06, 81000.0]],
        dtype=torch.float64,
    )
    hkls_int = torch.tensor(
        [[1, 1, 1, 1],
         [2, 0, 0, 2]],
        dtype=torch.long,
    )
    return hkls_real, hkls_int


def test_adapter_construction():
    params = _toy_params()
    hkls_real, hkls_int = _toy_hkls()
    adapter = IndexerForwardAdapter(
        params=params,
        hkls_real=hkls_real,
        hkls_int=hkls_int,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert adapter.ring_nr_per_hkl.shape == (2,)
    assert adapter.ring_radius_lut[1].item() == pytest.approx(56000.0)
    assert adapter.ring_radius_lut[2].item() == pytest.approx(81000.0)


def test_adapter_simulate_returns_correct_shape():
    params = _toy_params()
    hkls_real, hkls_int = _toy_hkls()
    adapter = IndexerForwardAdapter(
        params=params,
        hkls_real=hkls_real,
        hkls_int=hkls_int,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    N = 3
    R = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(N, 1, 1)
    pos = torch.zeros((N, 3), dtype=torch.float64)
    theor, valid = adapter.simulate(R, pos)
    M = hkls_real.shape[0]
    assert theor.shape == (N, 2 * M, 14)
    assert valid.shape == (N, 2 * M)


def test_adapter_omega_range_gating():
    """Setting a tight OmegaRange should mask out spots outside it."""
    params = _toy_params()
    params.OmegaRanges = [(-1.0, 1.0)]            # very narrow
    params.BoxSizes = [(-1e9, 1e9, -1e9, 1e9)]
    hkls_real, hkls_int = _toy_hkls()
    adapter = IndexerForwardAdapter(
        params=params, hkls_real=hkls_real, hkls_int=hkls_int,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    N = 1
    R = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(N, 1, 1)
    pos = torch.zeros((N, 3), dtype=torch.float64)
    theor, valid = adapter.simulate(R, pos)
    # With a tight OmegaRange, most spots should be masked off
    if int(valid.sum().item()) > 0:
        # All valid spots have omega in [-1, 1]
        omega_kept = theor[..., 6][valid]
        assert (omega_kept.abs() <= 1.0).all()


def test_adapter_position_shifts_yl_zl():
    """A nonzero position should change col 10/11 (yl_disp/zl_disp) but not col 4/5."""
    params = _toy_params()
    hkls_real, hkls_int = _toy_hkls()
    adapter = IndexerForwardAdapter(
        params=params, hkls_real=hkls_real, hkls_int=hkls_int,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    R = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    pos_zero = torch.zeros((1, 3), dtype=torch.float64)
    pos_nonzero = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float64)

    theor0, _ = adapter.simulate(R, pos_zero)
    theor1, _ = adapter.simulate(R, pos_nonzero)

    # Cols 4/5 (no-displacement yl/zl) stay identical
    np.testing.assert_allclose(theor0[..., 4].numpy(), theor1[..., 4].numpy())
    np.testing.assert_allclose(theor0[..., 5].numpy(), theor1[..., 5].numpy())
    # Cols 10/11 differ (because of displacement)
    assert not torch.allclose(theor0[..., 10], theor1[..., 10])
