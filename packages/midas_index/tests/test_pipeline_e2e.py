"""End-to-end synthetic pipeline smoke test.

Constructs a minimal in-memory dataset (1 grain, 1 ring, 2 spots), runs the
full Indexer.run() path, and checks the output shape. Doesn't validate
parity against C — that lives in test_regression_vs_c.py (P7).
"""

import math

import numpy as np
import torch

from midas_index import Indexer, IndexerParams


def _build_minimal_synthetic():
    """Construct just enough to drive the pipeline without errors."""
    p = IndexerParams()
    p.Distance = 1_000_000.0
    p.Wavelength = 0.172979
    p.Rsample = 200.0
    p.Hbeam = 200.0
    p.px = 200.0
    p.SpaceGroup = 225
    p.LatticeConstant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.StepsizePos = 100.0     # coarse → small position grid
    p.StepsizeOrient = 30.0   # coarse → small orientation grid
    p.MarginOme = 1.0
    p.MarginRad = 200.0
    p.MarginRadial = 200.0
    p.MarginEta = 1.0
    p.EtaBinSize = 1.0        # coarse for fewer bins
    p.OmeBinSize = 1.0
    p.ExcludePoleAngle = 5.0
    p.MinMatchesToAcceptFrac = 0.5
    p.RingNumbers = [1]
    p.RingRadii = {1: 56000.0}
    p.OmegaRanges = [(-180.0, 180.0)]
    p.BoxSizes = [(-1_500_000.0, 1_500_000.0, -1_500_000.0, 1_500_000.0)]
    p.UseFriedelPairs = 0
    p.OutputFolder = "."
    return p


def _toy_obs_and_bins():
    # 1 observed spot at omega=0, eta=0, ring 1.
    obs = np.array(
        [[0.0, 56000.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.5 / 2, 0.0]],
        dtype=np.float64,
    )
    # Build a flat (Data.bin, nData.bin) pair big enough to hold one bin.
    n_eta = math.ceil(360.0 / 1.0)
    n_ome = math.ceil(360.0 / 1.0)
    n_bins = 1 * n_eta * n_ome    # 1 ring
    ndata = np.zeros(2 * n_bins, dtype=np.int32)
    # Bin index for (ring=1, eta=0, omega=0): pos = 0*n_eta*n_ome + 180*n_ome + 180
    pos = 180 * n_ome + 180
    ndata[2 * pos] = 1
    ndata[2 * pos + 1] = 0
    data = np.array([0], dtype=np.int32)
    return obs, data, ndata


def _toy_hkls():
    hkls_real = np.array(
        [[0.184, 0.184, 0.184, 1.0, 3.124, 0.05, 56000.0]],
        dtype=np.float64,
    )
    hkls_int = np.array([[1, 1, 1, 1]], dtype=np.int64)
    return hkls_real, hkls_int


def test_pipeline_runs_end_to_end():
    p = _build_minimal_synthetic()
    obs, data, ndata = _toy_obs_and_bins()
    hkls_real, hkls_int = _toy_hkls()
    spot_ids = np.array([1], dtype=np.int64)

    indexer = Indexer(p, device="cpu", dtype="float64")
    indexer.load_observations(
        cwd=".",
        spots=obs,
        bins=(data, ndata),
        hkls=(hkls_real, hkls_int),
        spot_ids=spot_ids,
    )
    result = indexer.run(block_nr=0, n_blocks=1, n_spots_to_index=1, num_procs=1)
    # Pipeline ran without errors. Whether a seed result was produced depends
    # on whether the (very coarse) orientation grid found any match — we just
    # assert the structure is well-formed.
    assert result.block_nr == 0
    assert result.n_blocks == 1
    assert isinstance(result.seeds, list)


def test_pipeline_handles_missing_spot_id():
    p = _build_minimal_synthetic()
    obs, data, ndata = _toy_obs_and_bins()
    hkls_real, hkls_int = _toy_hkls()
    spot_ids = np.array([999], dtype=np.int64)  # not in obs

    indexer = Indexer(p, device="cpu", dtype="float64")
    indexer.load_observations(
        cwd=".",
        spots=obs, bins=(data, ndata), hkls=(hkls_real, hkls_int),
        spot_ids=spot_ids,
    )
    result = indexer.run(block_nr=0, n_blocks=1, n_spots_to_index=1)
    assert result.seeds == []
