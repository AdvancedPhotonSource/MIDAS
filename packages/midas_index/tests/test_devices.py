"""Device / dtype variation smoke tests.

Verifies the pipeline runs end-to-end on:
  - cpu/float64  (the parity reference)
  - cpu/float32  (FP32 path on CPU)
  - mps/float32  (Apple Silicon dev convenience; skipif unavailable)
  - cuda/float32 (skipif unavailable)
  - cuda/float64 (skipif unavailable)

Doesn't assert numerical parity — just that the code paths are reachable
on each device with the documented dtype defaults.
"""

import math

import numpy as np
import pytest
import torch

from midas_index import Indexer, IndexerParams
from midas_index.io import build_bin_index


def _toy_inputs():
    p = IndexerParams()
    p.Distance = 1_000_000.0
    p.Wavelength = 0.172979
    p.Rsample = 100.0
    p.Hbeam = 100.0
    p.px = 200.0
    p.SpaceGroup = 225
    p.LatticeConstant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.StepsizePos = 50.0
    p.StepsizeOrient = 5.0
    p.MarginOme = 1.0
    p.MarginRad = 5000.0
    p.MarginRadial = 1000.0
    p.MarginEta = 100.0
    p.EtaBinSize = 1.0
    p.OmeBinSize = 1.0
    p.ExcludePoleAngle = 1.0
    p.MinMatchesToAcceptFrac = 0.3
    p.RingNumbers = [1]
    p.RingRadii = {1: 56000.0}
    p.OmegaRanges = [(-180.0, 180.0)]
    p.BoxSizes = [(-2_000_000.0, 2_000_000.0, -2_000_000.0, 2_000_000.0)]
    p.UseFriedelPairs = 0
    p.OutputFolder = "."

    obs = np.array(
        [[0.0, 56000.0, 0.0, 56000.0, 1.0, 1.0, 0.0, 1.5, 0.0]],
        dtype=np.float64,
    )
    bin_data, bin_ndata = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1,
    )
    a = 4.08
    one_over_a = 1.0 / a
    hkls_real = np.array(
        [[one_over_a, one_over_a, one_over_a, 1.0,
          a / math.sqrt(3), 0.05, 56000.0]],
        dtype=np.float64,
    )
    hkls_int = np.array([[1, 1, 1, 1]], dtype=np.int64)
    spot_ids = np.array([1], dtype=np.int64)
    return p, obs, bin_data, bin_ndata, hkls_real, hkls_int, spot_ids


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_pipeline_cpu_dtype_variations(dtype):
    p, obs, data, ndata, hkls_real, hkls_int, spot_ids = _toy_inputs()
    indexer = Indexer(p, device="cpu", dtype=dtype)
    indexer.load_observations(
        cwd=".", spots=obs, bins=(data, ndata),
        hkls=(hkls_real, hkls_int), spot_ids=spot_ids,
    )
    res = indexer.run(block_nr=0, n_blocks=1, n_spots_to_index=1)
    assert isinstance(res.seeds, list)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
@pytest.mark.mps
@pytest.mark.slow
def test_pipeline_runs_on_mps_float32():
    p, obs, data, ndata, hkls_real, hkls_int, spot_ids = _toy_inputs()
    indexer = Indexer(p, device="mps", dtype="float32")
    indexer.load_observations(
        cwd=".", spots=obs, bins=(data, ndata),
        hkls=(hkls_real, hkls_int), spot_ids=spot_ids,
    )
    res = indexer.run(block_nr=0, n_blocks=1, n_spots_to_index=1)
    assert isinstance(res.seeds, list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_pipeline_runs_on_cuda(dtype):
    p, obs, data, ndata, hkls_real, hkls_int, spot_ids = _toy_inputs()
    indexer = Indexer(p, device="cuda", dtype=dtype)
    indexer.load_observations(
        cwd=".", spots=obs, bins=(data, ndata),
        hkls=(hkls_real, hkls_int), spot_ids=spot_ids,
    )
    res = indexer.run(block_nr=0, n_blocks=1, n_spots_to_index=1)
    assert isinstance(res.seeds, list)
