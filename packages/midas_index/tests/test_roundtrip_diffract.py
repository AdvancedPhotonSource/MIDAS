"""Reverse-roundtrip integration test: forward-simulate via midas-diffract,
then verify the indexer recovers the orientation.

Pipeline:
  1. Pick a known orientation R*, position p*, lattice.
  2. Use the same `IndexerForwardAdapter` we ship to forward-simulate the
     spots that *would be observed* for this grain.
  3. Pack those simulated spots into a fake `Spots.bin` table and build a
     binned index via `build_bin_index`.
  4. Run the full Indexer pipeline.
  5. Check the recovered orientation matches R* up to symmetry-equivalent
     misorientation.

This is a self-contained correctness test that does NOT require the C
indexer or any real experimental data.
"""

import math

import numpy as np
import pytest
import torch

from midas_index import Indexer, IndexerParams
from midas_index.compute.forward_adapter import IndexerForwardAdapter
from midas_index.io import build_bin_index
from midas_stress.orientation import (
    axis_angle_to_orient_mat,
    misorientation_om,
)


def _build_params() -> IndexerParams:
    p = IndexerParams()
    p.Distance = 1_000_000.0
    p.Wavelength = 0.172979
    p.Rsample = 100.0
    p.Hbeam = 100.0
    p.px = 200.0
    p.SpaceGroup = 225
    p.LatticeConstant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.StepsizePos = 100.0
    p.StepsizeOrient = 5.0
    p.MarginOme = 1.0
    p.MarginRad = 5000.0
    p.MarginRadial = 1000.0
    p.MarginEta = 100.0
    p.EtaBinSize = 1.0
    p.OmeBinSize = 1.0
    p.ExcludePoleAngle = 1.0
    p.MinMatchesToAcceptFrac = 0.3
    p.RingNumbers = [1, 2]
    p.RingRadii = {1: 56000.0, 2: 81000.0}
    p.OmegaRanges = [(-180.0, 180.0)]
    p.BoxSizes = [(-2_000_000.0, 2_000_000.0, -2_000_000.0, 2_000_000.0)]
    p.UseFriedelPairs = 0
    p.OutputFolder = "."
    return p


def _build_hkls():
    """A handful of FCC reflections on rings 1 (111) and 2 (200)."""
    # g-vectors in 1/Angstroms (matches midas-diffract's hkls.csv convention).
    a = 4.08
    one_over_a = 1.0 / a
    hkls = [
        # (h, k, l, ring_nr, d, theta_rad, radius)
        (1, -1, -1, 1, a / math.sqrt(3), 0.0, 56000.0),
        (1, 1, 1, 1, a / math.sqrt(3), 0.0, 56000.0),
        (-1, 1, 1, 1, a / math.sqrt(3), 0.0, 56000.0),
        (1, -1, 1, 1, a / math.sqrt(3), 0.0, 56000.0),
        (2, 0, 0, 2, a / 2.0, 0.0, 81000.0),
        (-2, 0, 0, 2, a / 2.0, 0.0, 81000.0),
        (0, 2, 0, 2, a / 2.0, 0.0, 81000.0),
        (0, -2, 0, 2, a / 2.0, 0.0, 81000.0),
        (0, 0, 2, 2, a / 2.0, 0.0, 81000.0),
        (0, 0, -2, 2, a / 2.0, 0.0, 81000.0),
    ]
    # Cartesian g-vectors and Bragg angles for Cu wavelength 0.172979 A.
    wl = 0.172979
    rows_real = []
    rows_int = []
    for h, k, l, ring, d, _, radius in hkls:
        g_cart = (h * one_over_a, k * one_over_a, l * one_over_a)
        d_real = 1.0 / math.sqrt(g_cart[0] ** 2 + g_cart[1] ** 2 + g_cart[2] ** 2)
        sin_th = wl / (2.0 * d_real)
        # Reject rings whose Bragg angle is unphysical at this wavelength.
        if not 0 < sin_th <= 1.0:
            continue
        theta_rad = math.asin(sin_th)
        rows_real.append(
            (g_cart[0], g_cart[1], g_cart[2], float(ring), d_real, theta_rad, radius)
        )
        rows_int.append((h, k, l, ring))
    return (
        np.asarray(rows_real, dtype=np.float64),
        np.asarray(rows_int, dtype=np.int64),
    )


def _simulate_observed_spots(
    R_true: torch.Tensor,
    pos_true: torch.Tensor,
    params: IndexerParams,
    hkls_real: np.ndarray,
    hkls_int: np.ndarray,
) -> np.ndarray:
    """Use IndexerForwardAdapter to produce simulated obs spots in (n,9) layout."""
    adapter = IndexerForwardAdapter(
        params=params,
        hkls_real=torch.as_tensor(hkls_real, dtype=torch.float64),
        hkls_int=torch.as_tensor(hkls_int, dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    theor, valid = adapter.simulate(R_true.unsqueeze(0), pos_true.unsqueeze(0))
    # theor shape (1, K, 14); pull only valid rows.
    theor1 = theor[0]
    valid1 = valid[0]
    rows = []
    spot_id = 1
    for k in range(theor1.shape[0]):
        if not bool(valid1[k]):
            continue
        # Layout cols: 6=omega, 7=eta, 8=theta, 9=ring, 10=yl_disp, 11=zl_disp, 13=rad_diff
        omega = float(theor1[k, 6].item())
        eta = float(theor1[k, 7].item())
        theta = float(theor1[k, 8].item())
        ring = int(theor1[k, 9].item())
        y = float(theor1[k, 10].item())
        z = float(theor1[k, 11].item())
        rad_diff = float(theor1[k, 13].item())
        # Spots.bin layout (IndexerOMP.c:1797 RefRad = obs[3]):
        # [y, z, omega, radial_pos, spot_id, ring, eta, ttheta, rad_diff]
        # Col 3 is the spot's actual sqrt(y² + z²) radial position.
        radial = math.sqrt(y * y + z * z)
        rows.append((y, z, omega, radial, float(spot_id), float(ring), eta, theta * 2, rad_diff))
        spot_id += 1
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.parametrize("axis,angle_deg", [
    ((0.0, 0.0, 1.0), 5.0),
    ((1.0, 0.0, 0.0), 7.0),
    ((1.0, 1.0, 1.0), 11.0),
    ((0.5, 0.7, 0.3), 17.0),
])
def test_reverse_roundtrip_recovers_orientation(axis, angle_deg):
    """Forward-simulate from a known R*, run indexer, recover R* (mod sym)."""
    R_true_np = axis_angle_to_orient_mat(axis, angle_deg)
    R_true = torch.as_tensor(R_true_np, dtype=torch.float64)
    pos_true = torch.zeros(3, dtype=torch.float64)

    params = _build_params()
    hkls_real, hkls_int = _build_hkls()
    obs = _simulate_observed_spots(R_true, pos_true, params, hkls_real, hkls_int)
    if obs.shape[0] < 4:
        pytest.skip(f"forward simulation produced only {obs.shape[0]} spots")

    # Build (Data.bin, nData.bin) for the simulated obs
    n_rings = max(params.RingRadii.keys())
    bin_data, bin_ndata = build_bin_index(
        obs,
        eta_bin_size=params.EtaBinSize,
        ome_bin_size=params.OmeBinSize,
        n_rings=n_rings,
        margin_eta=params.MarginEta,
        margin_ome=params.MarginOme,
        stepsize_orient=params.StepsizeOrient,
        ring_radii=params.RingRadii,
    )
    spot_ids = np.asarray(obs[:, 4], dtype=np.int64)

    # Run the indexer
    indexer = Indexer(params, device="cpu", dtype="float64")
    indexer.load_observations(
        cwd=".", spots=obs, bins=(bin_data, bin_ndata),
        hkls=(hkls_real, hkls_int),
        spot_ids=spot_ids[:1],   # index from just the first seed
    )
    result = indexer.run(block_nr=0, n_blocks=1, n_spots_to_index=1, num_procs=1)

    # We expect at least one seed result; verify recovered orientation
    assert len(result.seeds) >= 1, "indexer produced no seed results"
    seed = result.seeds[0]
    R_rec = seed.best_or_mat.to(torch.float64).reshape(9).cpu().numpy().reshape(3, 3)

    # Symmetry-aware misorientation (cubic group, sg 225)
    angle_rad, _ = misorientation_om(R_true_np.flatten().tolist(),
                                     R_rec.flatten().tolist(),
                                     params.SpaceGroup)
    miso_deg = math.degrees(float(angle_rad))
    # The orientation grid step is 5.0°, so we expect recovery within ~5°.
    assert miso_deg < 6.0, (
        f"recovered orientation deviates by {miso_deg:.3f}° "
        f"(allowed ≤ 6.0°)"
    )
    # And the recovered tuple must have a non-trivial fraction of matches.
    assert seed.n_matches >= 1, "no matches found at recovered orientation"
