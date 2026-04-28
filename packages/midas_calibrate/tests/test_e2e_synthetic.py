"""End-to-end synthetic calibration test.

Forward-simulate a CeO2 calibrant image at known geometry, perturb the geometry
seed, run the full E↔M alternating engine, and verify the truth is recovered.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_integrate.geometry import build_tilt_matrix, pixel_to_REta

from midas_calibrate import (
    CalibrationParams,
    autocalibrate,  # noqa: F401  (exported below)
    build_ring_table,
)
from midas_calibrate.orchestrator import autocalibrate


def _make_truth() -> CalibrationParams:
    p = CalibrationParams()
    p.NrPixelsY = 1024; p.NrPixelsZ = 1024     # smaller for test speed
    p.pxY = 200.0; p.pxZ = 200.0
    p.Lsd = 1_000_000.0
    p.BC_y = 512.0; p.BC_z = 512.0
    p.tx = 0.0; p.ty = 0.4; p.tz = 0.25
    p.Wavelength = 0.173
    p.SpaceGroup = 225
    p.LatticeConstant = (5.411, 5.411, 5.411, 90.0, 90.0, 90.0)
    p.MaxRingRad = 480.0
    p.MinRingRad = 0.0
    p.RhoD = 512.0
    p.Width = 1500.0      # wider per-ring window for stable centroiding
    p.EtaBinSize = 10.0   # 36 eta bins
    p.RBinSize = 1.0
    p.nIterations = 4
    p.RemoveOutliersBetweenIters = False
    p.SNRMin = 1.5
    p.tolLsd = 5000.0; p.tolBC = 8.0; p.tolTilts = 1.0
    p.tolDistortion = 0.0
    p.Refine = {
        "Lsd": True, "BC": True, "ty": True, "tz": True,
        "Wavelength": False, "Parallax": False,
        **{f"p{i}": False for i in range(15)},
    }
    return p


def _simulate_image(params: CalibrationParams, ring_thickness_px: float = 1.5) -> np.ndarray:
    """Render a 2D image with bright Gaussian rings on a noisy background."""
    rt = build_ring_table(params)
    NY, NZ = params.NrPixelsY, params.NrPixelsZ
    px = 0.5 * (params.pxY + params.pxZ)
    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)

    # For each pixel, compute its R under truth geometry.
    Y_grid, Z_grid = np.meshgrid(np.arange(NY, dtype=np.float64),
                                  np.arange(NZ, dtype=np.float64))
    R_pix, _ = pixel_to_REta(
        Y_grid, Z_grid,
        Ycen=params.BC_y, Zcen=params.BC_z, TRs=TRs,
        Lsd=params.Lsd, RhoD=params.RhoD, px=px, parallax=params.Parallax,
    )

    img = np.full(R_pix.shape, 50.0, dtype=np.float64)        # background
    rng = np.random.default_rng(0)
    img += rng.normal(0, 5.0, size=img.shape)                  # noise

    for r_ideal in rt.r_ideal_px:
        # Add a ring as a Gaussian centered at r_ideal with σ = ring_thickness_px.
        I_amp = 1000.0 / (1.0 + r_ideal / 100.0)               # falloff
        img += I_amp * np.exp(-0.5 * ((R_pix - r_ideal) / ring_thickness_px) ** 2)
    return img


def test_e2e_synthetic_recovery():
    truth = _make_truth()
    image = _simulate_image(truth)

    # Perturb starting params
    seed = _make_truth()
    seed.Lsd += 300.0
    seed.BC_y += 1.5
    seed.BC_z -= 1.0
    seed.ty -= 0.05
    seed.tz += 0.06

    result = autocalibrate(seed, image, verbose=True)

    final = result.history[-1]
    print(f"\n=== Recovery vs truth ===")
    print(f"  truth Lsd={truth.Lsd}  recovered={result.params.Lsd:.2f}  Δ={result.params.Lsd-truth.Lsd:+.2f} μm")
    print(f"  truth BC=({truth.BC_y},{truth.BC_z})  rec=({result.params.BC_y:.4f},{result.params.BC_z:.4f})")
    print(f"  truth ty={truth.ty}  recovered={result.params.ty:.5f}")
    print(f"  truth tz={truth.tz}  recovered={result.params.tz:.5f}")
    print(f"  final mean_strain = {final.mean_strain_uE:.1f} μϵ")

    # Looser tolerances than M-step-only test because the centroid extraction is approximate.
    assert abs(result.params.Lsd - truth.Lsd) < 200.0, "Lsd recovery insufficient"
    assert abs(result.params.BC_y - truth.BC_y) < 1.0
    assert abs(result.params.BC_z - truth.BC_z) < 1.0
    assert abs(result.params.ty - truth.ty) < 0.05
    assert abs(result.params.tz - truth.tz) < 0.05
    assert final.mean_strain_uE < 500.0, f"final strain {final.mean_strain_uE:.1f} μϵ exceeds 500"
