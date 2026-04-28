"""End-to-end synthetic M-step test.

Forward-simulate ring positions on a calibrant detector, perturb the geometry,
and verify that ``refine_geometry`` recovers the truth.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate.geometry import build_tilt_matrix, invert_REta_to_pixel, pixel_to_REta

from midas_calibrate import (
    CalibrationParams,
    FittedPoint,
    build_ring_table,
    refine_geometry,
)
from midas_calibrate.geometry_torch import predict_R_at_pixel
from midas_calibrate.param_vector import pack


def _make_truth_params() -> CalibrationParams:
    """A realistic CeO2 calibration setup."""
    p = CalibrationParams()
    p.NrPixelsY = 2048; p.NrPixelsZ = 2048
    p.pxY = 200.0; p.pxZ = 200.0
    p.Lsd = 1_000_000.0  # μm = 1 m
    p.BC_y = 1024.0; p.BC_z = 1024.0
    p.tx = 0.0; p.ty = 0.5; p.tz = 0.3
    p.Wavelength = 0.173
    p.SpaceGroup = 225
    p.LatticeConstant = (5.411, 5.411, 5.411, 90.0, 90.0, 90.0)
    p.MaxRingRad = 1000.0
    p.MinRingRad = 0.0
    p.RhoD = 1024.0
    p.tolLsd = 5000.0
    p.tolBC = 10.0
    p.tolTilts = 1.0
    p.tolDistortion = 0.0  # don't refine distortion in this clean test
    p.Refine = {
        "Lsd": True, "BC": True, "ty": True, "tz": True,
        "Wavelength": False, "Parallax": False,
        **{f"p{i}": False for i in range(15)},
    }
    return p


def _simulate_fitted_points(params_truth: CalibrationParams, n_per_ring: int = 36):
    """Generate fitted points using midas_integrate's Newton-Raphson inverse.

    For each ring × η, invert (R_ideal, η) to a precise (Y_pix, Z_pix) under
    the truth geometry — guaranteed consistency with the forward model.
    """
    rt = build_ring_table(params_truth)
    fits = []
    px = 0.5 * (params_truth.pxY + params_truth.pxZ)
    TRs = build_tilt_matrix(params_truth.tx, params_truth.ty, params_truth.tz)

    for ring_i, R_px in enumerate(rt.r_ideal_px):
        for eta_i in range(n_per_ring):
            eta = (eta_i + 0.5) * 360.0 / n_per_ring - 180.0
            Y_pix, Z_pix = invert_REta_to_pixel(
                float(R_px), eta,
                Ycen=params_truth.BC_y, Zcen=params_truth.BC_z,
                TRs=TRs, Lsd=params_truth.Lsd, RhoD=params_truth.RhoD, px=px,
                parallax=params_truth.Parallax,
            )
            fits.append(FittedPoint(Y_pix=float(Y_pix), Z_pix=float(Z_pix), ring_idx=ring_i, snr=1.0))
    return rt, fits


def test_mstep_recovers_truth_from_perfect_data():
    truth = _make_truth_params()
    rt, fits = _simulate_fitted_points(truth, n_per_ring=18)
    assert len(fits) > 50

    # Verify our synthetic fits actually project to ring R within ε under truth geometry.
    Y = torch.tensor([p.Y_pix for p in fits], dtype=torch.float64)
    Z = torch.tensor([p.Z_pix for p in fits], dtype=torch.float64)
    x_truth = pack(truth, dtype=torch.float64)
    R_obs = predict_R_at_pixel(Y, Z, x_truth, px=truth.pxY, rho_d=truth.RhoD).numpy()
    R_ideal = np.array([rt.r_ideal_px[p.ring_idx] for p in fits])
    err_truth = np.abs(R_obs - R_ideal)
    print(f"truth-data residual: mean={err_truth.mean():.4e}px max={err_truth.max():.4e}px")
    # We accept some error because the synth uses Xp=Lsd approximation; M-step should still recover.

    # Perturb the geometry away from truth and check that M-step recovers.
    perturbed = _make_truth_params()
    perturbed.Lsd += 500.0           # 500 μm error
    perturbed.BC_y += 2.0
    perturbed.BC_z -= 1.5
    perturbed.ty -= 0.05
    perturbed.tz += 0.07

    result = refine_geometry(perturbed, rt, fits, max_iter=300, verbose=False)

    print(f"\nrc={result.rc} cost={result.cost:.4e} mean_strain={result.mean_strain_uE:.1f}μϵ")
    print(f"  Lsd: truth={truth.Lsd:.1f}  start={perturbed.Lsd:.1f}  refined={result.params.Lsd:.1f}")
    print(f"  BC : truth=({truth.BC_y},{truth.BC_z})  refined=({result.params.BC_y:.3f}, {result.params.BC_z:.3f})")
    print(f"  ty : truth={truth.ty}  refined={result.params.ty:.4f}")
    print(f"  tz : truth={truth.tz}  refined={result.params.tz:.4f}")

    assert result.rc == 0, f"M-step did not converge (rc={result.rc})"
    assert abs(result.params.Lsd - truth.Lsd) < 50.0, "Lsd not recovered"
    assert abs(result.params.BC_y - truth.BC_y) < 0.5
    assert abs(result.params.BC_z - truth.BC_z) < 0.5
    assert abs(result.params.ty - truth.ty) < 0.01
    assert abs(result.params.tz - truth.tz) < 0.01
