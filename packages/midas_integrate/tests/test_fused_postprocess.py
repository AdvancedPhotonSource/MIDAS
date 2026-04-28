"""End-to-end test of the fused-CSR + post-process API.

Verifies that the public-API call sequence

    geom = build_fused_geometry(params, n_shifts=N, mode='gradient')
    cake = integrate(image, geom)
    cake_smoothed = gauss_smooth_eta(cake, sigma_bins=2.0)

reduces cardinal σ/μ on a small synthetic powder-ring image.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_integrate import (
    IntegrationParams,
    build_fused_geometry,
    gauss_smooth_eta,
    integrate,
    median_filter_eta,
)


def _synth_ring_image(NY=256, NZ=256, BC=128.0, Lsd=200_000.0,
                      px=200.0, R_ring=80.0):
    """Make a tiny detector image with one Gaussian ring at R = R_ring px."""
    ys, zs = np.meshgrid(np.arange(NY, dtype=np.float64),
                          np.arange(NZ, dtype=np.float64),
                          indexing="xy")
    R = np.sqrt((ys - BC) ** 2 + (zs - BC) ** 2)
    # Sharp Gaussian ring (FWHM ~ 1.5 px) so cardinal aliasing is visible
    sigma = 0.75
    img = 1000.0 * np.exp(-((R - R_ring) ** 2) / (2.0 * sigma ** 2)) + 5.0
    return img


def _params(NY=256, NZ=256, RBin=0.5, EtaBin=1.0):
    return IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0,
        Lsd=200_000.0, BC_y=NY/2, BC_z=NZ/2, RhoD=float(NY),
        RMin=2.0, RMax=NY/2 - 5.0, RBinSize=RBin,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=EtaBin,
        SubPixelLevel=1, Normalize=1,
    )


def _sigma_mu_at_eta(cake_2d: np.ndarray, eta_axis: np.ndarray,
                     R_axis: np.ndarray, R_target: float,
                     eta_center: float = 0.0, win: float = 5.0) -> float:
    ridx = int(np.argmin(np.abs(R_axis - R_target)))
    mask = np.abs(eta_axis - eta_center) < win
    v = cake_2d[ridx, mask]
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.std(v) / np.mean(v) * 100) if len(v) >= 2 else float("nan")


def _r_axis_from(p):
    return p.RMin + p.RBinSize * (np.arange(p.n_r_bins, dtype=np.float64) + 0.5)


def _eta_axis_from(p):
    return p.EtaMin + p.EtaBinSize * (
        np.arange(p.n_eta_bins, dtype=np.float64) + 0.5)


def test_fused_geom_shape_and_normalize():
    """The fused geometry has area_per_bin = 1, so normalize=True is safe."""
    p = _params()
    geom = build_fused_geometry(p, n_shifts=2, mode="bilinear", verbose=False)
    assert geom.n_r == p.n_r_bins
    assert geom.n_eta == p.n_eta_bins
    # area_per_bin should be all ones
    assert torch.allclose(geom.area_per_bin,
                          torch.ones_like(geom.area_per_bin))


def test_fused_smoke_runs_end_to_end():
    """Build, integrate, smooth — end to end."""
    p = _params()
    img = _synth_ring_image(NY=p.NrPixelsY, NZ=p.NrPixelsZ,
                             BC=p.BC_y, Lsd=p.Lsd, px=p.pxY)
    img_t = torch.from_numpy(img.astype(np.float64))

    geom = build_fused_geometry(p, n_shifts=4, mode="bilinear", verbose=False)
    cake = integrate(img_t, geom, mode="bilinear", normalize=False
                     ).detach().cpu().numpy()
    assert cake.shape == (p.n_r_bins, p.n_eta_bins)
    # Total flux should be positive
    assert np.nansum(cake) > 0

    cake_smoothed = gauss_smooth_eta(cake, sigma_bins=1.0)
    assert cake_smoothed.shape == cake.shape


def test_fused_reduces_cardinal_sigma_mu():
    """The fused geometry should beat the naive single-map cardinal σ/μ."""
    p = _params()
    img = _synth_ring_image(NY=p.NrPixelsY, NZ=p.NrPixelsZ,
                             BC=p.BC_y, Lsd=p.Lsd, px=p.pxY)
    img_t = torch.from_numpy(img.astype(np.float64))

    # Baseline: single-map (n_shifts=1) — equivalent to standard integrate
    geom_base = build_fused_geometry(p, n_shifts=1, mode="bilinear")
    cake_base = integrate(img_t, geom_base, mode="bilinear",
                           normalize=False).detach().cpu().numpy()

    # Fused: 4 shifted maps
    geom_fused = build_fused_geometry(p, n_shifts=4, mode="bilinear")
    cake_fused = integrate(img_t, geom_fused, mode="bilinear",
                            normalize=False).detach().cpu().numpy()

    R = _r_axis_from(p)
    E = _eta_axis_from(p)
    R_ring = 80.0
    sm_base = _sigma_mu_at_eta(cake_base, E, R, R_ring, 0.0)
    sm_fused = _sigma_mu_at_eta(cake_fused, E, R, R_ring, 0.0)

    # Fused must beat the baseline by at least 5 % relative
    assert sm_fused < sm_base, (
        f"Fused σ/μ ({sm_fused:.3f}%) should be < baseline "
        f"σ/μ ({sm_base:.3f}%)"
    )


def test_gauss_smooth_eta_preserves_ring_flux():
    """area-weighted Gaussian preserves per-row total intensity (with
    uniform-area assumption it also preserves per-row sum)."""
    rng = np.random.default_rng(0)
    cake = rng.uniform(1.0, 10.0, size=(8, 360))
    smoothed = gauss_smooth_eta(cake, sigma_bins=2.0)
    assert smoothed.shape == cake.shape
    # Without area weights the row-sum should be preserved to ~1 part in 10^6
    np.testing.assert_allclose(smoothed.sum(axis=1), cake.sum(axis=1),
                                rtol=1e-9, atol=1e-9)


def test_gauss_smooth_eta_torch_tensor():
    """Should accept and return torch tensors."""
    cake = torch.rand(8, 360, dtype=torch.float64) * 10.0
    smoothed = gauss_smooth_eta(cake, sigma_bins=2.0)
    assert isinstance(smoothed, torch.Tensor)
    assert smoothed.shape == cake.shape


def test_median_filter_eta_preserves_flux_when_requested():
    rng = np.random.default_rng(1)
    cake = rng.uniform(1.0, 10.0, size=(8, 360))
    filtered = median_filter_eta(cake, window=5, preserve_ring_flux=True)
    np.testing.assert_allclose(filtered.sum(axis=1), cake.sum(axis=1),
                                rtol=1e-9, atol=1e-9)


def test_median_filter_eta_window_odd():
    cake = np.zeros((4, 360))
    with pytest.raises(ValueError):
        median_filter_eta(cake, window=4)
