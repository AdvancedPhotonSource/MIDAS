"""End-to-end DetectorMapper smoke + correctness test on a tiny detector."""
from __future__ import annotations

import numpy as np
import pytest

from midas_integrate.bin_io import load_map
from midas_integrate.detector_mapper import build_and_write_map, build_map
from midas_integrate.kernels import build_csr, integrate, profile_1d
from midas_integrate.params import IntegrationParams
import torch


def _tiny_params(NY=32, NZ=32, RBin=2.0, EtaBin=30.0):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=100_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0,
        RhoD=NY,
        RMin=0.0, RMax=NY / 2.0, RBinSize=RBin,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=EtaBin,
    )
    return p


def test_build_map_no_distortion(tmp_path):
    p = _tiny_params(NY=32, NZ=32, RBin=2.0, EtaBin=30.0)
    res = build_map(p, n_jobs=1, verbose=False)
    # All pixels should be assigned to *some* bin (no holes for a flat detector
    # centered at beam-center, no tilt).
    total_area = float(res.pxList["areaWeight"].sum())
    # Total area must roughly equal total pixel area in pixel units = NY * NZ
    # but with some loss at the very boundary R = RMax. Allow 20% tolerance.
    expected = p.NrPixelsY * p.NrPixelsZ
    assert total_area > 0.5 * expected
    # All entries are within detector bounds
    assert res.pxList["y"].min() >= 0
    assert res.pxList["y"].max() < p.NrPixelsY
    assert res.pxList["z"].min() >= 0
    assert res.pxList["z"].max() < p.NrPixelsZ


def test_build_and_write_round_trips(tmp_path):
    p = _tiny_params(NY=24, NZ=24, RBin=2.0, EtaBin=60.0)
    map_path, nmap_path = build_and_write_map(
        p, output_dir=tmp_path, n_jobs=1, verbose=False,
    )
    pm = load_map(map_path, nmap_path)
    assert pm.n_bins == p.n_bins
    assert pm.map_header is not None
    assert pm.map_header.version == 3


def test_flat_field_correction_recovers_uniform_profile(tmp_path):
    """A constant raw image divided by a flat-field map should integrate to
    that constant divided by the flat — which we verify by feeding back the
    pre-multiplied image."""
    p = _tiny_params(NY=24, NZ=24, RBin=2.0, EtaBin=60.0)

    # Synthetic flat with mild gradient (0.5..1.5)
    flat = np.linspace(0.5, 1.5, p.NrPixelsY * p.NrPixelsZ,
                       dtype=np.float64).reshape(p.NrPixelsZ, p.NrPixelsY)

    # Build TWO maps: one without flat (baseline), one with flat folded in.
    res_baseline = build_map(p, n_jobs=1, verbose=False)
    res_flat     = build_map(p, n_jobs=1, verbose=False, flat=flat)

    # Per-entry (y, z, area) must match exactly; only frac changes by 1/flat.
    np.testing.assert_array_equal(res_baseline.pxList["y"],     res_flat.pxList["y"])
    np.testing.assert_array_equal(res_baseline.pxList["z"],     res_flat.pxList["z"])
    np.testing.assert_array_equal(res_baseline.pxList["areaWeight"],
                                   res_flat.pxList["areaWeight"])

    iy = res_baseline.pxList["y"].astype(np.int64)
    iz = res_baseline.pxList["z"].astype(np.int64)
    expected_frac = res_baseline.pxList["frac"] / flat[iz, iy]
    np.testing.assert_allclose(res_flat.pxList["frac"], expected_frac,
                               rtol=1e-12, atol=1e-15)


def test_integrate_with_built_map_constant_image_gives_constant_profile(tmp_path):
    """If the input image is constant, the area-weighted 1D profile must
    equal that constant in every R bin that has any pixels."""
    p = _tiny_params(NY=24, NZ=24, RBin=2.0, EtaBin=60.0)
    build_and_write_map(p, output_dir=tmp_path, n_jobs=1, verbose=False)
    pm = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    geom = build_csr(pm, n_r=p.n_r_bins, n_eta=p.n_eta_bins,
                     n_pixels_y=p.NrPixelsY, n_pixels_z=p.NrPixelsZ,
                     dtype=torch.float64,
                     bc_y=p.BC_y, bc_z=p.BC_z)
    img = torch.full((p.NrPixelsZ, p.NrPixelsY), 7.5, dtype=torch.float64)
    int2d = integrate(img, geom, mode="floor", normalize=True)
    prof = profile_1d(int2d, geom, mode="area_weighted").numpy()
    # Where there are pixels, profile should equal 7.5 (the constant)
    nonzero = np.where(prof > 0.0)[0]
    assert nonzero.size > 0
    np.testing.assert_array_almost_equal(prof[nonzero], 7.5, decimal=6)
