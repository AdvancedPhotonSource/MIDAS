"""Residual radial-correction parity tests.

Mirror of FF_HEDM/src/DetectorGeometry.h ``dg_residual_corr_lookup`` and
its application in ``dg_pixel_to_REta_corr``.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_integrate._mapper_numba import _residual_corr_lookup_njit
from midas_integrate.detector_mapper import build_map
from midas_integrate.params import IntegrationParams
from midas_integrate.residual_corr import (
    ResidualCorrection,
    empty_residual_corr_array,
    load_residual_correction_map,
    lookup_python,
)


def _params(NY=64, NZ=64):
    return IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=2.0, RMax=NY / 2.0 - 2.0, RBinSize=2.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
        SubPixelLevel=1,
    )


def test_lookup_at_pixel_centers():
    """Bilinear sample at integer (Y, Z) returns map[Z, Y] exactly."""
    NY, NZ = 16, 12
    rng = np.random.default_rng(0)
    arr = rng.uniform(-1, 1, size=(NZ, NY)).astype(np.float64)
    corr = ResidualCorrection(map=arr, NrPixelsY=NY, NrPixelsZ=NZ)
    for y in range(NY - 1):
        for z in range(NZ - 1):
            v_py = lookup_python(corr, float(y), float(z))
            v_nj = _residual_corr_lookup_njit(arr, NY, NZ, float(y), float(z))
            assert v_py == pytest.approx(arr[z, y], abs=1e-12)
            assert v_nj == pytest.approx(arr[z, y], abs=1e-12)


def test_lookup_clamps_out_of_bounds():
    """Negative coords clamp to 0; coords beyond N-1 clamp to N-1.001."""
    NY, NZ = 8, 8
    arr = np.zeros((NZ, NY), dtype=np.float64)
    arr[0, 0] = 7.0; arr[NZ - 1, NY - 1] = -3.0
    corr = ResidualCorrection(map=arr, NrPixelsY=NY, NrPixelsZ=NZ)

    # Negative input clamps to (0, 0)
    assert lookup_python(corr, -5.0, -5.0) == pytest.approx(7.0, abs=1e-12)
    # Above-range clamps to (N-1.001, N-1.001); bilinear interp at fy=fz=0.999
    # gives ~0.998 × map[N-1, N-1]. We just want to verify the corner value
    # dominates (rather than e.g. zero from extrapolation).
    near = lookup_python(corr, NY + 5, NZ + 5)
    assert near == pytest.approx(-3.0, abs=1e-2)


def test_python_and_numba_lookup_agree_on_random_grid():
    """Independent agreement on random non-integer queries."""
    NY, NZ = 24, 17
    rng = np.random.default_rng(99)
    arr = rng.uniform(-1, 1, size=(NZ, NY)).astype(np.float64)
    corr = ResidualCorrection(map=arr, NrPixelsY=NY, NrPixelsZ=NZ)
    qy = rng.uniform(-2, NY + 2, 200)
    qz = rng.uniform(-2, NZ + 2, 200)
    for y, z in zip(qy, qz):
        v_py = lookup_python(corr, y, z)
        v_nj = _residual_corr_lookup_njit(arr, NY, NZ, y, z)
        assert v_py == pytest.approx(v_nj, abs=1e-12)


def test_zero_residual_matches_no_residual():
    """All-zero residual map must produce identical mapping to no residual."""
    p = _params()
    NY, NZ = p.NrPixelsY, p.NrPixelsZ
    corr = ResidualCorrection(map=np.zeros((NZ, NY)),
                              NrPixelsY=NY, NrPixelsZ=NZ)
    res_corr = build_map(p, residual_corr=corr,
                         auto_load=False, verbose=False)
    res_none = build_map(p, residual_corr=None,
                         auto_load=False, verbose=False)
    np.testing.assert_array_equal(res_none.counts, res_corr.counts)
    # Subpixel kernel re-derives corners scalar-by-scalar; minor 1e-13
    # rounding-order differences vs the precomputed-corners path are
    # expected.
    np.testing.assert_allclose(res_none.pxList["frac"],
                               res_corr.pxList["frac"],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res_none.pxList["areaWeight"],
                               res_corr.pxList["areaWeight"],
                               rtol=1e-9, atol=1e-12)


def test_constant_residual_shifts_deltaR():
    """Adding a constant ΔR to every pixel shifts pxList["deltaR"] by that
    amount (all other quantities — counts, frac, area — unchanged because
    the bin selection only depends on Rt's bin-bucket, not its sub-pixel
    offset within the bucket, *as long as* the shift is small enough not
    to push any pixel across a bin boundary)."""
    p = _params()
    NY, NZ = p.NrPixelsY, p.NrPixelsZ
    delta = 1e-3   # small enough to keep every entry in its original bin
    corr = ResidualCorrection(map=np.full((NZ, NY), delta),
                              NrPixelsY=NY, NrPixelsZ=NZ)

    res_corr = build_map(p, residual_corr=corr,
                         auto_load=False, verbose=False)
    res_none = build_map(p, residual_corr=None,
                         auto_load=False, verbose=False)
    np.testing.assert_array_equal(res_none.counts, res_corr.counts)
    # deltaR shifted by delta; allow small float tolerance (deltaR is float32)
    np.testing.assert_allclose(
        res_corr.pxList["deltaR"] - res_none.pxList["deltaR"], delta,
        atol=2e-4,
    )


def test_residual_file_roundtrip(tmp_path):
    """File = NY*NZ float64 row-major doubles; round-trip via
    ``load_residual_correction_map``."""
    NY, NZ = 16, 12
    rng = np.random.default_rng(1)
    arr = rng.uniform(-1, 1, size=(NZ, NY)).astype(np.float64)
    fp = tmp_path / "residual.bin"
    fp.write_bytes(arr.tobytes())
    loaded = load_residual_correction_map(fp, NrPixelsY=NY, NrPixelsZ=NZ)
    np.testing.assert_array_equal(loaded.map, arr)
    assert loaded.NrPixelsY == NY and loaded.NrPixelsZ == NZ


def test_residual_lookup_returns_zero_when_empty():
    arr = empty_residual_corr_array()
    assert _residual_corr_lookup_njit(arr, 0, 0, 5.0, 5.0) == 0.0
