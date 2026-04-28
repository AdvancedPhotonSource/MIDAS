"""Distortion-map parity tests.

Mirror of FF_HEDM/src/DetectorMapper.c lines 659-676 (load distortion file
and apply TransOpt) and MapperCore.c lines 104-105 (apply Δy/Δz to pixel
coordinates).
"""
from __future__ import annotations

import numpy as np

from midas_integrate.detector_mapper import (
    _apply_trans_opt_forward,
    build_map,
    load_distortion_maps,
)
from midas_integrate.params import IntegrationParams


def _params(NY=64, NZ=64, **overrides):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=2.0, RMax=NY / 2.0 - 2.0, RBinSize=2.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
        SubPixelLevel=1,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def test_zero_distortion_matches_no_distortion():
    p = _params()
    NY, NZ = p.NrPixelsY, p.NrPixelsZ
    dy = np.zeros((NZ, NY), dtype=np.float64)
    dz = np.zeros((NZ, NY), dtype=np.float64)

    res_zero = build_map(p, distortion_y=dy, distortion_z=dz,
                         auto_load=False, verbose=False)
    res_none = build_map(p, auto_load=False, verbose=False)

    np.testing.assert_array_equal(res_none.counts, res_zero.counts)
    # Subpixel vs fast kernel disagree only at IEEE rounding noise (~1e-13)
    # because they evaluate corners in different orders.
    np.testing.assert_allclose(res_none.pxList["frac"], res_zero.pxList["frac"],
                                rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res_none.pxList["areaWeight"],
                                res_zero.pxList["areaWeight"],
                                rtol=1e-9, atol=1e-12)


def test_constant_distortion_equivalent_to_bc_shift():
    """Δy = k everywhere ⇔ shifting BC_y by -k (cf. test_panel_dy_eq_bc)."""
    p = _params()
    NY, NZ = p.NrPixelsY, p.NrPixelsZ
    k = 0.25
    dy = np.full((NZ, NY), k, dtype=np.float64)
    dz = np.zeros((NZ, NY), dtype=np.float64)
    res_dist = build_map(p, distortion_y=dy, distortion_z=dz,
                         auto_load=False, verbose=False)

    p_shift = _params(BC_y=p.BC_y - k)
    res_shift = build_map(p_shift, auto_load=False, verbose=False)

    np.testing.assert_array_equal(res_dist.counts, res_shift.counts)
    sort_d = np.lexsort((res_dist.pxList["y"], res_dist.pxList["z"]))
    sort_s = np.lexsort((res_shift.pxList["y"], res_shift.pxList["z"]))
    np.testing.assert_allclose(res_dist.pxList["frac"][sort_d],
                               res_shift.pxList["frac"][sort_s],
                               rtol=2e-7, atol=0)


def test_distortion_file_roundtrip(tmp_path):
    """File format = two raw float64 (NY*NZ) arrays — readable via
    ``load_distortion_maps`` (which also applies TransOpt)."""
    p = _params(NY=32, NZ=32)
    NY, NZ = p.NrPixelsY, p.NrPixelsZ
    rng = np.random.default_rng(7)
    dy = rng.uniform(-0.5, 0.5, size=(NZ, NY)).astype(np.float64)
    dz = rng.uniform(-0.5, 0.5, size=(NZ, NY)).astype(np.float64)

    fp = tmp_path / "distortion.bin"
    with open(fp, "wb") as f:
        f.write(dy.tobytes())
        f.write(dz.tobytes())

    p.DistortionFile = str(fp)
    dy_load, dz_load = load_distortion_maps(fp, p)
    np.testing.assert_array_equal(dy_load, dy)
    np.testing.assert_array_equal(dz_load, dz)


def test_trans_opt_forward_flip_lr():
    """TransOpt=[1] flips Y; loaded map must match np.fliplr(arr)."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = _apply_trans_opt_forward(arr, [1], NrPixelsY=4, NrPixelsZ=3)
    np.testing.assert_array_equal(out, arr[:, ::-1])


def test_trans_opt_forward_flip_ud():
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = _apply_trans_opt_forward(arr, [2], NrPixelsY=4, NrPixelsZ=3)
    np.testing.assert_array_equal(out, arr[::-1, :])


def test_trans_opt_forward_transpose_requires_square():
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    try:
        _apply_trans_opt_forward(arr, [3], NrPixelsY=4, NrPixelsZ=3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-square transpose")


def test_distortion_shape_validation():
    """Wrong-shape arrays must raise."""
    p = _params(NY=64, NZ=64)
    bad = np.zeros((10, 10), dtype=np.float64)
    try:
        build_map(p, distortion_y=bad, distortion_z=bad,
                  auto_load=False, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad distortion shape")
