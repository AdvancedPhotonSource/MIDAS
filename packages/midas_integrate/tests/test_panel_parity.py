"""Per-panel correction parity tests.

Mirror of FF_HEDM/src/Panel.h + the panel block in MapperCore.c (lines
94-105). These tests exercise the numba scalar helpers, the pure-Python
``apply_panel_correction``, and the end-to-end ``build_map`` integration.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_integrate._mapper_numba import (
    _apply_panel_correction_njit,
    _get_panel_index_njit,
)
from midas_integrate.detector_mapper import build_map
from midas_integrate.panel import (
    Panel,
    apply_panel_correction,
    generate_panels,
    get_panel_index,
    load_panel_shifts,
    panels_to_array,
    save_panel_shifts,
)
from midas_integrate.params import IntegrationParams


def _params(NY=64, NZ=64):
    return IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=2.0, RMax=NY / 2.0 - 2.0, RBinSize=2.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
    )


def test_generate_panels_layout():
    panels = generate_panels(2, 2, 32, 32, gaps_y=[4], gaps_z=[2])
    assert len(panels) == 4
    assert panels[0].yMin == 0 and panels[0].yMax == 31
    assert panels[0].zMin == 0 and panels[0].zMax == 31
    assert panels[1].yMin == 0 and panels[1].zMin == 34   # zStart += 32+2
    assert panels[2].yMin == 36 and panels[2].zMin == 0   # yStart += 32+4
    assert panels[3].yMin == 36 and panels[3].zMin == 34
    # Centers should be midpoints
    assert panels[0].centerY == 15.5
    assert panels[0].centerZ == 15.5


def test_apply_panel_correction_matches_numba():
    """Scalar Python and numba implementations must produce identical output."""
    rng = np.random.default_rng(42)
    panel = Panel(
        id=0, yMin=0, yMax=99, zMin=0, zMax=99,
        dY=0.7, dZ=-0.4, dTheta=0.5, dLsd=120.0, dP2=0.001,
    )
    panels_arr = panels_to_array([panel])

    for _ in range(50):
        y = rng.uniform(0, 99)
        z = rng.uniform(0, 99)
        py, pz = apply_panel_correction(y, z, panel)
        ny, nz = _apply_panel_correction_njit(y, z, panels_arr, 0)
        assert abs(py - ny) < 1e-12
        assert abs(pz - nz) < 1e-12


def test_get_panel_index_matches_python():
    panels = generate_panels(2, 2, 32, 32, gaps_y=[4], gaps_z=[2])
    panels_arr = panels_to_array(panels)
    rng = np.random.default_rng(0)
    for _ in range(100):
        y = rng.uniform(0, 70)
        z = rng.uniform(0, 70)
        py = get_panel_index(y, z, panels)
        ny = _get_panel_index_njit(y, z, panels_arr, len(panels))
        assert py == ny


def test_zero_shift_panels_match_no_panels():
    """Panels with every shift = 0 must produce identical output to no panels.

    Note: with panels present we route through the subpixel kernel (which
    rebuilds corners scalar-by-scalar) instead of the fast precomputed-
    corners kernel; the 1e-13 differences here are pure float-op-ordering
    noise.
    """
    p = _params()
    panels = generate_panels(2, 2, 32, 32)   # all shifts 0
    res_panels = build_map(p, panels=panels, auto_load=False, verbose=False)
    res_none = build_map(p, panels=None, auto_load=False, verbose=False)

    np.testing.assert_array_equal(res_none.counts, res_panels.counts)
    np.testing.assert_array_equal(res_none.pxList["y"], res_panels.pxList["y"])
    np.testing.assert_array_equal(res_none.pxList["z"], res_panels.pxList["z"])
    np.testing.assert_allclose(res_none.pxList["frac"],
                               res_panels.pxList["frac"],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res_none.pxList["areaWeight"],
                               res_panels.pxList["areaWeight"],
                               rtol=1e-9, atol=1e-12)


def test_uniform_panel_dy_equivalent_to_bc_shift():
    """One full-detector panel with dY = k must equal shifting BC_y by -k.

    Per Panel.h: the corrected pixel coordinate is ``y + dY``. Per
    geometry.pixel_to_REta: ``Yc = (-Y + Ycen) * px`` — so increasing Y
    by k is equivalent to decreasing Ycen by k.
    """
    p = _params()
    p.SubPixelLevel = 1                         # no cardinal splitting

    NY = p.NrPixelsY; NZ = p.NrPixelsZ
    k = 0.3
    panel = Panel(id=0, yMin=0, yMax=NY - 1, zMin=0, zMax=NZ - 1, dY=k)
    res_panel = build_map(p, panels=[panel], auto_load=False, verbose=False)

    p_shifted = _params()
    p_shifted.SubPixelLevel = 1
    p_shifted.BC_y = p.BC_y - k
    res_shift = build_map(p_shifted, auto_load=False, verbose=False)

    # Both runs must populate the same set of bins with matching frac values.
    np.testing.assert_array_equal(res_shift.counts, res_panel.counts)
    # Per-entry values must agree to high precision.
    sort_panel = np.lexsort((res_panel.pxList["y"], res_panel.pxList["z"]))
    sort_shift = np.lexsort((res_shift.pxList["y"], res_shift.pxList["z"]))
    np.testing.assert_allclose(
        res_panel.pxList["frac"][sort_panel],
        res_shift.pxList["frac"][sort_shift],
        rtol=2e-7, atol=0,
    )


def test_save_load_panel_shifts_roundtrip(tmp_path):
    panels = generate_panels(1, 2, 64, 32)
    panels[0].dY = 0.31; panels[0].dZ = -0.12; panels[0].dTheta = 0.05
    panels[0].dLsd = 250.0; panels[0].dP2 = 0.0017
    panels[1].dY = -0.42; panels[1].dZ = 0.27;  panels[1].dTheta = -0.03

    fp = tmp_path / "panels.txt"
    save_panel_shifts(fp, panels)

    panels2 = generate_panels(1, 2, 64, 32)   # fresh, all zero
    load_panel_shifts(fp, panels2)
    for a, b in zip(panels, panels2):
        assert a.dY == pytest.approx(b.dY, abs=1e-9)
        assert a.dZ == pytest.approx(b.dZ, abs=1e-9)
        assert a.dTheta == pytest.approx(b.dTheta, abs=1e-9)
        assert a.dLsd == pytest.approx(b.dLsd, abs=1e-9)
        assert a.dP2 == pytest.approx(b.dP2, abs=1e-9)
