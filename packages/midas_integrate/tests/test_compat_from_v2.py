"""Tests for the calibrate_v2 → integrate compatibility path.

Two regression guards:

1. **forward-model parity** — integrate's numpy ``pixel_to_REta`` and
   calibrate_v2's torch ``pixel_to_REta`` MUST stay bit-identical at
   fp64 precision.  They live in separate packages so a sign fix in
   one won't propagate; this test catches the drift immediately.

2. **adapter round-trip** — running calibrate_v2 on a synthetic
   geometry, exporting to v1 paramstest, and re-importing via
   ``compat.from_v2.params_from_v2_unpacked`` must produce an
   ``IntegrationParams`` whose ``pixel_to_REta`` agrees with v2's
   evaluated at the same params.

Skipped if ``midas_calibrate_v2`` is not importable (so this test
file won't break CI in repos that only depend on midas-integrate).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
v2 = pytest.importorskip("midas_calibrate_v2")

from midas_calibrate_v2.forward.geometry import pixel_to_REta as v2_pixel_to_REta
from midas_calibrate_v2.forward.distortion import build_p_coeffs

from midas_integrate.geometry import (
    pixel_to_REta as integrate_pixel_to_REta,
    build_tilt_matrix,
)
from midas_integrate.params import IntegrationParams
from midas_integrate.compat.from_v2 import params_from_v2_unpacked


def _evaluate_v2(p_dict: dict, Y: np.ndarray, Z: np.ndarray, rho_d: float):
    p_coeffs = build_p_coeffs(p_dict, dtype=torch.float64)
    out = v2_pixel_to_REta(
        torch.tensor(Y, dtype=torch.float64),
        torch.tensor(Z, dtype=torch.float64),
        Lsd=p_dict["Lsd"], BC_y=p_dict["BC_y"], BC_z=p_dict["BC_z"],
        tx=p_dict["tx"], ty=p_dict["ty"], tz=p_dict["tz"],
        p_coeffs=p_coeffs, parallax=p_dict["Parallax"],
        pxY=p_dict["pxY"], pxZ=p_dict["pxZ"],
        rho_d=torch.tensor(rho_d, dtype=torch.float64),
    )
    return out.R_px.cpu().numpy(), out.eta_deg.cpu().numpy()


def _evaluate_integrate(ip: IntegrationParams, Y: np.ndarray, Z: np.ndarray):
    TRs = build_tilt_matrix(ip.tx, ip.ty, ip.tz)
    R, Eta = integrate_pixel_to_REta(
        Y, Z,
        Ycen=ip.BC_y, Zcen=ip.BC_z, TRs=TRs,
        Lsd=ip.Lsd, RhoD=ip.RhoD, px=ip.pxY, parallax=ip.Parallax,
        p0=ip.p0, p1=ip.p1, p2=ip.p2, p3=ip.p3, p4=ip.p4,
        p5=ip.p5, p6=ip.p6, p7=ip.p7, p8=ip.p8, p9=ip.p9,
        p10=ip.p10, p11=ip.p11, p12=ip.p12, p13=ip.p13, p14=ip.p14,
    )
    return R, Eta


def _make_v2_dict(*, Lsd=895_900.0, BC_y=1447.0, BC_z=1469.0,
                   ty=-0.31, tz=0.39, parallax=0.0, **distortion):
    """Build a v2-named param dict with optional distortion overrides."""
    base = dict(
        Lsd=torch.tensor(Lsd, dtype=torch.float64),
        BC_y=torch.tensor(BC_y, dtype=torch.float64),
        BC_z=torch.tensor(BC_z, dtype=torch.float64),
        tx=torch.tensor(0.0, dtype=torch.float64),
        ty=torch.tensor(ty, dtype=torch.float64),
        tz=torch.tensor(tz, dtype=torch.float64),
        pxY=torch.tensor(150.0, dtype=torch.float64),
        pxZ=torch.tensor(150.0, dtype=torch.float64),
        Parallax=torch.tensor(parallax, dtype=torch.float64),
        # Default distortion: all zero, then override.
        iso_R2=torch.tensor(0.0, dtype=torch.float64),
        iso_R4=torch.tensor(0.0, dtype=torch.float64),
        iso_R6=torch.tensor(0.0, dtype=torch.float64),
        a1=torch.tensor(0.0, dtype=torch.float64),
        phi1=torch.tensor(0.0, dtype=torch.float64),
        a2=torch.tensor(0.0, dtype=torch.float64),
        phi2=torch.tensor(0.0, dtype=torch.float64),
        a3=torch.tensor(0.0, dtype=torch.float64),
        phi3=torch.tensor(0.0, dtype=torch.float64),
        a4=torch.tensor(0.0, dtype=torch.float64),
        phi4=torch.tensor(0.0, dtype=torch.float64),
        a5=torch.tensor(0.0, dtype=torch.float64),
        phi5=torch.tensor(0.0, dtype=torch.float64),
        a6=torch.tensor(0.0, dtype=torch.float64),
        phi6=torch.tensor(0.0, dtype=torch.float64),
    )
    for k, v in distortion.items():
        base[k] = torch.tensor(v, dtype=torch.float64)
    return base


def _grid(n: int = 8):
    ys = np.linspace(50, 2830, n)
    zs = np.linspace(50, 2830, n)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    return Y.flatten(), Z.flatten()


@pytest.mark.parametrize("scenario", [
    "no_distortion",
    "isotropic_only",
    "all_15_active",
    "with_parallax",
])
def test_forward_model_parity_calibrate_v2_vs_integrate(scenario):
    """integrate.geometry.pixel_to_REta must agree bit-identically with
    midas_calibrate_v2.forward.geometry.pixel_to_REta on every param
    configuration we care about."""
    if scenario == "no_distortion":
        v2_dict = _make_v2_dict()
    elif scenario == "isotropic_only":
        v2_dict = _make_v2_dict(iso_R2=3.4e-4, iso_R4=1.2e-3, iso_R6=-9.1e-4)
    elif scenario == "all_15_active":
        v2_dict = _make_v2_dict(
            iso_R2=3.4e-4, iso_R4=1.2e-3, iso_R6=-9.1e-4,
            a1=4.3e-4, phi1=110.1,
            a2=3.4e-4, phi2=-4.3,
            a3=-1.3e-4, phi3=127.6,
            a4=1.7e-4, phi4=-7.3,
            a5=-9.6e-5, phi5=277.1,
            a6=4.4e-6, phi6=-62.8,
        )
    elif scenario == "with_parallax":
        v2_dict = _make_v2_dict(parallax=12.5)

    Y, Z = _grid(7)
    rho_d = 309_094.286

    R_v2, Eta_v2 = _evaluate_v2(v2_dict, Y, Z, rho_d)
    template = IntegrationParams(
        NrPixelsY=2880, NrPixelsZ=2880,
        pxY=150.0, pxZ=150.0,
        RhoD=rho_d,
    )
    ip = params_from_v2_unpacked(v2_dict, template=template,
                                  warn_on_dropped=False)
    R_int, Eta_int = _evaluate_integrate(ip, Y, Z)

    np.testing.assert_allclose(R_int, R_v2, rtol=0, atol=1e-9,
                                err_msg=f"R drift in scenario {scenario}")
    np.testing.assert_allclose(Eta_int, Eta_v2, rtol=0, atol=1e-12,
                                err_msg=f"Eta drift in scenario {scenario}")


def test_from_v2_adapter_warns_on_delta_r_k():
    """δr_k cannot be carried into integrate's radial map; the adapter
    must emit a UserWarning pointing at the JSON sidecar."""
    import warnings
    v2_dict = _make_v2_dict()
    v2_dict["delta_r_k"] = torch.zeros(15, dtype=torch.float64)

    template = IntegrationParams(
        NrPixelsY=2880, NrPixelsZ=2880, pxY=150.0, pxZ=150.0, RhoD=309094.0,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        params_from_v2_unpacked(v2_dict, template=template,
                                 warn_on_dropped=True)
        msgs = [str(rec.message) for rec in w]
        assert any("delta_r_k" in m for m in msgs), \
            f"expected delta_r_k drop warning, got: {msgs}"
        assert any("write_per_ring_offsets_json" in m for m in msgs), \
            "warning should point at the JSON sidecar function"


def test_from_v2_adapter_warns_on_panel_blocks():
    """Per-panel blocks need the panel-shifts file path; adapter warns."""
    import warnings
    v2_dict = _make_v2_dict()
    v2_dict["panel_delta_yz"] = torch.zeros(48, 2, dtype=torch.float64)
    v2_dict["panel_delta_theta"] = torch.zeros(48, dtype=torch.float64)

    template = IntegrationParams(
        NrPixelsY=2880, NrPixelsZ=2880, pxY=150.0, pxZ=150.0, RhoD=309094.0,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        params_from_v2_unpacked(v2_dict, template=template,
                                 warn_on_dropped=True)
        msgs = [str(rec.message) for rec in w]
        assert any("panel_delta_yz" in m for m in msgs)


def test_from_v2_distortion_remap():
    """Verify v2 → v1 distortion-name remap puts each amplitude/phase in
    the slot the integrate forward model expects.  This is the
    single-most-likely source of silent integration error after a v2
    calibration if the remap drifts."""
    v2_dict = _make_v2_dict(
        iso_R2=1.001, iso_R4=2.002, iso_R6=3.003,
        a1=4.004, phi1=44.0,
        a2=5.005, phi2=55.0,
        a3=6.006, phi3=66.0,
        a4=7.007, phi4=77.0,
        a5=8.008, phi5=88.0,
        a6=9.009, phi6=99.0,
    )
    template = IntegrationParams(
        NrPixelsY=2880, NrPixelsZ=2880, pxY=150.0, pxZ=150.0, RhoD=309094.0,
    )
    ip = params_from_v2_unpacked(v2_dict, template=template,
                                  warn_on_dropped=False)

    # The canonical v2 → v1 mapping (kept in sync with
    # midas_calibrate_v2.compat.to_v1._V2_TO_V1_DISTORTION):
    #   iso_R2→p2, iso_R4→p5, iso_R6→p4,
    #   a1→p7,  phi1→p8,
    #   a2→p0,  phi2→p6,
    #   a3→p9,  phi3→p10,
    #   a4→p1,  phi4→p3,
    #   a5→p11, phi5→p12,
    #   a6→p13, phi6→p14
    assert ip.p2  == pytest.approx(1.001)
    assert ip.p5  == pytest.approx(2.002)
    assert ip.p4  == pytest.approx(3.003)
    assert ip.p7  == pytest.approx(4.004)
    assert ip.p8  == pytest.approx(44.0)
    assert ip.p0  == pytest.approx(5.005)
    assert ip.p6  == pytest.approx(55.0)
    assert ip.p9  == pytest.approx(6.006)
    assert ip.p10 == pytest.approx(66.0)
    assert ip.p1  == pytest.approx(7.007)
    assert ip.p3  == pytest.approx(77.0)
    assert ip.p11 == pytest.approx(8.008)
    assert ip.p12 == pytest.approx(88.0)
    assert ip.p13 == pytest.approx(9.009)
    assert ip.p14 == pytest.approx(99.0)


def test_file_path_matches_in_memory_adapter(tmp_path):
    """The two routes from a v2 unpacked dict to an IntegrationParams
    must agree on every geometry / distortion field:

      Route A (in-memory):
          unpacked → params_from_v2_unpacked(template) → IntegrationParams

      Route B (file-based, the path tools downstream of integrate use):
          unpacked → write_v1_paramstest → parse_params → IntegrationParams

    If they ever diverge, either the v2-side write_v1_paramstest dropped
    a field or the integrate-side parse_params lost a key. Both bugs are
    silent (no exception, just wrong R/Eta), so they need a regression
    pin.
    """
    from midas_integrate.params import parse_params
    from midas_calibrate_v2.compat.to_v1 import write_v1_paramstest
    from midas_calibrate.params import CalibrationParams as V1Params

    v2_dict = _make_v2_dict(
        Lsd=895_900.0, BC_y=1447.0, BC_z=1469.0,
        ty=-0.31, tz=0.39, parallax=12.5,
        iso_R2=3.4e-4, iso_R4=1.2e-3, iso_R6=-9.1e-4,
        a1=4.3e-4, phi1=110.1,
        a2=3.4e-4, phi2=-4.3,
        a3=-1.3e-4, phi3=127.6,
        a4=1.7e-4, phi4=-7.3,
        a5=-9.6e-5, phi5=277.1,
        a6=4.4e-6, phi6=-62.8,
    )

    integ_template = IntegrationParams(
        NrPixelsY=2880, NrPixelsZ=2880,
        pxY=150.0, pxZ=150.0,
        RhoD=309_094.286,
    )
    ip_mem = params_from_v2_unpacked(v2_dict, template=integ_template,
                                      warn_on_dropped=False)

    v1_template = V1Params(
        NrPixelsY=2880, NrPixelsZ=2880,
        pxY=150.0, pxZ=150.0,
        RhoD=309_094.286,
        Wavelength=0.172979,
    )
    paramstest_path = tmp_path / "v2_export.txt"
    write_v1_paramstest(v2_dict, v1_template, paramstest_path)
    ip_file = parse_params(paramstest_path)

    # Geometry + tilts + parallax + 15 distortions must all match.
    for k in ("Lsd", "BC_y", "BC_z", "tx", "ty", "tz", "Parallax",
              "pxY", "pxZ", "RhoD", "NrPixelsY", "NrPixelsZ",
              *(f"p{i}" for i in range(15))):
        assert getattr(ip_mem, k) == pytest.approx(getattr(ip_file, k)), (
            f"file-path vs in-memory differ on {k}: "
            f"file={getattr(ip_file, k)} mem={getattr(ip_mem, k)}"
        )


def test_panel_shifts_file_format_round_trips(tmp_path):
    """v2's write_panel_shifts_file emits 6 columns (id, dY, dZ, dTheta,
    dLsd, dP2). integrate's load_panel_shifts must consume all 6 — if it
    silently dropped dLsd or dP2 the per-panel Lsd / p2 corrections from
    v2 would never reach the integration kernel and per-panel residuals
    would silently inflate.
    """
    from midas_calibrate_v2.compat.to_v1 import write_panel_shifts_file
    from midas_integrate.panel import Panel, load_panel_shifts

    n_panels = 4
    dyz = torch.tensor([
        [0.10, -0.20],
        [0.30,  0.40],
        [-0.50, 0.60],
        [0.70, -0.80],
    ], dtype=torch.float64)
    dth = torch.tensor([1e-3, -2e-3, 3e-3, -4e-3], dtype=torch.float64)
    dlsd = torch.tensor([5.0, -10.0, 15.0, -20.0], dtype=torch.float64)
    dp2 = torch.tensor([1e-5, -2e-5, 3e-5, -4e-5], dtype=torch.float64)

    unpacked = {
        "panel_delta_yz": dyz,
        "panel_delta_theta": dth,
        "panel_delta_lsd": dlsd,
        "panel_delta_p2": dp2,
    }
    panel_path = tmp_path / "panel_shifts.txt"
    write_panel_shifts_file(unpacked, panel_path)

    # Stand up empty Panel slots and load back.
    panels = [Panel(id=k, yMin=0.0, yMax=1.0, zMin=0.0, zMax=1.0)
              for k in range(n_panels)]
    load_panel_shifts(panel_path, panels)

    for k in range(n_panels):
        assert panels[k].dY == pytest.approx(float(dyz[k, 0]))
        assert panels[k].dZ == pytest.approx(float(dyz[k, 1]))
        assert panels[k].dTheta == pytest.approx(float(dth[k]))
        assert panels[k].dLsd == pytest.approx(float(dlsd[k]))
        assert panels[k].dP2 == pytest.approx(float(dp2[k]))


def test_end_to_end_v2_to_integrated_profile(tmp_path):
    """Full pipeline: v2 unpacked dict → from_v2 adapter → build_map →
    integrate constant image → check 1D profile equals the constant.

    This is the closing guard on "v2 calibration parameters drive correct
    integrate output": every link in the chain — distortion remap, geometry
    transfer, map build, CSR construction, area-weighted integration —
    is exercised on a single configuration.
    """
    from midas_integrate.detector_mapper import build_and_write_map
    from midas_integrate.bin_io import load_map
    from midas_integrate.kernels import build_csr, integrate, profile_1d

    NY = NZ = 32
    v2_dict = _make_v2_dict(
        Lsd=1_000_000.0, BC_y=NY / 2.0, BC_z=NZ / 2.0,
        ty=0.0, tz=0.0,
        iso_R2=1.5e-4, a1=2.1e-4, phi1=37.0,
    )
    v2_dict["pxY"] = torch.tensor(200.0, dtype=torch.float64)
    v2_dict["pxZ"] = torch.tensor(200.0, dtype=torch.float64)

    template = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0,
        RhoD=float(NY),
        RMin=2.0, RMax=NY / 2.0 - 2.0, RBinSize=2.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
        SubPixelLevel=1,
    )
    ip = params_from_v2_unpacked(v2_dict, template=template,
                                  warn_on_dropped=False)
    # Distortion remap landed in the right slots:
    assert ip.p2 == pytest.approx(1.5e-4)
    assert ip.p7 == pytest.approx(2.1e-4)
    assert ip.p8 == pytest.approx(37.0)

    build_and_write_map(ip, output_dir=tmp_path, n_jobs=1, verbose=False)
    pm = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    geom = build_csr(pm, n_r=ip.n_r_bins, n_eta=ip.n_eta_bins,
                     n_pixels_y=ip.NrPixelsY, n_pixels_z=ip.NrPixelsZ,
                     dtype=torch.float64, bc_y=ip.BC_y, bc_z=ip.BC_z)
    img = torch.full((ip.NrPixelsZ, ip.NrPixelsY), 4.25, dtype=torch.float64)
    int2d = integrate(img, geom, mode="floor", normalize=True)
    prof = profile_1d(int2d, geom, mode="area_weighted").numpy()

    nonzero = np.where(prof > 0.0)[0]
    assert nonzero.size > 0, "integrated profile is all zero — map build failed"
    np.testing.assert_array_almost_equal(prof[nonzero], 4.25, decimal=6)
