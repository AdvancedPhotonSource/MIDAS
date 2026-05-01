"""Tests for variance propagation, tilt-aware solid angle, generalized polarization.

These tests cover the corrections requested by the JAC reviewer:
  - Poisson variance propagation through the SpMV pipeline.
  - Tilt-aware solid-angle correction reducing to cos^3(2θ) at zero tilt.
  - Generalized polarization with arbitrary plane angle η_pol.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_integrate.bin_io import write_synthetic_map, load_map
from midas_integrate.geometry import build_tilt_matrix, solid_angle_factor
from midas_integrate.kernels import (
    AREA_THRESHOLD,
    build_csr,
    integrate,
    integrate_with_variance,
    profile_1d,
    profile_1d_with_variance,
)


# ─────────────────────────────────────────────────────────────────────────────
# Variance propagation
# ─────────────────────────────────────────────────────────────────────────────
def _make_simple_pixmap(rng, NY, NZ, n_r, n_eta, avg_pix_per_bin=4):
    """Random in-bounds pxList for variance/sa tests (no out-of-bounds entries)."""
    n_bins = n_r * n_eta
    rec_list = []
    bin_lists = []
    cur = 0
    for b in range(n_bins):
        if rng.random() < 0.05:
            bin_lists.append([])
            continue
        n = int(rng.poisson(avg_pix_per_bin)) + 1
        idxs = []
        for _ in range(n):
            y = float(rng.integers(0, NY))
            z = float(rng.integers(0, NZ))
            frac = float(rng.uniform(0.05, 1.0))
            dr = float(rng.uniform(-0.4, 0.4))
            aw = float(rng.uniform(0.1, 1.0))
            rec_list.append((y, z, frac, dr, aw))
            idxs.append(cur)
            cur += 1
        bin_lists.append(idxs)
    return rec_list, bin_lists


def _reference_variance_floor(image, pxList, counts, offsets, NY, NZ, normalize=True):
    """Reference: Var(I_b) = Σ_l W_bl² · I_l, where W_bl coalesces multiple
    entries that hit the same (bin, pixel)."""
    n_bins = counts.shape[0]
    var_out = np.zeros(n_bins, dtype=np.float64)
    flat = image.reshape(-1)
    for b in range(n_bins):
        n = int(counts[b])
        if n == 0:
            continue
        s = int(offsets[b])
        # Coalesce by pixel
        per_pix: dict[int, float] = {}
        A = 0.0
        for k in range(n):
            e = pxList[s + k]
            y = int(e["y"])
            z = int(e["z"])
            if y < 0 or y >= NY or z < 0 or z >= NZ:
                continue
            offset = z * NY + y
            per_pix[offset] = per_pix.get(offset, 0.0) + float(e["frac"])
            A += float(e["areaWeight"])
        V = 0.0
        for offset, W in per_pix.items():
            V += (W * W) * float(flat[offset])
        if A > AREA_THRESHOLD and normalize:
            var_out[b] = V / (A * A)
        else:
            var_out[b] = V
    return var_out


def test_variance_propagation_floor_matches_reference(tmp_path):
    rng = np.random.default_rng(7)
    NY, NZ = 32, 24
    n_r, n_eta = 8, 9
    rec_list, bin_lists = _make_simple_pixmap(rng, NY, NZ, n_r, n_eta)
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=rec_list, bin_pixel_lists=bin_lists,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")

    geom = build_csr(pixmap, n_r=n_r, n_eta=n_eta, n_pixels_y=NY, n_pixels_z=NZ,
                     dtype=torch.float64, compute_variance=True)
    image = rng.uniform(10.0, 1000.0, size=(NZ, NY)).astype(np.float64)

    intensity, variance = integrate_with_variance(
        torch.from_numpy(image), geom, mode="floor", normalize=True,
    )

    expected_var = _reference_variance_floor(
        image, pixmap.pxList, pixmap.counts, pixmap.offsets, NY, NZ, normalize=True,
    ).reshape(n_r, n_eta)

    np.testing.assert_allclose(variance.numpy(), expected_var, rtol=1e-9, atol=1e-12)


def test_variance_propagation_explicit_variance_image(tmp_path):
    """Pass an explicit variance map ≠ image; propagation should use it."""
    rng = np.random.default_rng(11)
    NY, NZ = 24, 20
    n_r, n_eta = 6, 8
    rec_list, bin_lists = _make_simple_pixmap(rng, NY, NZ, n_r, n_eta)
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=rec_list, bin_pixel_lists=bin_lists,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    geom = build_csr(pixmap, n_r=n_r, n_eta=n_eta, n_pixels_y=NY, n_pixels_z=NZ,
                     dtype=torch.float64, compute_variance=True)

    image = rng.uniform(10.0, 1000.0, size=(NZ, NY)).astype(np.float64)
    var_map = rng.uniform(1.0, 50.0, size=(NZ, NY)).astype(np.float64)

    _, var_explicit = integrate_with_variance(
        torch.from_numpy(image), geom, mode="floor", normalize=True,
        variance_image=torch.from_numpy(var_map),
    )
    expected = _reference_variance_floor(
        var_map, pixmap.pxList, pixmap.counts, pixmap.offsets, NY, NZ, normalize=True,
    ).reshape(n_r, n_eta)
    np.testing.assert_allclose(var_explicit.numpy(), expected, rtol=1e-9, atol=1e-12)


def test_profile_1d_with_variance_consistent(tmp_path):
    rng = np.random.default_rng(13)
    NY, NZ = 24, 20
    n_r, n_eta = 6, 8
    rec_list, bin_lists = _make_simple_pixmap(rng, NY, NZ, n_r, n_eta)
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=rec_list, bin_pixel_lists=bin_lists,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    geom = build_csr(pixmap, n_r=n_r, n_eta=n_eta, n_pixels_y=NY, n_pixels_z=NZ,
                     dtype=torch.float64, compute_variance=True)

    image = rng.uniform(10.0, 1000.0, size=(NZ, NY)).astype(np.float64)
    int2d, var2d = integrate_with_variance(
        torch.from_numpy(image), geom, mode="floor", normalize=True,
    )
    I_1d, V_1d = profile_1d_with_variance(int2d, var2d, geom, mode="area_weighted")

    # I_1d must agree with profile_1d on the intensity.
    I_ref = profile_1d(int2d, geom, mode="area_weighted")
    np.testing.assert_allclose(I_1d.numpy(), I_ref.numpy(), rtol=1e-12)

    # V_1d must equal Σ_η (A² · Var) / (Σ_η A)²  computed manually.
    area_2d = geom.area_per_bin.reshape(n_r, n_eta).numpy()
    valid = area_2d > AREA_THRESHOLD
    A_sum = (area_2d * valid).sum(axis=1)
    AA = (area_2d * valid) ** 2
    V_expected = (var2d.numpy() * AA).sum(axis=1) / np.maximum(A_sum, AREA_THRESHOLD) ** 2
    V_expected = np.where(A_sum > AREA_THRESHOLD, V_expected, 0.0)
    np.testing.assert_allclose(V_1d.numpy(), V_expected, rtol=1e-9, atol=1e-12)


def test_variance_unbuilt_raises(tmp_path):
    """integrate_with_variance must fail loudly when the squared CSR was not built."""
    rng = np.random.default_rng(2)
    NY, NZ, n_r, n_eta = 16, 12, 4, 6
    rec_list, bin_lists = _make_simple_pixmap(rng, NY, NZ, n_r, n_eta)
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=rec_list, bin_pixel_lists=bin_lists,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    geom = build_csr(pixmap, n_r=n_r, n_eta=n_eta, n_pixels_y=NY, n_pixels_z=NZ,
                     dtype=torch.float64, compute_variance=False)
    image = torch.zeros((NZ, NY), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="compute_variance=True"):
        integrate_with_variance(image, geom, mode="floor")


# ─────────────────────────────────────────────────────────────────────────────
# Tilt-aware solid angle
# ─────────────────────────────────────────────────────────────────────────────
def test_solid_angle_zero_tilt_matches_cos_cubed():
    """At zero tilt, Ω/Ω_ref must equal cos³(2θ) per pixel."""
    NY, NZ = 64, 64
    Lsd = 200_000.0  # μm
    px = 200.0
    Ycen = NY / 2.0
    Zcen = NZ / 2.0
    TRs = build_tilt_matrix(0.0, 0.0, 0.0)

    yy, zz = np.meshgrid(np.arange(NY, dtype=np.float64),
                         np.arange(NZ, dtype=np.float64))
    sa = solid_angle_factor(yy, zz, Ycen=Ycen, Zcen=Zcen,
                            TRs=TRs, Lsd=Lsd, px=px)

    Yc = (-yy + Ycen) * px
    Zc = (zz - Zcen) * px
    # zero tilt: XYZ = (Lsd, Yc, Zc); 2θ = atan(sqrt(Yc²+Zc²)/Lsd)
    R = np.sqrt(Yc * Yc + Zc * Zc)
    two_theta = np.arctan(R / Lsd)
    expected = np.cos(two_theta) ** 3

    np.testing.assert_allclose(sa, expected, rtol=1e-10, atol=1e-12)


def test_solid_angle_nonzero_tilt_differs_from_cos_cubed():
    """At nonzero tilt, the tilt-aware factor must DEPART from naive cos³(2θ).

    The naive flat-detector form treats r = (Lsd, y, z) regardless of
    tilt; the tilt-aware form rotates the position vector with the
    detector. Under a 20° tilt those differ by O(few percent) over the
    detector surface, which is exactly the regime the reviewer flagged.
    """
    NY, NZ = 64, 64
    Lsd = 200_000.0
    px = 200.0
    Ycen = NY / 2.0
    Zcen = NZ / 2.0
    TRs = build_tilt_matrix(0.0, 20.0, 0.0)
    yy, zz = np.meshgrid(np.arange(NY, dtype=np.float64),
                         np.arange(NZ, dtype=np.float64))
    sa = solid_angle_factor(yy, zz, Ycen=Ycen, Zcen=Zcen,
                            TRs=TRs, Lsd=Lsd, px=px)
    # The naive cos³(2θ) using the *flat-detector* 2θ-from-radius:
    Yc = (-yy + Ycen) * px
    Zc = (zz - Zcen) * px
    R = np.sqrt(Yc * Yc + Zc * Zc)
    two_theta = np.arctan(R / Lsd)
    naive = np.cos(two_theta) ** 3
    rel_diff = np.abs(sa - naive) / np.maximum(naive, 1e-10)
    assert rel_diff.max() > 0.01, (
        f"Tilt-aware SA should diverge from cos³(2θ) under 20° tilt; "
        f"max rel diff was {rel_diff.max():.4e}"
    )


def test_solid_angle_beam_center_zero_tilt_is_unity():
    """Ω/Ω_ref = 1 at the beam center for a zero-tilt detector."""
    Lsd = 1e6
    px = 100.0
    Ycen = 100.5
    Zcen = 100.5
    TRs = build_tilt_matrix(0.0, 0.0, 0.0)
    sa = solid_angle_factor(np.array([Ycen]), np.array([Zcen]),
                            Ycen=Ycen, Zcen=Zcen, TRs=TRs,
                            Lsd=Lsd, px=px)
    assert abs(float(sa.item()) - 1.0) < 1e-9


def test_solid_angle_beam_center_under_tilt_is_cos_tilt():
    """For a single tilt about Y by angle t, the beam-center pixel's
    Ω/Ω_ref = cos(t), reflecting the fact that the detector normal is
    rotated away from the beam axis. This matches pyFAI's geometric
    solid-angle convention (the absolute Ω, not a tilt-aware-renormalized
    quantity) and is the geometric correction we want to apply per pixel
    so that off-axis pixels are corrected accurately at large tilt."""
    Lsd = 1e6
    px = 100.0
    Ycen = 100.5
    Zcen = 100.5
    for ty_deg in [5.0, 10.0, 20.0]:
        TRs = build_tilt_matrix(0.0, ty_deg, 0.0)
        sa = solid_angle_factor(np.array([Ycen]), np.array([Zcen]),
                                Ycen=Ycen, Zcen=Zcen, TRs=TRs,
                                Lsd=Lsd, px=px)
        expected = math.cos(math.radians(ty_deg))
        assert abs(float(sa.item()) - expected) < 1e-9, (
            f"ty={ty_deg}°: sa={float(sa.item())} expected={expected}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Generalized polarization
# ─────────────────────────────────────────────────────────────────────────────
def _polarization_factor(two_theta_rad: float, eta_deg: float,
                         pol_fraction: float, eta_pol_deg: float) -> float:
    eta_rad = math.radians(eta_deg - eta_pol_deg)
    return 1.0 - pol_fraction * math.sin(two_theta_rad) ** 2 * math.cos(eta_rad) ** 2


def test_polarization_eta_pol_zero_matches_legacy():
    """η_pol = 0 must reproduce the legacy cos²(η) form exactly."""
    for two_theta in [0.05, 0.3, 1.0]:
        for eta in [-180, -90, 0, 30, 90, 180]:
            legacy = 1.0 - 0.95 * math.sin(two_theta) ** 2 * math.cos(math.radians(eta)) ** 2
            new = _polarization_factor(two_theta, eta, 0.95, 0.0)
            assert abs(new - legacy) < 1e-12


def test_polarization_eta_pol_rotates_pattern():
    """Setting η_pol = 90° must move the polarization-loss minima from η=0,180
    to η=±90."""
    two_theta = 0.5
    f = 0.95
    p_eta0_at_eta0 = _polarization_factor(two_theta, 0.0, f, 0.0)
    p_eta0_at_eta90 = _polarization_factor(two_theta, 90.0, f, 0.0)
    # Legacy: maximum loss at η=0, minimum at η=±90.
    assert p_eta0_at_eta0 < p_eta0_at_eta90
    # Now η_pol = 90 should swap them.
    p_eta90_at_eta0 = _polarization_factor(two_theta, 0.0, f, 90.0)
    p_eta90_at_eta90 = _polarization_factor(two_theta, 90.0, f, 90.0)
    assert p_eta90_at_eta0 > p_eta90_at_eta90
    # Symmetry: η_pol=0 at η=0  ==  η_pol=90 at η=90  (90° rotation).
    assert abs(p_eta0_at_eta0 - p_eta90_at_eta90) < 1e-12
    assert abs(p_eta0_at_eta90 - p_eta90_at_eta0) < 1e-12


def test_polarization_unpolarized_is_isotropic():
    """f=0 → polarization factor is 1 everywhere."""
    for two_theta in [0.05, 1.0]:
        for eta in [-180, 0, 45, 180]:
            for pol_eta in [0.0, 30.0, 90.0]:
                p = _polarization_factor(two_theta, eta, 0.0, pol_eta)
                assert abs(p - 1.0) < 1e-12
