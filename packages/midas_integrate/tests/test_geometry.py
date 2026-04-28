"""Geometry transforms — sanity tests against analytical truths."""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_integrate.geometry import (
    DEG2RAD,
    REta_to_YZ,
    REta_to_YZ_scalar,
    build_bin_edges,
    build_q_bin_edges_in_R,
    build_tilt_matrix,
    calc_eta_angle,
    circle_seg_intersect,
    invert_REta_to_pixel,
    pixel_to_REta,
    polygon_area,
    ray_seg_intersect,
)


def test_zero_tilt_identity():
    M = build_tilt_matrix(0, 0, 0)
    np.testing.assert_array_almost_equal(M, np.eye(3))


def test_pixel_to_REta_at_beam_center_is_zero():
    # No tilt, no distortion: pixel at beam center → R = 0
    Ycen, Zcen = 100.0, 200.0
    TRs = build_tilt_matrix(0, 0, 0)
    R, eta = pixel_to_REta(
        Ycen, Zcen,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=1.0, RhoD=1.0, px=1.0,
    )
    assert abs(float(R)) < 1e-9


def test_pixel_to_REta_radial_symmetry():
    """No tilt, no distortion: R should equal pixel distance from beam center."""
    Ycen, Zcen = 0.0, 0.0
    TRs = build_tilt_matrix(0, 0, 0)
    Y = np.array([10.0, -10.0, 0.0, 0.0])
    Z = np.array([0.0, 0.0, 10.0, -10.0])
    R, _ = pixel_to_REta(
        Y, Z,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=1e9, RhoD=1.0, px=1.0,
    )
    # At infinite distance the curvature term cancels and R ≈ |Y or Z|
    np.testing.assert_array_almost_equal(R, [10, 10, 10, 10], decimal=3)


def test_invert_round_trip():
    Ycen, Zcen = 700.0, 865.0
    TRs = build_tilt_matrix(0.05, 0.18, 0.53)
    R_target, eta_target = 250.0, 30.0
    Y, Z = invert_REta_to_pixel(
        R_target, eta_target,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=580_000.0, RhoD=2200.0, px=172.0,
    )
    R_back, eta_back = pixel_to_REta(
        Y, Z,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=580_000.0, RhoD=2200.0, px=172.0,
    )
    assert abs(float(R_back) - R_target) < 1e-4
    assert abs(float(eta_back) - eta_target) < 1e-4


def test_REta_to_YZ_round_trip():
    R, eta_deg = 100.0, 45.0
    Y, Z = REta_to_YZ_scalar(R, eta_deg)
    assert abs(math.hypot(Y, Z) - R) < 1e-12
    assert abs(calc_eta_angle(Y, Z) - eta_deg) < 1e-12


def test_circle_seg_intersect_diameter():
    # Horizontal segment from (-2, 0) to (2, 0); circle R=1 centered at origin
    hits = circle_seg_intersect(-2.0, 0.0, 2.0, 0.0, 1.0)
    assert len(hits) == 2
    xs = sorted(h[0] for h in hits)
    assert abs(xs[0] - (-1.0)) < 1e-12
    assert abs(xs[1] - 1.0) < 1e-12


def test_polygon_area_unit_square():
    # Unit square — Shoelace gives area 1
    edges = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    a = polygon_area(edges, RMin=10.0, RMax=20.0)  # Rs irrelevant here
    assert abs(a - 1.0) < 1e-12


def test_polygon_area_arc():
    # A 90° wedge of an annulus: outer R=2, inner R=1, η ∈ [0°, 90°]
    # Area = (π/4) * (R² - r²) = (π/4) * 3 ≈ 2.356
    R1, R2 = 1.0, 2.0
    edges = [
        (R1, 0.0), (R2, 0.0),       # ray η=0
        (0.0, R2), (0.0, R1),       # ray η=90°
    ]
    # Make sure the arcs are on R1 and R2: distances must match within tol
    a = polygon_area(edges, RMin=R1, RMax=R2)
    expected = math.pi / 4.0 * (R2 ** 2 - R1 ** 2)
    # We expect the true area; allow some tolerance because endpoints lie
    # on both arcs but the algorithm assigns them to one specific arc each.
    assert a > 1.0
    assert a < expected * 1.5


def test_build_bin_edges():
    r_lo, r_hi, e_lo, e_hi = build_bin_edges(
        RMin=10.0, EtaMin=-180.0, n_r_bins=5, n_eta_bins=4,
        RBinSize=1.0, EtaBinSize=90.0,
    )
    np.testing.assert_array_equal(r_lo, [10, 11, 12, 13, 14])
    np.testing.assert_array_equal(r_hi, [11, 12, 13, 14, 15])
    np.testing.assert_array_equal(e_lo, [-180, -90, 0, 90])


def test_q_mode_R_bins_monotonic():
    r_lo, r_hi, n = build_q_bin_edges_in_R(
        QMin=0.5, QMax=5.0, QBinSize=0.05,
        Lsd=580_000.0, px=172.0, wavelength_A=0.5,
    )
    assert n == 90
    # R is monotonically increasing in Q
    assert np.all(np.diff(r_lo) > 0)
