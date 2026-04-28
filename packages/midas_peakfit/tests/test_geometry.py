"""Test geometry helpers — eta angle, distortion, goodCoords."""
import numpy as np
import pytest

from midas_peakfit.geometry import (
    calc_eta_angle,
    calc_eta_angle_np,
    yz_from_r_eta,
    tilt_matrix,
    compute_good_coords,
)
from midas_peakfit.params import ZarrParams


def test_calc_eta_angle_quadrants():
    # y=0, z=1 → alpha=0, no negation
    assert calc_eta_angle(0.0, 1.0) == 0.0
    # y=1, z=0 → alpha=90, negated → -90
    assert calc_eta_angle(1.0, 0.0) == -90.0
    # y=-1, z=0 → alpha=90, NOT negated → +90
    assert calc_eta_angle(-1.0, 0.0) == 90.0
    # y=0, z=-1 → alpha=180, no negation (y not > 0) → 180
    assert calc_eta_angle(0.0, -1.0) == 180.0


def test_calc_eta_angle_vectorized_matches_scalar():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    z = rng.normal(size=20)
    expected = np.array([calc_eta_angle(yi, zi) for yi, zi in zip(y, z)])
    np.testing.assert_allclose(calc_eta_angle_np(y, z), expected, atol=1e-12)


def test_yz_from_r_eta_round_trip():
    # Pick (Y, Z), compute (R, Eta), invert back
    y = np.array([10.0, -5.0, 3.0])
    z = np.array([0.0, 7.0, -2.0])
    R = np.sqrt(y * y + z * z)
    Eta = calc_eta_angle_np(-y, z)  # follow C convention
    yp, zp = yz_from_r_eta(R, Eta)
    # yzFromREta: Y = -R sin(Eta), Z = R cos(Eta) → recovers (-y, z) by definition
    np.testing.assert_allclose(yp, -y, atol=1e-10)
    np.testing.assert_allclose(zp, z, atol=1e-10)


def test_tilt_matrix_identity():
    M = tilt_matrix(0, 0, 0)
    np.testing.assert_allclose(M, np.eye(3), atol=1e-12)


def test_compute_good_coords_full_image():
    p = ZarrParams()
    p.NrPixels = 64
    p.DoFullImage = 1
    p.Thresholds = [42.0]
    p.RingNrs = [1]
    p.nRingsThresh = 1
    gc = compute_good_coords(p, panels=[], ringRads=None)
    assert gc.shape == (64, 64)
    np.testing.assert_array_equal(gc, np.full((64, 64), 42.0))
