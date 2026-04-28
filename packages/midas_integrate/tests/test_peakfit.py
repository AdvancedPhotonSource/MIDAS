"""Peak fitting smoke tests on synthetic pseudo-Voigt data."""
from __future__ import annotations

import math

import numpy as np

from midas_integrate.peakfit import (
    PF_PARAMS_PER_PEAK,
    fit_peaks,
    fit_peaks_autodetect,
    pseudo_voigt,
    snip_background,
    tch_eta_fwhm,
)


def _make_synthetic(n=400, peaks=[(150, 5.0, 100.0), (250, 6.0, 80.0)], bg=20.0):
    x = np.arange(n, dtype=np.float64)
    y = np.full(n, bg)
    for cx, fwhm, area in peaks:
        # Pure Gaussian for simplicity (TCH gives this in the limit gam→0)
        sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
        y += area / (sigma * math.sqrt(2 * math.pi)) * np.exp(
            -0.5 * ((x - cx) / sigma) ** 2
        )
    return x, y


def test_tch_eta_fwhm_pure_gaussian():
    fwhm, eta = tch_eta_fwhm(sig_centideg2=10.0, gam_centideg=0.0)
    assert eta < 0.05    # mostly Gaussian
    assert fwhm > 0


def test_snip_background_removes_baseline():
    rng = np.random.default_rng(0)
    x = np.arange(200, dtype=np.float64)
    bg = 50.0 + 0.05 * x
    sigma = 4.0
    peak = 200.0 / (sigma * math.sqrt(2 * math.pi)) * np.exp(
        -0.5 * ((x - 100) / sigma) ** 2
    )
    y = bg + peak
    bg_est = snip_background(y, n_iter=80)
    # Background away from the peak should be close to true linear bg.
    np.testing.assert_allclose(bg_est[:50], bg[:50], rtol=0.2, atol=10)
    np.testing.assert_allclose(bg_est[150:], bg[150:], rtol=0.2, atol=10)


def test_fit_peaks_recovers_centers():
    x, y = _make_synthetic(n=400, peaks=[(150, 5.0, 100.0), (250, 6.0, 80.0)])
    out = fit_peaks(x, y, peak_locations=[150, 250],
                    x_bin_size=1.0, fit_roi_padding=30)
    assert out.shape == (2, PF_PARAMS_PER_PEAK)
    # Centers should land near 150 and 250 (within ~2 bins given the
    # least_squares optimizer)
    assert abs(out[0, 1] - 150) < 2
    assert abs(out[1, 1] - 250) < 2
    # Areas should be positive
    assert out[0, 0] > 0
    assert out[1, 0] > 0


def test_fit_peaks_autodetect_finds_peaks():
    x, y = _make_synthetic(n=400, peaks=[(150, 5.0, 100.0), (250, 6.0, 80.0)],
                           bg=10.0)
    out = fit_peaks_autodetect(x, y, max_peaks=3,
                               x_bin_size=1.0, fit_roi_padding=30)
    assert out.shape == (3, PF_PARAMS_PER_PEAK)
    # At least the strongest peaks should be found
    centers_found = sorted(out[out[:, 0] > 0, 1].tolist())
    assert len(centers_found) >= 2
    # Closest to 150 must be within 5 bins
    assert any(abs(c - 150) < 5 for c in centers_found)
    assert any(abs(c - 250) < 5 for c in centers_found)
