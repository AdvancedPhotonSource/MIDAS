"""Tests for the Modregger 2025 moment-based shot-noise σ and the per-peak
photon-count quality flag.

References
----------
- Closed-form formulas: Modregger et al., J. Appl. Cryst. 58, 1653 (2025),
  equations (6)–(8).
- Thresholds for quality flag: midas_peakfit paper Appendix A
  ("Practical caveats"), 5 / 25 photon-count boundaries.
"""
import numpy as np
import pytest

from midas_peakfit.uncertainty import (
    M0_DEEP_POISSON_THRESHOLD,
    M0_MARGINAL_THRESHOLD,
    QUALITY_DEEP_POISSON,
    QUALITY_MARGINAL,
    QUALITY_OK,
    classify_peak_quality,
    compute_moment_sigma,
)


# ── compute_moment_sigma ───────────────────────────────────────────────


def test_compute_moment_sigma_scalar_basic():
    """A single Gaussian peak with M_0=10000, σ=1 px (M_2=1, M_4=3) should
    give u(M_1) = sqrt(1/10000) = 1e-2 px and u(M_0) = sqrt(10000) = 100."""
    u_M0, u_M1, u_M2 = compute_moment_sigma(M0=10000.0, M2=1.0, M4=3.0, dx=1.0)
    assert u_M0 == pytest.approx(100.0)
    assert u_M1 == pytest.approx(1e-2)
    # u(M_2) = sqrt((M_4 − M_2²) / M_0) = sqrt(2/10000)
    assert u_M2 == pytest.approx(np.sqrt(2.0 / 10000.0))


def test_compute_moment_sigma_dx_scaling():
    """u(M_0), u(M_1), u(M_2) all scale as sqrt(dx)."""
    args = dict(M0=1000.0, M2=4.0, M4=48.0)  # 48 ≥ M_2² = 16
    u0_dx1 = compute_moment_sigma(**args, dx=1.0)
    u0_dx4 = compute_moment_sigma(**args, dx=4.0)
    for a, b in zip(u0_dx1, u0_dx4):
        assert b == pytest.approx(a * 2.0)  # sqrt(4) = 2


def test_compute_moment_sigma_vectorised():
    """Vector inputs produce vector outputs of the same shape."""
    M0 = np.array([100.0, 1000.0, 10000.0])
    M2 = np.array([1.0, 1.0, 1.0])
    M4 = np.array([3.0, 3.0, 3.0])
    u0, u1, u2 = compute_moment_sigma(M0, M2, M4)
    assert u0.shape == M0.shape
    np.testing.assert_allclose(u0, np.sqrt(M0))
    np.testing.assert_allclose(u1, np.sqrt(1.0 / M0))
    np.testing.assert_allclose(u2, np.sqrt(2.0 / M0))


def test_compute_moment_sigma_handles_zero_photons():
    """M_0 ≤ 0 → NaN (formulas not applicable). Should not raise."""
    u0, u1, u2 = compute_moment_sigma(
        np.array([0.0, -1.0, 5.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([3.0, 3.0, 3.0]),
    )
    assert np.isnan(u0[0]) and np.isnan(u0[1])
    assert np.isnan(u1[0]) and np.isnan(u1[1])
    assert np.isnan(u2[0]) and np.isnan(u2[1])
    # Third entry is finite
    assert np.isfinite(u0[2])
    assert np.isfinite(u1[2])
    assert np.isfinite(u2[2])


def test_compute_moment_sigma_kurtosis_clamp():
    """M_4 ≥ M_2² always holds in exact arithmetic (kurtosis ≥ 1). If
    floating-point cancellation gives a slightly negative argument, the
    function must not return NaN — it must clip to 0."""
    M0, M2 = 1000.0, 1.0
    M4 = M2 * M2 - 1e-14  # tiny roundoff under the bound
    _, _, u2 = compute_moment_sigma(M0, M2, M4)
    assert u2 == 0.0


def test_compute_moment_sigma_monte_carlo_gaussian():
    """Empirical Monte-Carlo validation against synthetic Poisson-noised
    Gaussian peaks. Theoretical u(M_1) should match the sample σ of the
    moment estimator to within ~5% over 2000 trials."""
    rng = np.random.default_rng(0)
    n_trials = 2000
    n_bins = 41
    x = np.arange(n_bins) - n_bins // 2  # centered, dx = 1
    sigma_true = 2.5
    M0_target = 5000.0
    profile = M0_target * np.exp(-x * x / (2 * sigma_true ** 2))
    profile = profile / profile.sum() * M0_target

    centroids = np.empty(n_trials)
    for t in range(n_trials):
        noisy = rng.poisson(profile).astype(np.float64)
        M0 = noisy.sum()
        if M0 <= 0:
            centroids[t] = 0.0
            continue
        centroids[t] = (x * noisy).sum() / M0

    sample_sigma = centroids.std(ddof=1)
    # Theoretical floor — use the *true* M_2 (centroid spread = sigma_true²).
    _, theory_u_M1, _ = compute_moment_sigma(
        M0=M0_target, M2=sigma_true ** 2, M4=3 * sigma_true ** 4, dx=1.0
    )
    # 5 % matches the correlation coefficient (r > 0.95) reported across
    # all three experimental setups in Modregger 2025.
    assert sample_sigma == pytest.approx(theory_u_M1, rel=0.05)


# ── classify_peak_quality ──────────────────────────────────────────────


def test_classify_quality_thresholds():
    """0 = OK (≥ 25), 1 = marginal (5–25), 2 = deep-Poisson (<5)."""
    M0 = np.array([0.0, 1.0, 4.999, 5.0, 10.0, 24.999, 25.0, 100.0, 1e6])
    expected = np.array([
        QUALITY_DEEP_POISSON,
        QUALITY_DEEP_POISSON,
        QUALITY_DEEP_POISSON,
        QUALITY_MARGINAL,
        QUALITY_MARGINAL,
        QUALITY_MARGINAL,
        QUALITY_OK,
        QUALITY_OK,
        QUALITY_OK,
    ], dtype=np.int8)
    flags = classify_peak_quality(M0)
    np.testing.assert_array_equal(flags, expected)
    assert flags.dtype == np.int8


def test_classify_quality_thresholds_match_constants():
    """Thresholds are the named constants — guards against future drift."""
    assert M0_DEEP_POISSON_THRESHOLD == 5.0
    assert M0_MARGINAL_THRESHOLD == 25.0
    assert classify_peak_quality(M0_DEEP_POISSON_THRESHOLD - 1e-9) == QUALITY_DEEP_POISSON
    assert classify_peak_quality(M0_DEEP_POISSON_THRESHOLD) == QUALITY_MARGINAL
    assert classify_peak_quality(M0_MARGINAL_THRESHOLD - 1e-9) == QUALITY_MARGINAL
    assert classify_peak_quality(M0_MARGINAL_THRESHOLD) == QUALITY_OK


def test_classify_quality_scalar_returns_python_int():
    """Scalar input → Python int, not 0-d array."""
    flag = classify_peak_quality(50.0)
    assert isinstance(flag, int)
    assert flag == QUALITY_OK
