"""Pure-Python pseudo-Voigt peak fitting — port of FF_HEDM/src/PeakFit.c.

Implements the same area-normalized GSAS-II Thompson-Cox-Hastings (TCH)
pseudo-Voigt model with a 2-term Chebyshev background. The C code uses
NLopt L-BFGS with Nelder-Mead fallback; here we use
``scipy.optimize.least_squares`` (Trust-Region-Reflective with bounds) which
is mathematically equivalent for the smooth-residual case and gives bit-near
agreement with the C optimizer.

Output schema per peak (PF_PARAMS_PER_PEAK = 7 doubles, matches C exactly)::

    [0] area      integrated intensity above background
    [1] center    fitted peak position (same units as input x)
    [2] sig       Gaussian variance (centideg² when x is in degrees)
    [3] gam       Lorentzian FWHM  (centideg when x is in degrees)
    [4] FWHM      total FWHM (same units as x)
    [5] eta       pseudo-Voigt mixing (0=Gaussian, 1=Lorentzian)
    [6] chi_sq    Σ(residual²) / dof
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter

PF_PARAMS_PER_PEAK = 7   # match PeakFit.h
MAX_PEAK_LOCATIONS_PF = 200


# ─────────────────────────────────────────────────────────────────────────────
# Thompson-Cox-Hastings: derive total FWHM and mixing eta
# ─────────────────────────────────────────────────────────────────────────────
def tch_eta_fwhm(sig_centideg2: float, gam_centideg: float
                 ) -> tuple[float, float]:
    """Mirror pf_tch_eta_fwhm. Returns (FWHM_deg, eta in [0,1])."""
    fg = math.sqrt(max(8.0 * math.log(2.0) * max(sig_centideg2, 1e-12), 0.0)) / 100.0
    fl = max(gam_centideg, 1e-6) / 100.0
    fg2, fg3, fg4, fg5 = fg * fg, fg ** 3, fg ** 4, fg ** 5
    fl2, fl3, fl4, fl5 = fl * fl, fl ** 3, fl ** 4, fl ** 5
    FWHM = (fg5 + 2.69269 * fg4 * fl + 2.42843 * fg3 * fl2
            + 4.47163 * fg2 * fl3 + 0.07842 * fg * fl4 + fl5) ** 0.2
    if FWHM < 1e-15:
        FWHM = 1e-15
    ratio = fl / FWHM
    eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio
    return FWHM, max(0.0, min(1.0, eta))


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-Voigt model evaluation (vectorized over x)
# ─────────────────────────────────────────────────────────────────────────────
def pseudo_voigt(
    x: np.ndarray,
    n_peaks: int,
    params: Sequence[float],
    *,
    x_lo: float, x_hi: float,
) -> np.ndarray:
    """Compute the area-normalized pV model at points ``x``.

    ``params`` layout: [Area0, Center0, sig0, gam0, ..., bg0, bg1].
    Mirror of pf_calculate_model.
    """
    x = np.asarray(x, dtype=np.float64)
    bg0 = params[n_peaks * 4]
    bg1 = params[n_peaks * 4 + 1]
    x_range = max(x_hi - x_lo, 1e-12)
    x_norm = 2.0 * (x - x_lo) / x_range - 1.0
    out = bg0 + bg1 * x_norm
    for p in range(n_peaks):
        area = params[p * 4 + 0]
        center = params[p * 4 + 1]
        sig = max(params[p * 4 + 2], 1e-12)
        gam = max(params[p * 4 + 3], 1e-6)
        FWHM, eta = tch_eta_fwhm(sig, gam)
        if FWHM < 1e-15:
            FWHM = 1e-15
        sigma_g = FWHM / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        G_norm = 1.0 / (sigma_g * math.sqrt(2.0 * math.pi))
        half_fwhm = FWHM / 2.0
        L_norm = half_fwhm / math.pi
        dx = x - center
        G = G_norm * np.exp(-0.5 * dx * dx / (sigma_g * sigma_g))
        L = L_norm / (dx * dx + half_fwhm * half_fwhm)
        out = out + area * (eta * L + (1.0 - eta) * G)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SNIP background (Morháč LLS)
# ─────────────────────────────────────────────────────────────────────────────
def snip_background(intensity: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """SNIP background estimation. Mirror of pf_snip_background."""
    intensity = np.asarray(intensity, dtype=np.float64)
    n = intensity.size
    if n_iter <= 0 or n == 0:
        return np.zeros_like(intensity)
    y = np.maximum(intensity, 0.0)
    v = np.log(np.log(np.sqrt(y + 1.0) + 1.0) + 1.0)
    for p in range(n_iter, 0, -1):
        if 2 * p >= n:
            continue
        # Replace v[p:n-p] with min(v[p:n-p], (v[0:n-2p] + v[2p:n]) / 2)
        avg = 0.5 * (v[: n - 2 * p] + v[2 * p :])
        v[p : n - p] = np.minimum(v[p : n - p], avg)
    w = np.exp(np.exp(v) - 1.0) - 1.0
    bg = w * w - 1.0
    return np.maximum(bg, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Initial-parameter estimator (mirror pf_estimate_initial_params)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_initial_params(
    intensity_data: np.ndarray, peak_idx_local: int,
) -> Tuple[float, float, float]:
    """Returns (FWHM_bins, bg_guess, amp_guess)."""
    n = intensity_data.size
    bg_width = max(1, min(5, n // 4))
    bg_sum = float(intensity_data[:bg_width].sum() + intensity_data[-bg_width:].sum())
    bg = bg_sum / (2.0 * bg_width)
    amp = float(intensity_data[peak_idx_local]) - bg
    if amp <= 0:
        amp = float(intensity_data[peak_idx_local])
    half_max = bg + amp / 2.0
    # walk left until we drop below half-max
    left = peak_idx_local
    while left > 0 and intensity_data[left] > half_max:
        left -= 1
    right = peak_idx_local
    while right < n - 1 and intensity_data[right] > half_max:
        right += 1
    fwhm = max(right - left, 2)
    return float(fwhm), bg, amp


# ─────────────────────────────────────────────────────────────────────────────
# Peak detection (analog of pf_detect_peaks)
# ─────────────────────────────────────────────────────────────────────────────
def _smooth_savgol(y: np.ndarray, window: int = 7) -> np.ndarray:
    if y.size < window:
        return y.copy()
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=2)


def detect_peaks(
    x: np.ndarray, corrected: np.ndarray, *,
    max_peaks: int, min_separation: float,
) -> np.ndarray:
    """Auto-detect peaks via negative 2nd derivative, ranked by prominence.

    Returns x-locations (NOT indices) of up to ``max_peaks``, sorted by x.
    """
    n = corrected.size
    if n < 10 or max_peaks <= 0:
        return np.array([], dtype=np.float64)
    smooth = _smooth_savgol(corrected, window=7)
    # Negative 2nd derivative
    neg_d2 = np.zeros(n, dtype=np.float64)
    neg_d2[1:-1] = -(smooth[2:] - 2.0 * smooth[1:-1] + smooth[:-2])
    bin_width = abs(x[1] - x[0]) if n > 1 else 1.0
    min_sep_bins = max(3, int(min_separation / bin_width))

    # scipy.signal.find_peaks with prominence is the cleanest analog
    peak_indices, props = find_peaks(neg_d2, distance=min_sep_bins,
                                     prominence=0.0)
    # Filter: require corrected[i] > 0
    mask = corrected[peak_indices] > 0.0
    peak_indices = peak_indices[mask]
    prominences = props["prominences"][mask]
    # Rank by prominence desc, take top max_peaks
    if peak_indices.size > max_peaks:
        order = np.argsort(prominences)[::-1][:max_peaks]
        peak_indices = peak_indices[order]
    # Sort by position
    peak_indices = np.sort(peak_indices)
    return x[peak_indices].astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# ROI partitioning (split independent peak groups)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _FitJob:
    start: int
    end: int
    peak_indices: List[int]
    peak_x: List[float]
    peak_intensities: List[float]


def _partition_into_jobs(
    peak_indices: Sequence[int],
    peak_x: Sequence[float],
    peak_intensities: Sequence[float],
    *,
    n_bins: int,
    fit_roi_padding: int,
) -> List[_FitJob]:
    """Group peaks whose ROIs overlap into one joint fit. Mirrors the
    job-construction logic of fitPeaks() in PeakFit.c.
    """
    if len(peak_indices) == 0:
        return []
    order = sorted(range(len(peak_indices)), key=lambda i: peak_indices[i])
    pidx = [peak_indices[i] for i in order]
    px = [peak_x[i] for i in order]
    pint = [peak_intensities[i] for i in order]

    jobs: List[_FitJob] = []
    cur_idx = [pidx[0]]
    cur_x = [px[0]]
    cur_int = [pint[0]]
    cur_start = max(0, pidx[0] - fit_roi_padding)
    cur_end = min(n_bins - 1, pidx[0] + fit_roi_padding)
    for k in range(1, len(pidx)):
        next_start = max(0, pidx[k] - fit_roi_padding)
        next_end = min(n_bins - 1, pidx[k] + fit_roi_padding)
        if next_start <= cur_end:
            cur_end = max(cur_end, next_end)
            cur_idx.append(pidx[k])
            cur_x.append(px[k])
            cur_int.append(pint[k])
        else:
            jobs.append(_FitJob(cur_start, cur_end, cur_idx, cur_x, cur_int))
            cur_start = next_start
            cur_end = next_end
            cur_idx = [pidx[k]]
            cur_x = [px[k]]
            cur_int = [pint[k]]
    jobs.append(_FitJob(cur_start, cur_end, cur_idx, cur_x, cur_int))
    return jobs


def _fit_one_job(
    x: np.ndarray, intensity: np.ndarray, job: _FitJob, x_bin_size: float,
) -> np.ndarray:
    """Fit one ROI containing ``job.numPeaks`` peaks. Returns array of shape
    (n_peaks_in_job, PF_PARAMS_PER_PEAK)."""
    n_p = len(job.peak_indices)
    s, e = job.start, job.end
    x_w = x[s : e + 1].astype(np.float64)
    y_w = intensity[s : e + 1].astype(np.float64)
    n_pts = x_w.size
    if n_pts <= n_p * 4 + 2:
        return np.zeros((n_p, PF_PARAMS_PER_PEAK), dtype=np.float64)

    # Initial params per peak
    p0 = np.zeros(n_p * 4 + 2, dtype=np.float64)
    bg_guess = 0.0
    for k, peak_local_idx in enumerate(job.peak_indices):
        local_idx_in_window = peak_local_idx - s
        local_idx_in_window = max(0, min(n_pts - 1, local_idx_in_window))
        fwhm_bins, bg_local, amp = estimate_initial_params(y_w, local_idx_in_window)
        bg_guess = bg_local
        fwhm_x = fwhm_bins * x_bin_size       # FWHM in user x-units
        # Initial sig = (fwhm_x / 2.355)² * 100² (centideg² assumption)
        sigma = fwhm_x / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        sig_init = (sigma * 100.0) ** 2
        gam_init = fwhm_x * 100.0 * 0.5
        area_init = max(amp, 1e-6) * fwhm_x * 1.0645   # ~Voigt area
        p0[k * 4 + 0] = area_init
        p0[k * 4 + 1] = job.peak_x[k]
        p0[k * 4 + 2] = sig_init
        p0[k * 4 + 3] = gam_init
    p0[n_p * 4 + 0] = bg_guess     # bg0
    p0[n_p * 4 + 1] = 0.0          # bg1

    # Bounds per parameter
    lb = np.empty_like(p0)
    ub = np.empty_like(p0)
    for k in range(n_p):
        lb[k * 4 + 0] = 0.0
        ub[k * 4 + 0] = max(1.0, abs(p0[k * 4 + 0]) * 100.0)
        # Center constrained to ±5 bins
        lb[k * 4 + 1] = job.peak_x[k] - 5.0 * x_bin_size
        ub[k * 4 + 1] = job.peak_x[k] + 5.0 * x_bin_size
        lb[k * 4 + 2] = 1e-12
        ub[k * 4 + 2] = max(1.0, p0[k * 4 + 2] * 100.0)
        lb[k * 4 + 3] = 1e-6
        ub[k * 4 + 3] = max(1.0, p0[k * 4 + 3] * 100.0)
    # Background bounds
    lb[n_p * 4 + 0] = -abs(bg_guess) * 10.0 - 1.0
    ub[n_p * 4 + 0] =  abs(bg_guess) * 10.0 + max(abs(y_w).max(), 1.0)
    lb[n_p * 4 + 1] = -max(abs(y_w).max(), 1.0)
    ub[n_p * 4 + 1] =  max(abs(y_w).max(), 1.0)

    x_lo = float(x_w[0])
    x_hi = float(x_w[-1])

    def residual(p):
        return pseudo_voigt(x_w, n_p, p, x_lo=x_lo, x_hi=x_hi) - y_w

    try:
        result = least_squares(
            residual, p0,
            bounds=(lb, ub),
            method="trf",
            x_scale="jac",
            ftol=1e-9, xtol=1e-9, gtol=1e-9,
            max_nfev=4000,
        )
        p_fit = result.x
        chi_sq = float((result.fun ** 2).sum()) / max(n_pts - n_p * 4 - 2, 1)
    except Exception:
        return np.zeros((n_p, PF_PARAMS_PER_PEAK), dtype=np.float64)

    out = np.zeros((n_p, PF_PARAMS_PER_PEAK), dtype=np.float64)
    for k in range(n_p):
        area = p_fit[k * 4 + 0]
        center = p_fit[k * 4 + 1]
        sig = p_fit[k * 4 + 2]
        gam = p_fit[k * 4 + 3]
        FWHM, eta = tch_eta_fwhm(sig, gam)
        out[k, 0] = area
        out[k, 1] = center
        out[k, 2] = sig
        out[k, 3] = gam
        out[k, 4] = FWHM
        out[k, 5] = eta
        out[k, 6] = chi_sq
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def fit_peaks(
    x: np.ndarray, intensity: np.ndarray,
    peak_locations: Sequence[float],
    *,
    x_bin_size: float,
    fit_roi_padding: int = 20,
) -> np.ndarray:
    """Fit a list of specified peaks. Returns (n_peaks, 7) array.

    Mirror of fitPeaks() in PeakFit.c — the layout of the output rows is
    bit-identical (PF_PARAMS_PER_PEAK = 7).
    """
    x = np.asarray(x, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    n_bins = x.size
    n_peaks = len(peak_locations)
    if n_peaks == 0 or n_bins == 0:
        return np.zeros((0, PF_PARAMS_PER_PEAK), dtype=np.float64)

    # Map specified locations → nearest bin index
    peak_indices: List[int] = []
    valid_x: List[float] = []
    valid_int: List[float] = []
    for loc in peak_locations:
        idx = int(np.argmin(np.abs(x - loc)))
        if abs(x[idx] - loc) <= 2.0 * x_bin_size:
            peak_indices.append(idx)
            valid_x.append(float(x[idx]))
            valid_int.append(float(intensity[idx]))

    if not peak_indices:
        return np.zeros((n_peaks, PF_PARAMS_PER_PEAK), dtype=np.float64)

    jobs = _partition_into_jobs(peak_indices, valid_x, valid_int,
                                n_bins=n_bins, fit_roi_padding=fit_roi_padding)
    out = np.zeros((len(peak_indices), PF_PARAMS_PER_PEAK), dtype=np.float64)
    cursor = 0
    for job in jobs:
        block = _fit_one_job(x, intensity, job, x_bin_size)
        n = block.shape[0]
        out[cursor:cursor + n] = block
        cursor += n
    return out


def fit_peaks_autodetect(
    x: np.ndarray, intensity: np.ndarray, *,
    max_peaks: int, x_bin_size: float,
    fit_roi_padding: int = 20, snip_iter: int = 50,
) -> np.ndarray:
    """Auto-detect + fit. Mirror of fitPeaksAutoDetect()."""
    x = np.asarray(x, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    n_bins = x.size
    if n_bins == 0 or max_peaks <= 0:
        return np.zeros((0, PF_PARAMS_PER_PEAK), dtype=np.float64)

    bg = snip_background(intensity, snip_iter)
    corrected = np.maximum(intensity - bg, 0.0)

    over_select = min(max_peaks * 3, n_bins // 2)
    locs = detect_peaks(x, corrected,
                        max_peaks=over_select,
                        min_separation=x_bin_size * 3.0)
    if locs.size == 0:
        return np.zeros((max_peaks, PF_PARAMS_PER_PEAK), dtype=np.float64)

    raw = fit_peaks(x, corrected, locs.tolist(),
                    x_bin_size=x_bin_size, fit_roi_padding=fit_roi_padding)
    if raw.shape[0] == 0:
        return np.zeros((max_peaks, PF_PARAMS_PER_PEAK), dtype=np.float64)

    # Quality filter — area > 0 & fwhm > 0; cap FWHM at 5× median
    valid_mask = (raw[:, 0] > 0) & (raw[:, 4] > 0)
    valid = raw[valid_mask]
    if valid.shape[0] > 2:
        med_fwhm = float(np.median(valid[:, 4]))
        fwhm_cap = max(5.0 * med_fwhm, 2.0 * x_bin_size)
        valid = valid[valid[:, 4] <= fwhm_cap]

    if valid.shape[0] > max_peaks:
        order = np.argsort(valid[:, 0])[::-1][:max_peaks]
        valid = valid[order]
    valid = valid[np.argsort(valid[:, 1])]

    out = np.zeros((max_peaks, PF_PARAMS_PER_PEAK), dtype=np.float64)
    out[: valid.shape[0]] = valid
    return out
