"""Bucket dispatcher + per-region fit orchestration.

Groups regions by ``(n_peaks, padded_npixels_bucket)`` so each LM call sees a
uniform tensor batch. Pixel padding is masked via ``pixel_mask`` (zeros in
padded slots → zero residual contribution).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from midas_peakfit.adam_fallback import adam_polish
from midas_peakfit.lm import LMConfig, lm_solve
from midas_peakfit.model import integrated_intensity, residuals
from midas_peakfit.postfit import N_PEAK_COLS, build_peak_rows
from midas_peakfit.seeds import SeededRegion
from midas_peakfit.uncertainty import compute_moment_sigma


# Pixel-count buckets. Powers of two are convenient and limit the number
# of distinct shapes (cuts kernel-launch overhead on GPU).
_PIXEL_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 10000]


def _pixel_bucket(n: int) -> int:
    for b in _PIXEL_BUCKETS:
        if n <= b:
            return b
    return _PIXEL_BUCKETS[-1]


N_UNC_COLS = 9  # BG_sigma + 8 per-peak sigmas (Imax, R, Eta, Mu, sgR, slR, sgE, slE)

# Moment-sidecar layout per peak (Modregger 2025; see uncertainty.compute_moment_sigma):
#   col 0: M_0                     — background-subtracted integrated counts
#   col 1: quality_flag (0/1/2)    — Modregger photon-count regime (cast to double)
#   col 2: u_M1_R   [pixel]        — shot-noise σ on radial centroid
#   col 3: u_M1_Eta [degree]       — shot-noise σ on azimuthal centroid
#   col 4: u_M2_R   [pixel²]       — shot-noise σ on radial M_2
#   col 5: u_M2_Eta [degree²]      — shot-noise σ on azimuthal M_2
N_MOMENT_COLS = 6


@dataclass
class FitOutput:
    """Result of fitting one ``SeededRegion``: 29-column rows + raw pixel coords.

    ``rows_unc`` (optional, set only when ``compute_uncertainty`` is on) holds
    a ``(n_peaks, 9)`` per-peak σ table: column 0 is the shared region BG σ
    (duplicated across the region's peaks for downstream parsing convenience);
    columns 1..8 are σ for (Imax, R, Eta, Mu, σGR, σLR, σGEta, σLEta) — same
    parameter order as in the LM ``x`` vector.

    ``rows_moment`` (optional, set only when ``compute_moments`` is on) holds
    a ``(n_peaks, 6)`` per-peak moment-sensitivity table per the layout
    documented at ``N_MOMENT_COLS`` above. Model-free, derived from the
    seed-stage Voronoi-partitioned moments; complementary to the
    Hessian-based ``rows_unc``.
    """

    region_id: int
    rows: np.ndarray  # (n_peaks_this_region, 29) float64
    pixel_y: np.ndarray  # int16 (n_pixels_this_region,)  Y row
    pixel_z: np.ndarray  # int16 (n_pixels_this_region,)  Z col
    rows_unc: np.ndarray | None = None  # (n_peaks, 9) float64 or None
    rows_moment: np.ndarray | None = None  # (n_peaks, 6) float64 or None


def _build_unc_rows(sigma_x_b: np.ndarray, n_peaks: int) -> np.ndarray:
    """Format the per-region [N] sigma vector into (n_peaks, 9).

    Column 0: BG sigma (shared, replicated across peaks).
    Columns 1..8: per-peak (Imax, R, Eta, Mu, σGR, σLR, σGEta, σLEta) — matches
    the ``model.split_params`` slot order.
    """
    out = np.zeros((n_peaks, N_UNC_COLS), dtype=np.float64)
    out[:, 0] = float(sigma_x_b[0])  # BG_sigma replicated
    per_peak = sigma_x_b[1:].reshape(n_peaks, 8)
    out[:, 1:9] = per_peak
    return out


def _build_moment_rows(sr: SeededRegion) -> np.ndarray | None:
    """Build the per-peak moment-sensitivity table for one seeded region.

    Returns ``None`` if higher-moment data was not collected (i.e. the seed
    pass ran with ``compute_moments=False``).
    """
    if sr.peak_M2_R is None:
        return None
    n = sr.n_peaks
    out = np.zeros((n, N_MOMENT_COLS), dtype=np.float64)
    out[:, 0] = sr.peak_M0
    out[:, 1] = sr.peak_quality.astype(np.float64)
    # dx = 1.0: R is in pixel units, η is in degree units; both already
    # parameterise the per-pixel coordinate at sampling spacing 1.
    u_M0_R, u_M1_R, u_M2_R = compute_moment_sigma(
        sr.peak_M0, sr.peak_M2_R, sr.peak_M4_R, dx=1.0
    )
    u_M0_E, u_M1_E, u_M2_E = compute_moment_sigma(
        sr.peak_M0, sr.peak_M2_Eta, sr.peak_M4_Eta, dx=1.0
    )
    out[:, 2] = u_M1_R
    out[:, 3] = u_M1_E
    out[:, 4] = u_M2_R
    out[:, 5] = u_M2_E
    return out


def fit_regions(
    seeded: List[SeededRegion],
    *,
    omega: float,
    Ycen: float,
    Zcen: float,
    do_peak_fit: int,
    local_maxima_only: int,
    device: torch.device,
    dtype: torch.dtype,
    lm_config: LMConfig = LMConfig(),
    use_adam_fallback: bool = True,
    spot_id_start: int = 1,
) -> Tuple[List[FitOutput], int]:
    """Fit a list of seeded regions; emit one ``FitOutput`` per region.

    Returns ``(outputs, next_spot_id_start)``.

    If ``do_peak_fit == 0`` or ``local_maxima_only == 1``, the fit is skipped
    and rows are filled from initial seed values (matches C semantics).
    """
    outputs: List[FitOutput] = []
    if not seeded:
        return outputs, spot_id_start

    if do_peak_fit == 0 or local_maxima_only == 1:
        # No fitting: emit one row per regional max with seed parameters
        for sr in seeded:
            rows = _emit_seed_only_rows(
                sr,
                omega=omega,
                Ycen=Ycen,
                Zcen=Zcen,
                spot_id_start=spot_id_start,
            )
            outputs.append(
                FitOutput(
                    region_id=sr.region_id,
                    rows=rows,
                    pixel_y=sr.pixels_y.astype(np.int16),
                    pixel_z=sr.pixels_z.astype(np.int16),
                    rows_moment=_build_moment_rows(sr),
                )
            )
            spot_id_start += sr.n_peaks
        return outputs, spot_id_start

    # ── Bucket by (n_peaks, padded_n_pixels) ────────────────────────────
    buckets: Dict[Tuple[int, int], List[SeededRegion]] = defaultdict(list)
    for sr in seeded:
        key = (sr.n_peaks, _pixel_bucket(sr.n_pixels))
        buckets[key].append(sr)

    # ── Fit each bucket via batched LM ──────────────────────────────────
    fitted_per_region: Dict[int, np.ndarray] = {}  # region_id → x_fit (N_total,)
    cost_per_region: Dict[int, float] = {}
    rc_per_region: Dict[int, int] = {}
    sigma_per_region: Dict[int, np.ndarray] = {}  # region_id → sigma_x (N_total,)

    for (n_peaks, M_pad), srs in buckets.items():
        B = len(srs)
        N_params = 1 + 8 * n_peaks
        x_init = torch.zeros((B, N_params), dtype=dtype, device=device)
        lo = torch.zeros((B, N_params), dtype=dtype, device=device)
        hi = torch.zeros((B, N_params), dtype=dtype, device=device)
        z = torch.zeros((B, M_pad), dtype=dtype, device=device)
        Rs = torch.zeros((B, M_pad), dtype=dtype, device=device)
        Etas = torch.zeros((B, M_pad), dtype=dtype, device=device)
        pmask = torch.zeros((B, M_pad), dtype=dtype, device=device)

        for b, sr in enumerate(srs):
            x_init[b] = torch.from_numpy(sr.x0).to(dtype=dtype, device=device)
            lo[b] = torch.from_numpy(sr.xl).to(dtype=dtype, device=device)
            hi[b] = torch.from_numpy(sr.xu).to(dtype=dtype, device=device)
            n = sr.n_pixels
            z[b, :n] = torch.from_numpy(sr.z_values).to(dtype=dtype, device=device)
            Rs[b, :n] = torch.from_numpy(sr.Rs).to(dtype=dtype, device=device)
            Etas[b, :n] = torch.from_numpy(sr.Etas).to(dtype=dtype, device=device)
            pmask[b, :n] = 1.0

        x_fit, c_fit, rc, sigma_x = lm_solve(
            x_init, lo, hi, z, Rs, Etas, pmask, n_peaks, config=lm_config
        )
        sigma_x_np_per_bucket = (
            sigma_x.detach().cpu().numpy() if sigma_x.numel() > 0 else None
        )

        # Adam fallback for non-zero return codes
        if use_adam_fallback:
            non_conv = (rc != 0)
            if non_conv.any():
                idx = non_conv.nonzero(as_tuple=False).squeeze(-1)
                x_a, c_a = adam_polish(
                    x_fit[idx], lo[idx], hi[idx], z[idx], Rs[idx],
                    Etas[idx], pmask[idx], n_peaks,
                )
                # Adam result replaces LM result; mark with rc=-1 (per design)
                x_fit[idx] = x_a
                c_fit[idx] = c_a
                rc[idx] = -1

        x_fit_np = x_fit.detach().cpu().numpy()
        c_fit_np = c_fit.detach().cpu().numpy()
        rc_np = rc.detach().cpu().numpy()
        for b, sr in enumerate(srs):
            fitted_per_region[sr.region_id] = x_fit_np[b]
            cost_per_region[sr.region_id] = float(c_fit_np[b])
            rc_per_region[sr.region_id] = int(rc_np[b])
            if sigma_x_np_per_bucket is not None:
                sigma_per_region[sr.region_id] = sigma_x_np_per_bucket[b]

    # ── Compute integrated intensity per region (post-fit) ─────────────
    # Re-batch by bucket (cheaper than per-region) — same buckets as above.
    integrated_per_region: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for (n_peaks, M_pad), srs in buckets.items():
        B = len(srs)
        x = torch.zeros((B, 1 + 8 * n_peaks), dtype=dtype, device=device)
        Rs = torch.zeros((B, M_pad), dtype=dtype, device=device)
        Etas = torch.zeros((B, M_pad), dtype=dtype, device=device)
        pmask = torch.zeros((B, M_pad), dtype=dtype, device=device)

        for b, sr in enumerate(srs):
            x[b] = torch.from_numpy(fitted_per_region[sr.region_id]).to(
                dtype=dtype, device=device
            )
            n = sr.n_pixels
            Rs[b, :n] = torch.from_numpy(sr.Rs).to(dtype=dtype, device=device)
            Etas[b, :n] = torch.from_numpy(sr.Etas).to(dtype=dtype, device=device)
            pmask[b, :n] = 1.0

        ii, np_ = integrated_intensity(x, Rs, Etas, pmask, n_peaks)
        ii_np = ii.detach().cpu().numpy()
        np_np = np_.detach().cpu().numpy()
        for b, sr in enumerate(srs):
            integrated_per_region[sr.region_id] = (
                ii_np[b][: sr.n_peaks],
                np_np[b][: sr.n_peaks],
            )

    # ── Build the per-region rows ──────────────────────────────────────
    for sr in seeded:
        x_fit = fitted_per_region[sr.region_id]
        cost_v = cost_per_region[sr.region_id]
        rc_v = rc_per_region[sr.region_id]
        ii, n_pix_per_peak = integrated_per_region[sr.region_id]

        # Slice fitted parameters
        bg_f = float(x_fit[0])
        per_peak = x_fit[1:].reshape(sr.n_peaks, 8)
        Imax_f = per_peak[:, 0]
        R_f = per_peak[:, 1]
        Eta_f = per_peak[:, 2]
        Mu_f = per_peak[:, 3]
        sgR_f = per_peak[:, 4]
        slR_f = per_peak[:, 5]
        sgE_f = per_peak[:, 6]
        slE_f = per_peak[:, 7]

        rmse = float(np.sqrt(max(cost_v, 0.0)))

        rows = build_peak_rows(
            spot_id_start=spot_id_start,
            omega=omega,
            Ycen=Ycen,
            Zcen=Zcen,
            n_peaks_in_region=sr.n_peaks,
            n_pixels_in_region=sr.n_pixels,
            raw_sum_intensity=sr.raw_sum,
            mask_touched=sr.mask_touched,
            return_code=rc_v,
            fit_rmse=rmse,
            bg=bg_f,
            Imax=Imax_f,
            R=R_f,
            Eta=Eta_f,
            sigmaGR=sgR_f,
            sigmaLR=slR_f,
            sigmaGEta=sgE_f,
            sigmaLEta=slE_f,
            Mu=Mu_f,
            integrated_intensity=ii,
            n_pixels_per_peak=n_pix_per_peak,
            maxY=sr.maxY,
            maxZ=sr.maxZ,
            raw_imax=sr.maxima_values,
        )
        rows_unc = None
        if sr.region_id in sigma_per_region:
            rows_unc = _build_unc_rows(
                sigma_per_region[sr.region_id], sr.n_peaks
            )
        outputs.append(
            FitOutput(
                region_id=sr.region_id,
                rows=rows,
                pixel_y=sr.pixels_y.astype(np.int16),
                pixel_z=sr.pixels_z.astype(np.int16),
                rows_unc=rows_unc,
                rows_moment=_build_moment_rows(sr),
            )
        )
        spot_id_start += sr.n_peaks

    return outputs, spot_id_start


def _emit_seed_only_rows(
    sr: SeededRegion,
    *,
    omega: float,
    Ycen: float,
    Zcen: float,
    spot_id_start: int,
) -> np.ndarray:
    """For ``doPeakFit=0`` or ``localMaximaOnly=1``: row per regional max with
    seed-only values (no fitted widths)."""
    n_peaks = sr.n_peaks
    # Seed values from x0
    bg = float(sr.x0[0])
    per_peak = sr.x0[1:].reshape(n_peaks, 8)
    return build_peak_rows(
        spot_id_start=spot_id_start,
        omega=omega,
        Ycen=Ycen,
        Zcen=Zcen,
        n_peaks_in_region=n_peaks,
        n_pixels_in_region=sr.n_pixels,
        raw_sum_intensity=sr.raw_sum,
        mask_touched=sr.mask_touched,
        return_code=0,
        fit_rmse=0.0,
        bg=bg,
        Imax=per_peak[:, 0],
        R=per_peak[:, 1],
        Eta=per_peak[:, 2],
        sigmaGR=per_peak[:, 4],
        sigmaLR=per_peak[:, 5],
        sigmaGEta=per_peak[:, 6],
        sigmaLEta=per_peak[:, 7],
        Mu=per_peak[:, 3],
        integrated_intensity=sr.maxima_values,  # placeholder = peak intensity
        n_pixels_per_peak=np.ones(n_peaks, dtype=np.int32),
        maxY=sr.maxY,
        maxZ=sr.maxZ,
        raw_imax=sr.maxima_values,
    )


__all__ = ["FitOutput", "fit_regions", "N_UNC_COLS", "N_MOMENT_COLS"]
