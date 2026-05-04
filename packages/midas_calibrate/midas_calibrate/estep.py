"""E-step: integrate calibrant image, extract per-(ring, η-bin) peak positions.

Strategy:
  1. Build a uniform-R, uniform-η bin grid spanning the calibrant ring range
     (subsetting per-ring windows after integration).
  2. Build midas_integrate's PixelMap + CSR from the current geometry.
  3. Integrate the image into a 2D (R, η) cake.
  4. For each (ring, η-bin), compute a weighted centroid in the radial window
     to get R_fit (px).  v0.1 uses centroid; future versions can swap in a
     pseudo-Voigt LM via midas_peakfit.lm_solve_generic.
  5. Convert (R_fit, η_bin_center) → (Y_pix, Z_pix) via midas_integrate's
     Newton-Raphson inverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from midas_integrate.detector_mapper import build_map
from midas_integrate.geometry import (
    build_tilt_matrix,
    invert_REta_to_pixel,
    invert_REta_to_pixel_batch,
)
from midas_integrate.kernels import build_csr, integrate
from midas_integrate.params import IntegrationParams

from .params import CalibrationParams
from .refine import FittedPoint
from .rings import RingTable


def _calibration_to_integration_params(
    params: CalibrationParams, *, R_min: float, R_max: float, R_bin_size: float, eta_bin_size: float,
) -> IntegrationParams:
    ip = IntegrationParams()
    ip.NrPixelsY = params.NrPixelsY
    ip.NrPixelsZ = params.NrPixelsZ
    ip.pxY = params.pxY
    ip.pxZ = params.pxZ if params.pxZ > 0 else params.pxY
    ip.Lsd = params.Lsd
    ip.BC_y = params.BC_y
    ip.BC_z = params.BC_z
    ip.tx = params.tx
    ip.ty = params.ty
    ip.tz = params.tz
    for i in range(15):
        setattr(ip, f"p{i}", getattr(params, f"p{i}"))
    ip.RhoD = params.RhoD if params.RhoD > 0 else params.MaxRingRad
    ip.Parallax = params.Parallax
    ip.Wavelength = params.Wavelength
    ip.RMin = float(R_min)
    ip.RMax = float(R_max)
    ip.RBinSize = float(R_bin_size)
    ip.EtaMin = -180.0
    ip.EtaMax = 180.0
    ip.EtaBinSize = float(eta_bin_size)
    ip.SolidAngleCorrection = 0
    ip.PolarizationCorrection = 0
    return ip


@dataclass
class CakeProfile:
    R_centers: np.ndarray
    eta_centers: np.ndarray
    intensity: np.ndarray  # [n_R, n_eta]


def integrate_cake(
    params: CalibrationParams,
    image: np.ndarray,
    rt: RingTable,
    *, dark: Optional[np.ndarray] = None,
) -> CakeProfile:
    """Build CSR + integrate the image into a uniform (R, η) cake."""
    if dark is not None:
        image = image - dark

    # R range: half-Width margin around min/max ring radius.
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    half_px = 0.5 * params.Width / px
    R_min = max(0.0, float(rt.r_ideal_px.min()) - half_px - 1.0)
    R_max = float(rt.r_ideal_px.max()) + half_px + 1.0
    ip = _calibration_to_integration_params(
        params, R_min=R_min, R_max=R_max,
        R_bin_size=params.RBinSize, eta_bin_size=params.EtaBinSize,
    )

    pmap_result = build_map(ip, verbose=False)
    from midas_integrate.bin_io import PixelMap as _PixelMap
    pmap = _PixelMap(
        pxList=pmap_result.pxList,
        counts=pmap_result.counts,
        offsets=pmap_result.offsets,
        map_header=None, nmap_header=None,
    )
    geom = build_csr(
        pmap,
        n_r=ip.n_r_bins, n_eta=ip.n_eta_bins,
        n_pixels_y=ip.NrPixelsY, n_pixels_z=ip.NrPixelsZ,
        bc_y=ip.BC_y, bc_z=ip.BC_z,
        device="cpu", dtype=torch.float64,
        build_modes=("bilinear",),
    )

    img_t = torch.as_tensor(image, dtype=torch.float64).contiguous()
    cake = integrate(img_t, geom, mode="bilinear", normalize=True).numpy()

    R_edges = np.linspace(ip.RMin, ip.RMin + ip.RBinSize * ip.n_r_bins, ip.n_r_bins + 1)
    eta_edges = np.linspace(ip.EtaMin, ip.EtaMax, ip.n_eta_bins + 1)
    return CakeProfile(
        R_centers=0.5 * (R_edges[:-1] + R_edges[1:]),
        eta_centers=0.5 * (eta_edges[:-1] + eta_edges[1:]),
        intensity=cake,
    )


def extract_fitted_points(
    cake: CakeProfile, rt: RingTable, params: CalibrationParams,
    *, snr_min: float = 1.0,
) -> List[FittedPoint]:
    """Per (ring × η-bin): centroid in the radial window → (R_fit, η) → (Y_pix, Z_pix).

    Vectorised: the η-axis centroid + SNR filter run as numpy array ops
    per ring, and the Newton inversion runs as a single batched call
    over all surviving (ring, η) pairs. This replaces ~1000 scalar calls
    into the inverter (3-5 rings × 360 η-bins) with one array call.
    """
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    half_px = 0.5 * params.Width / px
    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)
    eta_centers = np.asarray(cake.eta_centers, dtype=np.float64)

    R_chunks: List[np.ndarray] = []
    Eta_chunks: List[np.ndarray] = []
    ring_idx_chunks: List[np.ndarray] = []
    snr_chunks: List[np.ndarray] = []

    for ring_i, r_ideal in enumerate(rt.r_ideal_px):
        idx = np.where(np.abs(cake.R_centers - r_ideal) <= half_px)[0]
        if idx.size < 3:
            continue
        R_window = cake.R_centers[idx].astype(np.float64)        # (n_R,)
        I_block = cake.intensity[idx, :].astype(np.float64)      # (n_R, n_eta)
        # Per-η baseline subtract across the radial window (matches the
        # ``I - I.min()`` per-η-bin step of the previous scalar loop).
        I = np.maximum(I_block - I_block.min(axis=0, keepdims=True), 0.0)
        tot = I.sum(axis=0)                                      # (n_eta,)
        valid_tot = tot > 0.0
        if not valid_tot.any():
            continue
        # Centroid R_fit per η; safe-divide on bins with zero total.
        safe_tot = np.where(valid_tot, tot, 1.0)
        R_fit = (I * R_window[:, None]).sum(axis=0) / safe_tot
        peak = I.max(axis=0)
        mean = I.mean(axis=0) + 1e-12
        snr = peak / mean
        keep = valid_tot & (snr >= snr_min)
        if not keep.any():
            continue
        R_chunks.append(R_fit[keep])
        Eta_chunks.append(eta_centers[keep])
        ring_idx_chunks.append(np.full(int(keep.sum()), ring_i, dtype=np.int64))
        snr_chunks.append(snr[keep])

    if not R_chunks:
        return []

    R_targets = np.concatenate(R_chunks)
    Eta_targets = np.concatenate(Eta_chunks)
    ring_idxs = np.concatenate(ring_idx_chunks)
    snrs = np.concatenate(snr_chunks)

    Y_pix, Z_pix = invert_REta_to_pixel_batch(
        R_targets, Eta_targets,
        Ycen=params.BC_y, Zcen=params.BC_z, TRs=TRs,
        Lsd=params.Lsd, RhoD=(params.RhoD if params.RhoD > 0 else params.MaxRingRad),
        px=px, parallax=params.Parallax,
    )

    return [
        FittedPoint(
            Y_pix=float(Y_pix[i]), Z_pix=float(Z_pix[i]),
            ring_idx=int(ring_idxs[i]), snr=float(snrs[i]),
        )
        for i in range(R_targets.shape[0])
    ]


def run_estep(
    params: CalibrationParams,
    image: np.ndarray,
    rt: RingTable,
    *, dark: Optional[np.ndarray] = None,
) -> Tuple[CakeProfile, List[FittedPoint]]:
    cake = integrate_cake(params, image, rt, dark=dark)
    fits = extract_fitted_points(cake, rt, params, snr_min=params.SNRMin)
    return cake, fits
