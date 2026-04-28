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
from midas_integrate.geometry import build_tilt_matrix, invert_REta_to_pixel
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
    """Per (ring × η-bin): centroid in the radial window → (R_fit, η) → (Y_pix, Z_pix)."""
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    half_px = 0.5 * params.Width / px
    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)

    fits: List[FittedPoint] = []
    for ring_i, r_ideal in enumerate(rt.r_ideal_px):
        idx = np.where(np.abs(cake.R_centers - r_ideal) <= half_px)[0]
        if idx.size < 3:
            continue
        R_window = cake.R_centers[idx]
        for eta_j, eta in enumerate(cake.eta_centers):
            I = cake.intensity[idx, eta_j]
            I = np.maximum(I - I.min(), 0.0)
            tot = I.sum()
            if tot <= 0.0:
                continue
            R_fit = float((I * R_window).sum() / tot)
            peak = float(I.max())
            mean = float(I.mean()) + 1e-12
            snr = peak / mean
            if snr < snr_min:
                continue
            try:
                Y_pix, Z_pix = invert_REta_to_pixel(
                    R_fit, eta,
                    Ycen=params.BC_y, Zcen=params.BC_z, TRs=TRs,
                    Lsd=params.Lsd, RhoD=(params.RhoD if params.RhoD > 0 else params.MaxRingRad),
                    px=px, parallax=params.Parallax,
                )
            except Exception:
                continue
            fits.append(FittedPoint(
                Y_pix=float(Y_pix), Z_pix=float(Z_pix),
                ring_idx=ring_i, snr=snr,
            ))
    return fits


def run_estep(
    params: CalibrationParams,
    image: np.ndarray,
    rt: RingTable,
    *, dark: Optional[np.ndarray] = None,
) -> Tuple[CakeProfile, List[FittedPoint]]:
    cake = integrate_cake(params, image, rt, dark=dark)
    fits = extract_fitted_points(cake, rt, params, snr_min=params.SNRMin)
    return cake, fits
