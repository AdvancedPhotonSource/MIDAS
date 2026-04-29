"""calc_radius: drop-in replacement for ``CalcRadiusAllZarr``.

Reads the 17-column ``Result_*.csv`` produced by the merge stage and writes
the 24-column ``Radius_*.csv``. The ring filter, Bragg angle, grain volume,
and powder intensity calculations are direct ports of
``CalcRadiusAllZarr.c:347-434``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from ..io import csv as csv_io
from ..params import ZarrParams

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


@dataclass
class RadiusResult:
    """In-memory result of the calc_radius stage."""

    spots: torch.Tensor                # (N, 24) float64
    ring_radii: torch.Tensor           # (n_configured_rings,) float64
    ring_numbers: List[int] = field(default_factory=list)
    n_frames: int = 0


# ---------------------------------------------------------------------------
# Pure-tensor kernel
# ---------------------------------------------------------------------------


def _filter_and_compute_radius(
    result_arr: torch.Tensor,             # (N_in, 17) — 17-col Result_*.csv
    ring_numbers: torch.Tensor,           # (R,) int64
    ring_radii_um: torch.Tensor,          # (R,) float64 — RADII IN µm (read straight from hkls.csv col 10)
    width_px: float,
    px_um: float,
    Lsd_um: float,
    OmegaStep: float,
    Hbeam: float,
    Rsample: float,
    Vsample: float,
    DiscModel: int,
    DiscArea: float,
    n_frames: int,
    top_layer: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised port of ``CalcRadiusAllZarr.c:347-434``.

    Returns:
        out  : (N_out, 24) float64 — the 24-column Radius CSV layout.
        powder_int : (R,) — per-ring averaged powder intensity (after /n_frames).
        m_hkl : (R,) — count of hkl entries hitting each configured ring.

    The ring filter uses ``first match wins`` (deterministic by ring-table order),
    matching the C code's iteration order. Spots whose radius matches no ring
    within ``Width`` are dropped.
    """
    device = result_arr.device
    dtype = result_arr.dtype
    N_in = result_arr.shape[0]
    R = ring_numbers.shape[0]

    # Result_*.csv columns (per merge output):
    # 0=SpotID, 1=IntegratedIntensity, 2=Omega, 3=YCen, 4=ZCen, 5=IMax,
    # 6=MinOme, 7=MaxOme, 8=SigmaR, 9=SigmaEta, 10=NrPx, 11=NrPxTot,
    # 12=Radius, 13=Eta, 14=RawSumIntensity, 15=maskTouched, 16=FitRMSE
    R_obs_um = result_arr[:, 12] * px_um  # observed radius in µm

    # Ring assignment via first-match argmin within the |diff|<width window.
    # ``ring_radii_um`` and ``R_obs_um`` are both in µm.
    rad_um = ring_radii_um.to(device=device, dtype=dtype)
    diff = (R_obs_um.unsqueeze(1) - rad_um.unsqueeze(0)).abs()  # (N, R)
    in_window = diff < width_px
    has_match = in_window.any(dim=1)
    # First-match index: argmax on bool returns first True position.
    first_match = torch.argmax(in_window.to(torch.int8), dim=1)
    sel = has_match
    spot_match = first_match[sel]
    spots_in = result_arr[sel]
    N_out = spots_in.shape[0]

    if N_out == 0:
        empty24 = torch.empty((0, 24), dtype=dtype, device=device)
        return empty24, torch.zeros(R, dtype=dtype, device=device), torch.zeros(R, dtype=torch.int64, device=device)

    Eta = spots_in[:, 13]
    Omega = spots_in[:, 2]
    MinOme = spots_in[:, 6]
    MaxOme = spots_in[:, 7]
    Radius = spots_in[:, 12]

    # TopLayer filter: drop near-equator spots when active.
    if top_layer:
        mask = torch.abs(Eta) >= 90.0
        sel_idx = torch.nonzero(sel, as_tuple=False).squeeze(1)[mask]
        spots_in = spots_in[mask]
        spot_match = spot_match[mask]
        Eta = Eta[mask]
        Omega = Omega[mask]
        MinOme = MinOme[mask]
        MaxOme = MaxOme[mask]
        Radius = Radius[mask]
        N_out = spots_in.shape[0]
        if N_out == 0:
            empty24 = torch.empty((0, 24), dtype=dtype, device=device)
            return (
                empty24,
                torch.zeros(R, dtype=dtype, device=device),
                torch.zeros(R, dtype=torch.int64, device=device),
            )

    # Bragg: theta = 0.5 * atan(Radius_px * px / Lsd) [degrees]
    Theta = 0.5 * _RAD2DEG * torch.atan(Radius * px_um / Lsd_um)
    DeltaOmega = abs(OmegaStep) + MaxOme - MinOme
    NImgs = DeltaOmega / abs(OmegaStep)
    RingNr = ring_numbers.to(device=device)[spot_match].to(dtype)

    # Per-ring powder intensity: sum of IntegratedIntensity / n_frames.
    powder_int = torch.zeros(R, dtype=dtype, device=device)
    powder_int.scatter_add_(0, spot_match, spots_in[:, 1])
    powder_int = powder_int / max(n_frames, 1)

    # m_hkl is read from hkls.csv; we count how many hkl entries fall on each
    # configured ring. The caller supplies this since it depends on hkls.csv,
    # not on the spot table. We approximate here as 1 per ring (same as the
    # C code when hkls.csv is missing); the public entry point overrides.
    m_hkl = torch.ones(R, dtype=torch.int64, device=device)

    # Grain volume.
    if DiscModel == 1:
        Vgauge = DiscArea
    else:
        Vgauge = Hbeam * math.pi * Rsample * Rsample
        if Vsample != 0:
            Vgauge = Vsample

    # ΔΘ formula: deltaTheta = deg2rad * (asin(sin(Theta)·cos(DeltaOmega) + cos(Theta)·|sin(Eta)|·sin(DeltaOmega)) - Theta)
    sin_th = torch.sin(Theta * _DEG2RAD)
    cos_th = torch.cos(Theta * _DEG2RAD)
    sin_dom = torch.sin(DeltaOmega * _DEG2RAD)
    cos_dom = torch.cos(DeltaOmega * _DEG2RAD)
    sin_eta_abs = torch.abs(torch.sin(Eta * _DEG2RAD))
    arg = (sin_th * cos_dom + cos_th * sin_eta_abs * sin_dom).clamp(min=-1.0, max=1.0)
    deltaTheta = _DEG2RAD * (_RAD2DEG * torch.asin(arg) - Theta)

    GrainVolume = (
        0.5
        * m_hkl[spot_match].to(dtype)
        * deltaTheta
        * cos_th
        * Vgauge
        * spots_in[:, 1]
        / (NImgs * powder_int[spot_match])
    )
    if DiscModel == 1:
        GrainRadius = torch.sqrt(GrainVolume / math.pi)
    else:
        GrainRadius = torch.sign(GrainVolume) * torch.abs(GrainVolume).pow(1.0 / 3.0) * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

    # 24-col layout: SpotID(=counter+1), IntInt, Omega, YCen, ZCen, IMax,
    # MinOme, MaxOme, Radius, Theta, Eta, DeltaOmega, NImgs, RingNr,
    # GrainVolume, GrainRadius, PowderIntensity, SigmaR, SigmaEta,
    # NrPx, NrPxTot, RawSumIntensity, maskTouched, FitRMSE
    spot_ids_renumbered = torch.arange(1, N_out + 1, device=device, dtype=dtype)
    out = torch.stack([
        spot_ids_renumbered,
        spots_in[:, 1],   # IntegratedIntensity
        spots_in[:, 2],   # Omega
        spots_in[:, 3],   # YCen
        spots_in[:, 4],   # ZCen
        spots_in[:, 5],   # IMax
        spots_in[:, 6],   # MinOme
        spots_in[:, 7],   # MaxOme
        Radius,
        Theta,
        Eta,
        DeltaOmega,
        NImgs,
        RingNr,
        GrainVolume,
        GrainRadius,
        powder_int[spot_match],
        spots_in[:, 8],   # SigmaR
        spots_in[:, 9],   # SigmaEta
        spots_in[:, 10],  # NrPx
        spots_in[:, 11],  # NrPxTot
        spots_in[:, 14],  # RawSumIntensity
        spots_in[:, 15],  # maskTouched
        spots_in[:, 16],  # FitRMSE
    ], dim=1)
    return out, powder_int, m_hkl


def _read_hkls_ring_radii(
    hkls_path: Path, ring_numbers: List[int]
) -> Tuple[List[float], List[int]]:
    """Read hkls.csv and return (RingRadii, m_hkl) for the requested ring numbers.

    Per ``CalcRadiusAllZarr.c:303-318``, hkls.csv has these whitespace-separated
    columns: ``h k l ds RN ?? ?? ?? ?? ?? RingRad`` (RingRad at column index 10).
    """
    radii = [0.0] * len(ring_numbers)
    counts = [0] * len(ring_numbers)
    if not hkls_path.exists():
        return radii, counts
    with open(hkls_path, "r") as f:
        f.readline()  # skip header
        for line in f:
            tokens = line.split()
            if len(tokens) < 11:
                continue
            try:
                rn = int(tokens[4])
                rrad = float(tokens[10])
            except (ValueError, IndexError):
                continue
            for i, target in enumerate(ring_numbers):
                if rn == target:
                    radii[i] = rrad
                    counts[i] += 1
                    break
    return radii, counts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def calc_radius(
    result_folder: Union[str, Path] = ".",
    *,
    zarr_params: Optional[ZarrParams] = None,
    result_csv: Optional[Union[str, Path]] = None,
    hkls_path: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    result_array: Optional[np.ndarray] = None,
    start_nr: Optional[int] = None,
    end_nr: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    write: bool = True,
    Vsample: float = 0.0,
    DiscModel: int = 0,
    DiscArea: float = 0.0,
    top_layer: bool = False,
) -> RadiusResult:
    """Compute per-spot ring/Bragg/grain-volume properties; replaces
    ``CalcRadiusAllZarr``."""

    rf = Path(result_folder)
    out_dir = Path(out_dir) if out_dir is not None else rf

    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)

    if zarr_params is None:
        raise ValueError(
            "calc_radius requires zarr_params (parsed from the Zarr archive). "
            "Use midas_transforms.params.read_zarr_params() to obtain one."
        )

    # Default StartNr/EndNr from the Zarr archive (1-based).
    if start_nr is None:
        start_nr = 1
    if end_nr is None:
        end_nr = zarr_params.EndNr if zarr_params.EndNr > 0 else 1
    n_frames = max(end_nr - start_nr + 1, 1)

    # Inputs.
    if result_array is None:
        if result_csv is None:
            result_csv = rf / f"Result_StartNr_{start_nr}_EndNr_{end_nr}.csv"
        result_array = csv_io.read_result_csv(result_csv)

    if hkls_path is None:
        hkls_path = rf / "hkls.csv"

    ring_numbers = [int(rn) for (rn, _) in zarr_params.RingThresh]
    if not ring_numbers:
        raise ValueError("ZarrParams.RingThresh is empty — no rings configured.")

    radii_um, mhkl = _read_hkls_ring_radii(Path(hkls_path), ring_numbers)
    rn_t = torch.tensor(ring_numbers, dtype=torch.int64, device=dev)
    rr_t = torch.tensor(radii_um, dtype=dt, device=dev)

    spots_t = torch.from_numpy(np.asarray(result_array, dtype=np.float64)).to(
        device=dev, dtype=dt
    )

    out, powder_int, _mhkl_default = _filter_and_compute_radius(
        spots_t,
        ring_numbers=rn_t,
        ring_radii_um=rr_t,
        width_px=zarr_params.Width if zarr_params.Width > 0 else (zarr_params.WidthOrig or 1.0),
        px_um=zarr_params.PixelSize,
        Lsd_um=zarr_params.Lsd,
        OmegaStep=zarr_params.OmegaStep if zarr_params.OmegaStep else 1.0,
        Hbeam=zarr_params.Hbeam,
        Rsample=zarr_params.Rsample,
        Vsample=Vsample,
        DiscModel=DiscModel,
        DiscArea=DiscArea,
        n_frames=n_frames,
        top_layer=top_layer,
    )

    # Re-apply m_hkl from hkls.csv.
    if any(mhkl):
        mhkl_t = torch.tensor(mhkl, dtype=dt, device=dev)
        # Re-scale the GrainVolume / GrainRadius columns by mhkl[ring]:
        # the kernel uses 1; multiplier is `mhkl[ring] / 1`.
        # Recompute by scaling col 14 (GrainVolume) and col 15 (GrainRadius).
        # Identify ring index by ring number.
        ring_nrs_out = out[:, 13].long()
        ring_idx_out = torch.zeros_like(ring_nrs_out)
        for i, rn in enumerate(ring_numbers):
            ring_idx_out[ring_nrs_out == rn] = i
        scale = mhkl_t[ring_idx_out]
        out[:, 14] = out[:, 14] * scale  # GrainVolume scales linearly
        if DiscModel == 1:
            out[:, 15] = torch.sqrt(out[:, 14] / math.pi)
        else:
            out[:, 15] = (
                torch.sign(out[:, 14])
                * torch.abs(out[:, 14]).pow(1.0 / 3.0)
                * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
            )

    if write:
        out_path = out_dir / f"Radius_StartNr_{start_nr}_EndNr_{end_nr}.csv"
        csv_io.write_radius_csv(out_path, out.detach().cpu().numpy().astype(np.float64))

    return RadiusResult(
        spots=out,
        ring_radii=rr_t,
        ring_numbers=list(ring_numbers),
        n_frames=n_frames,
    )
