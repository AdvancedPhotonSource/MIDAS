"""HDF5 output of peak-fitting results — port of FF_HEDM/src/PeakFitIO.c.

Produces ``caked_peaks.h5`` files with the exact same group/dataset layout
the C version writes, so existing visualization tooling
(``plot_caked_peaks.py``) consumes either output transparently.

Schema:
    /metadata/zarr_source       (attribute, str)
    /metadata/tth_axis          (float64, n_r_bins)
    /metadata/eta_axis          (float64, n_eta_bins)
    /metadata/frame_keys        (vlen str, n_frames)
    /peaks/frame_idx            (int32,   n_rows)
    /peaks/eta_idx              (int32,   n_rows)
    /peaks/peak_nr              (int32,   n_rows)
    /peaks/eta_deg              (float64, n_rows)
    /peaks/center_2theta        (float64, n_rows)
    /peaks/area                 (float64, n_rows)
    /peaks/sig                  (float64, n_rows)
    /peaks/gam                  (float64, n_rows)
    /peaks/FWHM_deg             (float64, n_rows)
    /peaks/eta_mix              (float64, n_rows)
    /peaks/d_spacing_A          (float64, n_rows)
    /peaks/chi_sq               (float64, n_rows)
    /peaks/frame_key            (vlen str, n_rows)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np

from midas_integrate.peakfit import PF_PARAMS_PER_PEAK


@dataclass
class PeakRow:
    frame_idx: int
    eta_idx: int
    peak_nr: int
    eta_deg: float
    center_2theta: float
    area: float
    sig: float
    gam: float
    FWHM_deg: float
    eta_mix: float
    d_spacing_A: float
    chi_sq: float


def make_row(
    *,
    frame_idx: int, eta_idx: int, peak_nr: int, eta_deg: float,
    gsas_params: Sequence[float],
    px_um: float, Lsd_um: float, wavelength_A: float,
) -> PeakRow:
    """Convert one row of GSAS-II fit output (PF_PARAMS_PER_PEAK doubles) to a
    PeakRow with derived (2θ, d-spacing) — mirrors pfio_make_row.

    ``gsas_params[1]`` is the fitted center; the C code interprets it as
    a *radius in pixels* and converts to 2θ via ``atan(R*px / Lsd)``.
    """
    if len(gsas_params) < PF_PARAMS_PER_PEAK:
        raise ValueError("gsas_params must have at least 7 doubles")
    area = float(gsas_params[0])
    center_px = float(gsas_params[1])
    sig = float(gsas_params[2])
    gam = float(gsas_params[3])
    fwhm = float(gsas_params[4])
    eta_mix = float(gsas_params[5])
    chi = float(gsas_params[6])

    if Lsd_um > 0:
        two_theta_rad = math.atan(center_px * px_um / Lsd_um)
    else:
        two_theta_rad = 0.0
    two_theta_deg = math.degrees(two_theta_rad)
    if wavelength_A > 0 and two_theta_rad > 0:
        d = wavelength_A / (2.0 * math.sin(two_theta_rad / 2.0))
    else:
        d = 0.0

    return PeakRow(
        frame_idx=int(frame_idx),
        eta_idx=int(eta_idx),
        peak_nr=int(peak_nr),
        eta_deg=float(eta_deg),
        center_2theta=two_theta_deg,
        area=area, sig=sig, gam=gam,
        FWHM_deg=fwhm, eta_mix=eta_mix,
        d_spacing_A=d, chi_sq=chi,
    )


@dataclass
class PeakBuffer:
    """Append-only buffer for collected peak rows. Mirrors PeakH5Buffer."""
    rows: List[PeakRow] = field(default_factory=list)

    def append(self, row: PeakRow) -> None:
        self.rows.append(row)

    def extend(self, rows: Sequence[PeakRow]) -> None:
        self.rows.extend(rows)

    def __len__(self) -> int:
        return len(self.rows)


def write_peaks_h5(
    filename: str | Path,
    buf: PeakBuffer,
    *,
    tth_axis: np.ndarray, eta_axis: np.ndarray,
    n_frames: int, source_name: str = "midas_integrate",
) -> None:
    """Write ``buf`` to ``filename`` in the exact PeakFitIO.c schema."""
    n = len(buf)
    with h5py.File(str(filename), "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["zarr_source"] = np.bytes_(source_name)
        meta.create_dataset("tth_axis", data=np.asarray(tth_axis, dtype=np.float64))
        meta.create_dataset("eta_axis", data=np.asarray(eta_axis, dtype=np.float64))
        # vlen strings, matching the C side
        str_dt = h5py.string_dtype(encoding="ascii")
        meta.create_dataset("frame_keys",
                            data=np.array([str(i) for i in range(n_frames)], dtype=object),
                            dtype=str_dt)
        if n == 0:
            return
        peaks = f.create_group("peaks")
        cols = {
            "frame_idx":     np.array([r.frame_idx     for r in buf.rows], dtype=np.int32),
            "eta_idx":       np.array([r.eta_idx       for r in buf.rows], dtype=np.int32),
            "peak_nr":       np.array([r.peak_nr       for r in buf.rows], dtype=np.int32),
            "eta_deg":       np.array([r.eta_deg       for r in buf.rows], dtype=np.float64),
            "center_2theta": np.array([r.center_2theta for r in buf.rows], dtype=np.float64),
            "area":          np.array([r.area          for r in buf.rows], dtype=np.float64),
            "sig":           np.array([r.sig           for r in buf.rows], dtype=np.float64),
            "gam":           np.array([r.gam           for r in buf.rows], dtype=np.float64),
            "FWHM_deg":      np.array([r.FWHM_deg      for r in buf.rows], dtype=np.float64),
            "eta_mix":       np.array([r.eta_mix       for r in buf.rows], dtype=np.float64),
            "d_spacing_A":   np.array([r.d_spacing_A   for r in buf.rows], dtype=np.float64),
            "chi_sq":        np.array([r.chi_sq        for r in buf.rows], dtype=np.float64),
        }
        for name, arr in cols.items():
            peaks.create_dataset(name, data=arr)
        peaks.create_dataset(
            "frame_key",
            data=np.array([str(r.frame_idx) for r in buf.rows], dtype=object),
            dtype=str_dt,
        )
