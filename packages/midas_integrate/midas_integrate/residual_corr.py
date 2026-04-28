"""Residual radial-correction map.

Pure-Python (numpy) port of the ``DGResidualCorr`` machinery declared in
FF_HEDM/src/DetectorGeometry.h. The map is a smooth empirical ΔR(Y, Z)
field at detector resolution, generated from calibrant ring residuals
after the analytical p0–p14 model converges. ``mapper_build_map``
applies it as ``Rt += bilinear_interp(map, Y, Z)`` after the analytical
distortion + parallax stage.

Storage convention
------------------
* ``map_arr`` is a 2-D float64 array of shape ``(NrPixelsZ, NrPixelsY)``
  in row-major order — the same layout the C reads with
  ``map[z * Ny + y]``.
* When no residual correction is configured, the kernels expect a
  sentinel ``(0, 0)`` array and ``corr_present == 0``. In that mode the
  helper returns 0 for every lookup.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ResidualCorrection:
    """Detector-resolution ΔR(Y, Z) map in pixels."""

    map: np.ndarray              # shape (NrPixelsZ, NrPixelsY), float64
    NrPixelsY: int
    NrPixelsZ: int

    def __post_init__(self) -> None:
        if self.map.ndim != 2:
            raise ValueError(
                f"residual map must be 2-D; got shape {self.map.shape}"
            )
        if (self.map.shape[0] != self.NrPixelsZ
                or self.map.shape[1] != self.NrPixelsY):
            raise ValueError(
                f"residual map shape {self.map.shape} does not match "
                f"(NrPixelsZ={self.NrPixelsZ}, NrPixelsY={self.NrPixelsY})"
            )
        if self.map.dtype != np.float64 or not self.map.flags["C_CONTIGUOUS"]:
            self.map = np.ascontiguousarray(self.map, dtype=np.float64)


def load_residual_correction_map(
    filename: str | Path,
    NrPixelsY: int,
    NrPixelsZ: int,
) -> ResidualCorrection:
    """Read a binary ``ΔR(Y, Z)`` map (NrPixelsY × NrPixelsZ doubles).

    Mirrors the load step in ``DetectorMapper.c`` — file is a raw stream
    of ``NrPixelsY * NrPixelsZ`` little-endian float64 values laid out in
    ``map[z * NrPixelsY + y]`` order.
    """
    expected = NrPixelsY * NrPixelsZ * 8
    raw = Path(filename).read_bytes()
    if len(raw) != expected:
        raise ValueError(
            f"residual correction map size mismatch in {filename}: "
            f"got {len(raw)} bytes, expected {expected} "
            f"({NrPixelsY}x{NrPixelsZ} doubles)"
        )
    arr = np.frombuffer(raw, dtype=np.float64).reshape(NrPixelsZ, NrPixelsY)
    return ResidualCorrection(
        map=np.ascontiguousarray(arr, dtype=np.float64),
        NrPixelsY=NrPixelsY,
        NrPixelsZ=NrPixelsZ,
    )


def empty_residual_corr_array() -> np.ndarray:
    """Sentinel ``(0, 0)`` map signalling 'no residual correction'."""
    return np.zeros((0, 0), dtype=np.float64)


def lookup_python(corr: Optional[ResidualCorrection],
                  Y: float, Z: float) -> float:
    """Pure-Python bilinear lookup, identical to ``dg_residual_corr_lookup``.

    Used only by tests and the pure-Python mapper fallback.
    """
    if corr is None:
        return 0.0
    NY = corr.NrPixelsY
    NZ = corr.NrPixelsZ
    if NY == 0 or NZ == 0:
        return 0.0
    y = Y if Y > 0.0 else 0.0
    z = Z if Z > 0.0 else 0.0
    if y >= NY - 1.0:
        y = NY - 1.001
    if z >= NZ - 1.0:
        z = NZ - 1.001
    y0 = int(y)
    z0 = int(z)
    fy = y - y0
    fz = z - z0
    m = corr.map
    v00 = m[z0,     y0]
    v10 = m[z0,     y0 + 1]
    v01 = m[z0 + 1, y0]
    v11 = m[z0 + 1, y0 + 1]
    return (v00 * (1 - fy) * (1 - fz)
            + v10 * fy       * (1 - fz)
            + v01 * (1 - fy) * fz
            + v11 * fy       * fz)
