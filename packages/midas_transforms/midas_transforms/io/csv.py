"""CSV readers / writers for the FF-HEDM intermediate stages.

The file formats here are the contract with the C binaries — every column,
every header, every separator must match.

Header / column references:
- ``Result_*.csv``: ``MergeOverlappingPeaksAllZarr.c:357`` after qsort.
- ``Radius_*.csv``: ``CalcRadiusAllZarr.c:412``.
- ``InputAll.csv``: ``FitSetupParamsAllZarr.c`` (8 cols).
- ``InputAllExtraInfoFittingAll.csv``: same source, 18 cols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np


# --- Result_*.csv (merge output) ------------------------------------------

RESULT_HEADER = (
    "SpotID IntegratedIntensity Omega YCen ZCen IMax MinOme MaxOme "
    "SigmaR SigmaEta NrPx NrPxTot Radius Eta RawSumIntensity maskTouched FitRMSE"
)
RESULT_NCOLS = 17


def read_result_csv(path: Union[str, Path]) -> np.ndarray:
    """Read a 17-col Result_*.csv into a (N, 17) float64 array."""
    return np.loadtxt(path, skiprows=1, dtype=np.float64).reshape(-1, RESULT_NCOLS)


def write_result_csv(path: Union[str, Path], data: np.ndarray) -> None:
    """Write a (N, 17) array to Result_*.csv with the C-compatible header."""
    if data.shape[1] != RESULT_NCOLS:
        raise ValueError(f"expected {RESULT_NCOLS} columns, got {data.shape[1]}")
    fmt = " ".join(["%.6f"] * RESULT_NCOLS)
    with open(path, "w") as f:
        f.write(RESULT_HEADER + "\n")
        np.savetxt(f, data, fmt=fmt)


# --- Radius_*.csv (calc_radius output) ------------------------------------

RADIUS_HEADER = (
    "SpotID IntegratedIntensity Omega YCen ZCen IMax MinOme MaxOme "
    "Radius Theta Eta DeltaOmega NImgs RingNr GrainVolume GrainRadius "
    "PowderIntensity SigmaR SigmaEta NrPx NrPxTot RawSumIntensity "
    "maskTouched FitRMSE"
)
RADIUS_NCOLS = 24


def read_radius_csv(path: Union[str, Path]) -> np.ndarray:
    return np.loadtxt(path, skiprows=1, dtype=np.float64).reshape(-1, RADIUS_NCOLS)


def write_radius_csv(path: Union[str, Path], data: np.ndarray) -> None:
    if data.shape[1] != RADIUS_NCOLS:
        raise ValueError(f"expected {RADIUS_NCOLS} columns, got {data.shape[1]}")
    fmt = " ".join(["%.6f"] * RADIUS_NCOLS)
    with open(path, "w") as f:
        f.write(RADIUS_HEADER + "\n")
        np.savetxt(f, data, fmt=fmt)


# --- InputAll.csv ---------------------------------------------------------

INPUTALL_HEADER = "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta"
INPUTALL_NCOLS = 8


def read_inputall_csv(path: Union[str, Path]) -> np.ndarray:
    return np.loadtxt(path, skiprows=1, dtype=np.float64).reshape(-1, INPUTALL_NCOLS)


def write_inputall_csv(path: Union[str, Path], data: np.ndarray) -> None:
    if data.shape[1] != INPUTALL_NCOLS:
        raise ValueError(f"expected {INPUTALL_NCOLS} columns, got {data.shape[1]}")
    fmt = " ".join(["%.6f"] * INPUTALL_NCOLS)
    with open(path, "w") as f:
        f.write(INPUTALL_HEADER + "\n")
        np.savetxt(f, data, fmt=fmt)


# --- InputAllExtraInfoFittingAll.csv --------------------------------------

INPUTALL_EXTRA_HEADER = (
    "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta "
    "OmegaIni YOrigDetCor ZOrigDetCor YOrigNoWedge ZOrigNoWedge "
    "IntegratedIntensity RawSumIntensity FitRMSE maskTouched FitErrCode"
)
INPUTALL_EXTRA_NCOLS = 18


def read_inputall_extra_csv(path: Union[str, Path]) -> np.ndarray:
    return np.loadtxt(path, skiprows=1, dtype=np.float64).reshape(-1, INPUTALL_EXTRA_NCOLS)


def write_inputall_extra_csv(path: Union[str, Path], data: np.ndarray) -> None:
    if data.shape[1] != INPUTALL_EXTRA_NCOLS:
        raise ValueError(f"expected {INPUTALL_EXTRA_NCOLS} columns, got {data.shape[1]}")
    fmt = " ".join(["%.6f"] * INPUTALL_EXTRA_NCOLS)
    with open(path, "w") as f:
        f.write(INPUTALL_EXTRA_HEADER + "\n")
        np.savetxt(f, data, fmt=fmt)


# --- SpotsToIndex.csv -----------------------------------------------------


def read_spots_to_index(path: Union[str, Path]) -> np.ndarray:
    """Single-column CSV of spot IDs."""
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


def write_spots_to_index(path: Union[str, Path], spot_ids: Iterable[int]) -> None:
    with open(path, "w") as f:
        for sid in spot_ids:
            f.write(f"{int(sid)}\n")


# --- hkls.csv -------------------------------------------------------------


def read_hkls_csv(path: Union[str, Path]) -> np.ndarray:
    """Load hkls.csv as a (N, ncols) float64 array.

    The MIDAS hkls.csv schema is:
        col 0..2  : h, k, l
        col 3..5  : ds, theta, multiplicity (or similar — packages vary)
        col 4     : ring radius (px) in the modern schema
        ...
    We return the raw float matrix; consumers index by column.
    """
    return np.loadtxt(path, skiprows=1, dtype=np.float64)
