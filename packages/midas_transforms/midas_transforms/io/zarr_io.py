"""Readers for the peakfit-output binary blobs that live alongside the
Zarr archive.

The relevant files (as produced by ``midas-peakfit`` /
``PeaksFittingOMPZarrRefactor``) are:

- ``AllPeaks_PS.bin`` — peak summary, ``[N_peaks, 29]`` float64 per frame.
- ``AllPeaks_PX.bin`` — per-peak pixel-list blob (optional, pixel-overlap mode).

File format for ``AllPeaks_PS.bin`` (from
``FF_HEDM/src/PeaksFittingConsolidatedIO.h:8-9, 162-211``)::

    Header: int32 nFrames
            int32 nPeaks[nFrames]
            int64 offsets[nFrames]    -- byte offsets from file start
    Data:   for each frame: double[nPeaks[f] x 29]

The 29 columns per peak are ``PEAK_COL_NAMES`` from the same header
(SpotID, IntegratedIntensity, Omega, YCen, ZCen, IMax, Radius, Eta, ...,
RawSumIntensity, maskTouched, FitRMSE).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


ALLPEAKS_PS_NCOLS = 29


def read_allpeaks_ps_frames(path: Union[str, Path]) -> List[np.ndarray]:
    """Read ``AllPeaks_PS.bin`` and return a list (one entry per frame) of
    ``(n_peaks, 29)`` float64 arrays.

    Mirrors ``ConsolidatedPeakReader_open`` /
    ``ConsolidatedPeakReader_getFrame`` from ``PeaksFittingConsolidatedIO.h``.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"{path}: too small to be a valid AllPeaks_PS.bin")
    n_frames = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    header_size = 4 + n_frames * 4 + n_frames * 8
    if raw.size < header_size:
        raise ValueError(
            f"{path}: header truncated (need {header_size} bytes, "
            f"have {raw.size})"
        )
    n_peaks = np.frombuffer(raw[4 : 4 + n_frames * 4], dtype=np.int32).copy()
    offsets = np.frombuffer(
        raw[4 + n_frames * 4 : 4 + n_frames * 4 + n_frames * 8], dtype=np.int64
    ).copy()

    frames: List[np.ndarray] = []
    for f in range(n_frames):
        n = int(n_peaks[f])
        if n <= 0:
            frames.append(np.empty((0, ALLPEAKS_PS_NCOLS), dtype=np.float64))
            continue
        start = int(offsets[f])
        nbytes = n * ALLPEAKS_PS_NCOLS * 8
        block = raw[start : start + nbytes]
        if block.size != nbytes:
            raise ValueError(
                f"{path}: frame {f} truncated (need {nbytes} bytes "
                f"at offset {start}, have {block.size})"
            )
        frames.append(
            np.frombuffer(block, dtype=np.float64).reshape(n, ALLPEAKS_PS_NCOLS).copy()
        )
    return frames


def read_allpeaks_ps(path: Union[str, Path]) -> np.ndarray:
    """Convenience: concatenate all frames into a single ``(N, 29)`` array."""
    return np.concatenate(read_allpeaks_ps_frames(path), axis=0)


def read_allpeaks_px(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Pixel-overlap blob — schema is variable; we return the raw bytes
    (consumers can decode per ``ConsolidatedPixelReader_getFrame``)."""
    p = Path(path)
    if not p.exists():
        return None
    return np.fromfile(p, dtype=np.uint8)
