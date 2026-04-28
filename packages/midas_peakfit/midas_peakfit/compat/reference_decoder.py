"""Read C-produced ``AllPeaks_PS.bin`` and ``AllPeaks_PX.bin`` for parity testing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from midas_peakfit.postfit import N_PEAK_COLS


@dataclass
class PSData:
    n_frames: int
    n_peaks: np.ndarray  # int32 (nFrames,)
    offsets: np.ndarray  # int64 (nFrames,)
    rows_per_frame: List[np.ndarray]  # each (nPeaks_f, 29)


@dataclass
class PXData:
    n_frames: int
    nr_pixels: int
    n_peaks: np.ndarray  # int32 (nFrames,)
    offsets: np.ndarray  # int64 (nFrames,)
    pixels_per_frame: List[List[Tuple[np.ndarray, np.ndarray]]]
    # pixels_per_frame[f][p] = (y_array, z_array) of shape (nPx,) each


def read_ps(path: str | Path) -> PSData:
    with open(path, "rb") as f:
        buf = f.read()
    cursor = 0
    n_frames = int(np.frombuffer(buf, dtype=np.int32, count=1, offset=cursor)[0])
    cursor += 4
    n_peaks = np.frombuffer(buf, dtype=np.int32, count=n_frames, offset=cursor).copy()
    cursor += n_frames * 4
    offsets = np.frombuffer(buf, dtype=np.int64, count=n_frames, offset=cursor).copy()
    cursor += n_frames * 8
    header_size = cursor

    rows_per_frame: List[np.ndarray] = []
    for f_idx in range(n_frames):
        nPk = int(n_peaks[f_idx])
        if nPk == 0:
            rows_per_frame.append(np.zeros((0, N_PEAK_COLS), dtype=np.float64))
            continue
        off = int(offsets[f_idx])
        # Offsets in C are file-byte absolute, including header.
        rows = np.frombuffer(
            buf,
            dtype=np.float64,
            count=nPk * N_PEAK_COLS,
            offset=off,
        ).reshape(nPk, N_PEAK_COLS).copy()
        rows_per_frame.append(rows)
    return PSData(
        n_frames=n_frames,
        n_peaks=n_peaks,
        offsets=offsets,
        rows_per_frame=rows_per_frame,
    )


def read_px(path: str | Path) -> PXData:
    with open(path, "rb") as f:
        buf = f.read()
    cursor = 0
    n_frames = int(np.frombuffer(buf, dtype=np.int32, count=1, offset=cursor)[0])
    cursor += 4
    nr_pixels = int(np.frombuffer(buf, dtype=np.int32, count=1, offset=cursor)[0])
    cursor += 4
    n_peaks = np.frombuffer(buf, dtype=np.int32, count=n_frames, offset=cursor).copy()
    cursor += n_frames * 4
    offsets = np.frombuffer(buf, dtype=np.int64, count=n_frames, offset=cursor).copy()
    cursor += n_frames * 8

    pixels_per_frame: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    for f_idx in range(n_frames):
        nPk = int(n_peaks[f_idx])
        peaks: List[Tuple[np.ndarray, np.ndarray]] = []
        if nPk > 0:
            off = int(offsets[f_idx])
            cursor = off
            for _ in range(nPk):
                nPx = int(
                    np.frombuffer(buf, dtype=np.int32, count=1, offset=cursor)[0]
                )
                cursor += 4
                yz = np.frombuffer(
                    buf, dtype=np.int16, count=nPx * 2, offset=cursor
                ).reshape(nPx, 2).copy()
                cursor += nPx * 2 * 2
                peaks.append((yz[:, 0], yz[:, 1]))
        pixels_per_frame.append(peaks)
    return PXData(
        n_frames=n_frames,
        nr_pixels=nr_pixels,
        n_peaks=n_peaks,
        offsets=offsets,
        pixels_per_frame=pixels_per_frame,
    )


__all__ = ["PSData", "PXData", "read_ps", "read_px"]
