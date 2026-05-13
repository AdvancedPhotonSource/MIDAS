"""End-to-end-ish wiring tests: SeededRegion → FitOutput → AllPeaks_PS_moment.bin.

Verifies that ``compute_moments=True`` propagates through ``seed_region``,
``fit_regions``, and ``write_consolidated_peak_files`` to produce a
correctly-formatted moment sidecar, and that the sidecar is *absent* by
default. The fitted-peak path also touches ``compute_moment_sigma`` via
``_build_moment_rows``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from midas_peakfit.connected import Region
from midas_peakfit.fit import N_MOMENT_COLS, fit_regions
from midas_peakfit.output import FrameAccumulator, write_consolidated_peak_files
from midas_peakfit.seeds import seed_region
from midas_peakfit.uncertainty import QUALITY_OK


def _make_synthetic_region(amp: float = 2000.0, sigma: float = 2.0) -> tuple:
    """Build a 16x16-pixel image containing a single Gaussian peak and the
    matching ``Region`` object."""
    nr_pixels = 64
    img = np.zeros((nr_pixels, nr_pixels), dtype=np.float64)
    y0, z0 = 32.0, 32.0
    Y, Z = np.meshgrid(np.arange(nr_pixels), np.arange(nr_pixels), indexing="xy")
    img = amp * np.exp(-((Y - y0) ** 2 + (Z - z0) ** 2) / (2 * sigma ** 2))
    threshold = 5.0
    mask = (img > threshold)
    rows, cols = np.where(mask)
    intensities = img[rows, cols]
    region = Region(
        id=0,
        pixel_rows=rows.astype(np.int32),
        pixel_cols=cols.astype(np.int32),
        intensities=intensities,
        threshold=threshold,
        raw_sum=float(intensities.sum()),
    )
    return img, region, nr_pixels


def test_seed_region_default_omits_higher_moments():
    """Default fast path (compute_moments=False) → M_0 + quality always
    populated, higher moments None."""
    img, region, _ = _make_synthetic_region()
    sr = seed_region(
        region, img, mask=np.zeros((0, 0), dtype=np.int8),
        Ycen=32.0, Zcen=32.0, int_sat=1e9,
        max_n_peaks=50, panels=[],
    )
    assert sr is not None
    assert sr.peak_M0 is not None
    assert sr.peak_quality is not None
    assert sr.peak_M0.shape == (sr.n_peaks,)
    assert sr.peak_quality.shape == (sr.n_peaks,)
    assert sr.peak_quality.dtype == np.int8
    # High-amplitude Gaussian → comfortably OK regime
    assert sr.peak_quality[0] == QUALITY_OK
    # Higher moments NOT computed
    assert sr.peak_M2_R is None
    assert sr.peak_M2_Eta is None
    assert sr.peak_M4_R is None
    assert sr.peak_M4_Eta is None


def test_seed_region_compute_moments_populates_M2_M4():
    """compute_moments=True → all four higher-moment arrays populated."""
    img, region, _ = _make_synthetic_region()
    sr = seed_region(
        region, img, mask=np.zeros((0, 0), dtype=np.int8),
        Ycen=32.0, Zcen=32.0, int_sat=1e9,
        max_n_peaks=50, panels=[],
        compute_moments=True,
    )
    assert sr is not None
    for arr in (sr.peak_M2_R, sr.peak_M2_Eta, sr.peak_M4_R, sr.peak_M4_Eta):
        assert arr is not None
        assert arr.shape == (sr.n_peaks,)
        assert np.all(np.isfinite(arr))
    # M_4 ≥ M_2² for any non-degenerate distribution (kurtosis ≥ 1).
    assert (sr.peak_M4_R + 1e-9 >= sr.peak_M2_R ** 2).all()
    assert (sr.peak_M4_Eta + 1e-9 >= sr.peak_M2_Eta ** 2).all()


def test_seed_only_path_emits_moment_rows():
    """doPeakFit=0 short-circuit must still emit moment rows when the seed
    region carries higher-moment data."""
    img, region, _ = _make_synthetic_region()
    sr = seed_region(
        region, img, mask=np.zeros((0, 0), dtype=np.int8),
        Ycen=32.0, Zcen=32.0, int_sat=1e9,
        max_n_peaks=50, panels=[],
        compute_moments=True,
    )
    outs, _ = fit_regions(
        [sr], omega=0.0, Ycen=32.0, Zcen=32.0,
        do_peak_fit=0, local_maxima_only=0,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    assert len(outs) == 1
    fo = outs[0]
    assert fo.rows_moment is not None
    assert fo.rows_moment.shape == (sr.n_peaks, N_MOMENT_COLS)
    # Col 0 = M_0 > 0; col 1 = quality flag; col 2..5 = u(M_1/M_2) finite/positive
    assert (fo.rows_moment[:, 0] > 0).all()
    assert np.all(np.isin(fo.rows_moment[:, 1], [0, 1, 2]))
    assert (fo.rows_moment[:, 2:] >= 0).all()


def test_seed_only_path_omits_moment_rows_by_default():
    """Without compute_moments the FitOutput must NOT carry moment rows."""
    img, region, _ = _make_synthetic_region()
    sr = seed_region(
        region, img, mask=np.zeros((0, 0), dtype=np.int8),
        Ycen=32.0, Zcen=32.0, int_sat=1e9,
        max_n_peaks=50, panels=[],
    )
    outs, _ = fit_regions(
        [sr], omega=0.0, Ycen=32.0, Zcen=32.0,
        do_peak_fit=0, local_maxima_only=0,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    assert outs[0].rows_moment is None


def test_sidecar_writer_present_when_data_carried(out_tmpdir):
    """When any accumulator carries moment rows, AllPeaks_PS_moment.bin is
    emitted with the correct header + payload layout."""
    img, region, nr_pixels = _make_synthetic_region()
    sr = seed_region(
        region, img, mask=np.zeros((0, 0), dtype=np.int8),
        Ycen=32.0, Zcen=32.0, int_sat=1e9,
        max_n_peaks=50, panels=[],
        compute_moments=True,
    )
    outs, _ = fit_regions(
        [sr], omega=0.0, Ycen=32.0, Zcen=32.0,
        do_peak_fit=0, local_maxima_only=0,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    acc = FrameAccumulator()
    for fo in outs:
        acc.add(fo)
    assert acc.has_moment

    out_dir = Path(out_tmpdir) / "Temp"
    write_consolidated_peak_files(
        [acc],
        n_total_frames=1, start_frame=0, end_frame=1,
        nr_pixels=nr_pixels,
        out_folder=out_dir,
    )
    moment_path = out_dir / "AllPeaks_PS_moment.bin"
    assert moment_path.exists()

    # Decode header: int32 nFrames, int32[nFrames] nPeaks, int64[nFrames] off
    with open(moment_path, "rb") as f:
        raw = f.read()
    n_frames = np.frombuffer(raw[:4], dtype=np.int32)[0]
    assert n_frames == 1
    n_peaks_arr = np.frombuffer(raw[4:4 + 4 * n_frames], dtype=np.int32)
    offsets = np.frombuffer(
        raw[4 + 4 * n_frames:4 + 4 * n_frames + 8 * n_frames], dtype=np.int64
    )
    n_peaks = int(n_peaks_arr[0])
    assert n_peaks == sr.n_peaks
    off = int(offsets[0])
    payload = np.frombuffer(
        raw[off:off + n_peaks * N_MOMENT_COLS * 8], dtype=np.float64
    ).reshape(n_peaks, N_MOMENT_COLS)
    # Quality flags are valid; M_0 positive; u(M_1) and u(M_2) finite ≥ 0.
    assert np.all(np.isin(payload[:, 1], [0, 1, 2]))
    assert (payload[:, 0] > 0).all()
    assert np.all(np.isfinite(payload[:, 2:]))
    assert (payload[:, 2:] >= 0).all()


def test_sidecar_absent_by_default(out_tmpdir):
    """No moment data carried → no sidecar file."""
    img, region, nr_pixels = _make_synthetic_region()
    sr = seed_region(
        region, img, mask=np.zeros((0, 0), dtype=np.int8),
        Ycen=32.0, Zcen=32.0, int_sat=1e9,
        max_n_peaks=50, panels=[],
    )
    outs, _ = fit_regions(
        [sr], omega=0.0, Ycen=32.0, Zcen=32.0,
        do_peak_fit=0, local_maxima_only=0,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    acc = FrameAccumulator()
    for fo in outs:
        acc.add(fo)
    assert not acc.has_moment

    out_dir = Path(out_tmpdir) / "Temp"
    write_consolidated_peak_files(
        [acc],
        n_total_frames=1, start_frame=0, end_frame=1,
        nr_pixels=nr_pixels,
        out_folder=out_dir,
    )
    assert not (out_dir / "AllPeaks_PS_moment.bin").exists()
