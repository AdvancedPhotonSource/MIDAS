"""Test binary output writers + round-trip readers."""
import numpy as np

from midas_peakfit.fit import FitOutput
from midas_peakfit.output import FrameAccumulator, write_consolidated_peak_files
from midas_peakfit.compat.reference_decoder import read_ps, read_px
from midas_peakfit.postfit import N_PEAK_COLS


def test_writer_round_trip_single_frame(out_tmpdir):
    rows = np.zeros((2, N_PEAK_COLS), dtype=np.float64)
    rows[0, 0] = 1.0  # SpotID
    rows[0, 3] = 100.5  # YCen
    rows[0, 4] = 200.5  # ZCen
    rows[1, 0] = 2.0
    rows[1, 3] = 50.0
    rows[1, 4] = 60.0

    pix_y = np.array([10, 11, 12], dtype=np.int16)
    pix_z = np.array([20, 21, 22], dtype=np.int16)

    fo = FitOutput(region_id=1, rows=rows, pixel_y=pix_y, pixel_z=pix_z)
    acc = FrameAccumulator()
    acc.add(fo)

    out_dir = out_tmpdir / "Temp"
    ps_path, px_path = write_consolidated_peak_files(
        [acc],
        n_total_frames=1,
        start_frame=0,
        end_frame=1,
        nr_pixels=2048,
        out_folder=out_dir,
    )

    # Round-trip PS
    ps = read_ps(ps_path)
    assert ps.n_frames == 1
    assert ps.n_peaks[0] == 2
    np.testing.assert_array_equal(ps.rows_per_frame[0], rows)

    # Round-trip PX
    px = read_px(px_path)
    assert px.n_frames == 1
    assert px.nr_pixels == 2048
    assert px.n_peaks[0] == 2
    # Two peaks, both reusing the same pixel set
    for peak_y, peak_z in px.pixels_per_frame[0]:
        np.testing.assert_array_equal(peak_y, pix_y)
        np.testing.assert_array_equal(peak_z, pix_z)


def test_writer_multi_block_layout(out_tmpdir):
    """Frames outside [start, end) get nPeaks=0 with full-length headers."""
    rows = np.zeros((1, N_PEAK_COLS), dtype=np.float64)
    rows[0, 0] = 1.0
    fo = FitOutput(region_id=1, rows=rows, pixel_y=np.array([0], dtype=np.int16),
                   pixel_z=np.array([0], dtype=np.int16))
    acc = FrameAccumulator()
    acc.add(fo)

    # Block 1 of 3 (frame 1)
    out_dir = out_tmpdir / "Temp"
    write_consolidated_peak_files(
        [acc],
        n_total_frames=3,
        start_frame=1,
        end_frame=2,
        nr_pixels=2048,
        out_folder=out_dir,
    )
    ps = read_ps(out_dir / "AllPeaks_PS.bin")
    assert ps.n_frames == 3
    assert ps.n_peaks.tolist() == [0, 1, 0]
    assert ps.rows_per_frame[1].shape[0] == 1
    assert ps.rows_per_frame[0].shape[0] == 0
    assert ps.rows_per_frame[2].shape[0] == 0
