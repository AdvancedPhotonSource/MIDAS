"""SpotsToIndex.csv aggregation — multiple-mode output."""

from __future__ import annotations

import numpy as np

from midas_pipeline.find_grains import write_spots_to_index_csv


def test_spots_to_index_csv_ordering_and_format(tmp_path):
    """Voxels emitted in ascending order; one line per cluster row."""
    rows = {
        2: np.array([[10, 5, 0, 0, 1], [11, 6, 0, 0, 2]], dtype=np.uint64),
        0: np.array([[20, 4, 0, 0, 0]], dtype=np.uint64),
    }
    path = tmp_path / "SpotsToIndex.csv"
    write_spots_to_index_csv(path, rows)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    # Sorted ascending: voxel 0 first, then voxel 2 (two rows).
    assert lines[0].startswith("0 20 4 0 0 0")
    assert lines[1].startswith("2 10 5 0 0 1")
    assert lines[2].startswith("2 11 6 0 0 2")


def test_empty_dict_writes_empty_file(tmp_path):
    path = tmp_path / "SpotsToIndex.csv"
    write_spots_to_index_csv(path, {})
    assert path.read_text() == ""
