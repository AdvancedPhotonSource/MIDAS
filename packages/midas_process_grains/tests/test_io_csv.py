"""IO smoke tests for the CSV writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from midas_process_grains.io.csv import (
    write_grain_ids_key_csv,
    write_grains_csv,
    write_spot_matrix_csv,
)


def test_write_grains_csv_writes_headers_and_rows(tmp_path: Path):
    n = 2
    grains = {
        "ids": np.array([1, 2], dtype=np.int32),
        "orient_mat": np.tile(np.eye(3).reshape(-1), (n, 1)),
        "positions": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        "strains_lab": np.zeros((n, 6)),
        "grain_radius": np.array([5.0, 7.0]),
        "confidence": np.array([0.97, 0.99]),
    }
    p = tmp_path / "Grains.csv"
    write_grains_csv(p, grains, sg_nr=225, lattice=(3.6,) * 3 + (90.0,) * 3)
    txt = p.read_text()
    assert "%NumGrains 2" in txt
    assert "%SpaceGroup:225" in txt or "SpaceGroup:225" in txt
    assert "%ID\tO11" in txt
    # Body lines
    body = [l for l in txt.splitlines() if l and not l.startswith("%")]
    assert len(body) == 2
    fields = body[0].split("\t")
    assert int(fields[0]) == 1
    np.testing.assert_allclose([float(x) for x in fields[1:10]], np.eye(3).reshape(-1))


def test_write_spot_matrix_csv(tmp_path: Path):
    rows = np.array([
        [1, 101, 0.5, 100.0, 200.0, 0.55, 1.2, 1, 5.0, 7.0, 2.4, 1e-4],
        [1, 102, 0.6, 110.0, 210.0, 0.65, 1.3, 1, 5.5, 7.5, 2.4, 1e-4],
    ])
    p = tmp_path / "SpotMatrix.csv"
    write_spot_matrix_csv(p, rows)
    text = p.read_text()
    assert "%GrainID\tSpotID" in text
    body = [l for l in text.splitlines() if l and not l.startswith("%")]
    assert len(body) == 2
    f = body[0].split("\t")
    assert int(f[0]) == 1
    assert int(f[1]) == 101
    assert int(f[7]) == 1


def test_write_grain_ids_key_csv(tmp_path: Path):
    clusters = [
        (5, 0, [(7, 1), (9, 2)]),
        (10, 3, []),
    ]
    p = tmp_path / "GrainIDsKey.csv"
    write_grain_ids_key_csv(p, clusters)
    lines = p.read_text().strip().splitlines()
    assert lines == ["5 0 7 1 9 2", "10 3"]
