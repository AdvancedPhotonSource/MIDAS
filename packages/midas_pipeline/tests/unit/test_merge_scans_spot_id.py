"""Spot-ID renumbering contract for merge_scans.

The C output (``mergeScansScanning.c:210-211``) overwrites col 4 with
``i + 1`` for ``i`` in ``[0, n_all)``. We replicate that — verified
explicitly here so any future API change that touches the renumber step
fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.stages.merge_scans import merge_scans


def _make_18col(y, z, omega, weight, spot_id, ring):
    row = np.zeros(18, dtype=np.float64)
    row[0] = y
    row[1] = z
    row[2] = omega
    row[3] = weight
    row[4] = spot_id
    row[5] = ring
    row[16] = 100.0 + ring
    row[17] = 200.0 + ring
    return row


def _write_18col(path: Path, rows: np.ndarray) -> None:
    header = " ".join(f"col{i}" for i in range(18))
    with path.open("w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.6f}" for v in r) + "\n")


def test_spot_id_renumbered_contiguous_1_indexed(tmp_path: Path):
    """Output spot IDs are 1..N, contiguous, regardless of input IDs."""
    # 3 scans, each with 4 spots. Input IDs are scattered (300, 17, 99, ...)
    # so the output cannot accidentally match input IDs.
    paths = []
    for s in range(3):
        rows = []
        for ring in (1, 2):
            for sp_idx in range(2):
                y = 10.0 * ring + 1.0 * sp_idx
                z = 20.0 * ring + 0.5 * sp_idx
                omega = 5.0 * ring + 2.0 * sp_idx
                spot_id = 1_000 * s + 100 * ring + sp_idx + 7
                rows.append(_make_18col(y, z, omega, 1.0, spot_id, float(ring)))
        p = tmp_path / f"InputAllExtraInfoFittingAll{s}.csv"
        _write_18col(p, np.array(rows))
        paths.append(p)

    summary = merge_scans(
        per_scan_csvs=paths,
        scan_positions=np.array([0.0, 2.0, 4.0]),
        tol_px=0.5, tol_ome=0.5, n_merges=3,
        out_dir=tmp_path / "out",
    )

    arr = np.loadtxt(summary.out_csvs[0], skiprows=1)
    ids = arr[:, 4].astype(int)
    n = arr.shape[0]
    np.testing.assert_array_equal(ids, np.arange(1, n + 1))


def test_spot_id_renumbered_per_merged_group(tmp_path: Path):
    """Each merged-output file restarts the 1..N renumber from 1."""
    # 4 scans → with n_merges=2, two output groups, each independently
    # renumbered starting at 1.
    paths = []
    for s in range(4):
        rows = []
        for ring in (1, 2):
            y = 10.0 * ring + 0.5 * s    # different across scans → no merge
            rows.append(_make_18col(y, 1.0, 1.0, 1.0, 9000 + s, float(ring)))
        p = tmp_path / f"InputAllExtraInfoFittingAll{s}.csv"
        _write_18col(p, np.array(rows))
        paths.append(p)
    summary = merge_scans(
        per_scan_csvs=paths,
        scan_positions=np.array([0.0, 1.0, 2.0, 3.0]),
        tol_px=0.01, tol_ome=0.01, n_merges=2,
        out_dir=tmp_path / "out",
    )
    assert len(summary.out_csvs) == 2
    for out_path in summary.out_csvs:
        arr = np.loadtxt(out_path, skiprows=1)
        ids = arr[:, 4].astype(int)
        np.testing.assert_array_equal(ids, np.arange(1, len(arr) + 1))
