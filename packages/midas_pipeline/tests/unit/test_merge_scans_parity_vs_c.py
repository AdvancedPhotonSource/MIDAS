"""Bit-exact parity vs the C ``mergeScansScanning`` reference.

Gated on a frozen-fixture path under
``packages/midas_pipeline/dev/golden_data/test_pf_5grain/``. When the
user lands that fixture (post-paper-prep, per plan §11d), this test
auto-activates and exercises the gate.

Expected fixture layout once it lands::

    test_pf_5grain/
    ├── inputs/
    │   ├── InputAllExtraInfoFittingAll{0..N-1}.csv
    │   └── scan_positions.txt        # one Y per line
    ├── params/
    │   ├── n_merges.txt
    │   ├── tol_px.txt
    │   └── tol_ome.txt
    └── golden/
        ├── InputAllExtraInfoFittingAll{0..n_fin-1}_merged.csv
        └── positions.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _golden_root() -> Path | None:
    """Walk up from this test file to find the golden data directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "dev" / "golden_data" / "test_pf_5grain"
        if candidate.exists():
            return candidate
    return None


GOLDEN = _golden_root()


@pytest.mark.skipif(
    GOLDEN is None
    or not (GOLDEN / "golden" / "InputAllExtraInfoFittingAll0_merged.csv").exists(),
    reason="Golden fixture for mergeScansScanning not yet frozen.",
)
def test_merge_scans_parity_vs_c(tmp_path):
    """``merge_scans`` output must match the C reference to ``atol=1e-6``.

    The merged CSV is written via ``%f`` (6-digit precision), so 1e-6
    is the natural quantization floor. Looser-than-bit-exact (since
    ``%f`` rounds, not truncates) but still tight enough to catch any
    algorithmic divergence.
    """
    from midas_pipeline.stages.merge_scans import merge_scans

    inputs_dir = GOLDEN / "inputs"
    paths = sorted(inputs_dir.glob("InputAllExtraInfoFittingAll*.csv"))
    sp = np.loadtxt(inputs_dir / "scan_positions.txt", dtype=np.float64)
    params_dir = GOLDEN / "params"
    n_merges = int((params_dir / "n_merges.txt").read_text().strip())
    tol_px = float((params_dir / "tol_px.txt").read_text().strip())
    tol_ome = float((params_dir / "tol_ome.txt").read_text().strip())

    summary = merge_scans(
        per_scan_csvs=paths,
        scan_positions=sp,
        tol_px=tol_px,
        tol_ome=tol_ome,
        n_merges=n_merges,
        out_dir=tmp_path,
    )

    gold_dir = GOLDEN / "golden"
    for i, out_path in enumerate(summary.out_csvs):
        gold = gold_dir / f"InputAllExtraInfoFittingAll{i}_merged.csv"
        if not gold.exists():
            pytest.skip(f"golden {gold.name} not present")
        new = np.loadtxt(out_path, skiprows=1)
        ref = np.loadtxt(gold, skiprows=1)
        # The C merger appends spots in iteration order, which can
        # differ from ours when ties exist; sort both by (ring, omega,
        # spot_id) before comparing.
        order_new = np.lexsort((new[:, 4], new[:, 2], new[:, 5]))
        order_ref = np.lexsort((ref[:, 4], ref[:, 2], ref[:, 5]))
        np.testing.assert_allclose(
            new[order_new], ref[order_ref], rtol=0, atol=1e-6,
        )

    # Positions must match exactly (they're a simple mean of integers /
    # near-integers).
    gold_positions = np.loadtxt(gold_dir / "positions.csv")
    new_positions = summary.positions
    np.testing.assert_allclose(new_positions, gold_positions, atol=1e-6)
