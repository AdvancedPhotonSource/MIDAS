"""Stage B of merged-FF seeding: merge all per-scan spot lists into one.

Thin wrapper around :func:`midas_pipeline.stages.merge_scans.merge_scans`
that drives it with ``n_merges == n_scans`` so the output is a SINGLE
``InputAllExtraInfoFittingAll0.csv`` plus a one-row ``positions.csv``.

Why a wrapper at all? The downstream FF indexer expects specific file
names (``InputAllExtraInfoFittingAll.csv``, no scan suffix) and a
single ``positions.csv`` entry. We adapt the merge_scans output to
those expectations here so the merged-FF pipeline doesn't need to
patch midas-index for a different file convention.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..stages.merge_scans import merge_scans


@dataclass
class MergeAllSummary:
    """Per-call result of :func:`merge_all_scans`."""

    merged_csv: Path                          # the renamed FF-input file
    positions_csv: Path
    n_spots_in: int
    n_spots_out: int
    outputs: Dict[str, str] = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)


def merge_all_scans(
    *,
    layer_dir: str | Path,
    n_scans: int,
    tol_px: float,
    tol_ome: float,
    ff_input_name: str = "InputAllExtraInfoFittingAll.csv",
    csv_template: str = "original_InputAllExtraInfoFittingAll{n}.csv",
    positions_in: str = "original_positions.csv",
) -> MergeAllSummary:
    """Merge all per-scan spots into one FF-style spot file.

    Parameters
    ----------
    layer_dir : path
        Directory containing ``original_positions.csv`` and the
        per-scan input CSVs.
    n_scans : int
        Total scan count.
    tol_px, tol_ome : float
        Match tolerances passed through to merge_scans.
    ff_input_name : str
        Final file name produced for the FF indexer to consume.
        Defaults to ``"InputAllExtraInfoFittingAll.csv"`` (the C
        indexer's default).
    csv_template : str
        Filename template for the per-scan inputs in ``layer_dir``.
        ``{n}`` is the 0-based scan index.
    positions_in : str
        Filename of the input scan-positions CSV in ``layer_dir``.

    Returns
    -------
    :class:`MergeAllSummary`
    """
    layer_dir = Path(layer_dir)
    positions_in_path = layer_dir / positions_in
    scan_positions = np.loadtxt(positions_in_path, dtype=np.float64).reshape(-1)
    if scan_positions.size != n_scans:
        raise ValueError(
            f"{positions_in_path} has {scan_positions.size} positions but "
            f"n_scans={n_scans}"
        )
    per_scan_csvs: List[Path] = [
        layer_dir / csv_template.format(n=i) for i in range(n_scans)
    ]

    summary = merge_scans(
        per_scan_csvs,
        scan_positions,
        tol_px=tol_px,
        tol_ome=tol_ome,
        n_merges=n_scans,                # merge all into a single fin scan
        out_dir=layer_dir,
    )
    # merge_scans wrote ``InputAllExtraInfoFittingAll0.csv`` (the one
    # fin-scan output); rename it to the FF indexer's expected default
    # so downstream stages don't need a custom RefinementFileName.
    src = summary.out_csvs[0]
    dst = layer_dir / ff_input_name
    if src != dst:
        shutil.copy2(src, dst)
    outputs = {str(dst): "", str(summary.positions_csv): ""}
    return MergeAllSummary(
        merged_csv=dst,
        positions_csv=summary.positions_csv,
        n_spots_in=summary.n_spots_in,
        n_spots_out=summary.n_spots_out,
        outputs=outputs,
        metrics={
            "n_fin_scans": 1,
            "n_total_in": summary.n_spots_in,
            "n_total_out": summary.n_spots_out,
            "tol_px": tol_px,
            "tol_ome": tol_ome,
            "n_merges": n_scans,
        },
    )
