"""Stage: binning.

Wires ``midas_transforms.bin_data_unified`` into the orchestrator. The
unified entry point dispatches on whether ``scan_positions`` was passed:

- FF mode (``scan_mode='ff'``): calls ``bin_data`` (writes 9-col
  Spots.bin); behaviour is bit-identical to today's FF flow.
- PF mode (``scan_mode='pf'``): calls ``bin_data_scanning`` (writes
  10-col Spots.bin + ``voxel_scan_pos.bin`` sidecar + per-scan ID map).

The stage is a thin shell — all binning logic lives in
``midas_transforms.bin_data``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from .._logging import LOG
from ..results import BinningResult
from ._base import StageContext

if TYPE_CHECKING:
    from ..config import PipelineConfig


def run(ctx: StageContext) -> BinningResult:
    """Run the binning stage in either FF or PF mode."""
    started = time.time()

    layer_dir = ctx.layer_dir
    cfg = ctx.config

    # Resolve out_dir: place outputs alongside the per-layer inputs.
    out_dir = layer_dir

    if ctx.is_pf:
        return _run_pf(ctx, started, out_dir)
    return _run_ff(ctx, started, out_dir)


def _run_ff(ctx: StageContext, started: float, out_dir: Path) -> BinningResult:
    """FF-mode binning: delegate to ``midas_transforms.bin_data``.

    Inputs:
        - ``InputAll.csv`` (8 or 9 cols) in ``layer_dir``.
        - ``InputAllExtraInfoFittingAll.csv`` (18 cols) in ``layer_dir``.
        - ``paramstest.txt`` in ``layer_dir``.

    Outputs (written under ``out_dir == layer_dir``):
        - ``Spots.bin`` (N, 9) float64.
        - ``ExtraInfo.bin`` (N, 16) float64.
        - ``Data.bin``, ``nData.bin`` (unless ``NoSaveAll==1``).
    """
    from midas_transforms.bin_data import bin_data

    inputs = {
        "input_all_csv": str(ctx.layer_dir / "InputAll.csv"),
        "input_all_extra_csv": str(ctx.layer_dir / "InputAllExtraInfoFittingAll.csv"),
        "paramstest": str(ctx.layer_dir / "paramstest.txt"),
    }

    # Check that the inputs are present; absent inputs are a fast-fail.
    missing = [k for k, v in inputs.items() if not Path(v).exists()]
    if missing:
        LOG.warning(
            "binning(ff): inputs missing (%s); marking stage as skipped",
            ", ".join(missing),
        )
        finished = time.time()
        return BinningResult(
            stage_name="binning",
            started_at=started,
            finished_at=finished,
            duration_s=finished - started,
            inputs={k: v for k, v in inputs.items() if Path(v).exists()},
            outputs={},
            metrics={"scan_mode": "ff", "missing_inputs": missing},
            skipped=True,
        )

    res = bin_data(
        result_folder=ctx.layer_dir,
        out_dir=out_dir,
        device=ctx.config.device,
        dtype=ctx.config.dtype,
        write=True,
    )

    outputs = {
        "spots_bin": str(out_dir / "Spots.bin"),
        "extra_info_bin": str(out_dir / "ExtraInfo.bin"),
    }
    if (out_dir / "Data.bin").exists():
        outputs["data_bin"] = str(out_dir / "Data.bin")
    if (out_dir / "nData.bin").exists():
        outputs["ndata_bin"] = str(out_dir / "nData.bin")

    n_bins = res.n_ring_bins * res.n_eta_bins * res.n_ome_bins
    finished = time.time()
    return BinningResult(
        stage_name="binning",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        inputs=inputs,
        outputs=outputs,
        metrics={
            "scan_mode": "ff",
            "n_spots": int(res.spots.shape[0]),
            "n_ring_bins": int(res.n_ring_bins),
            "n_eta_bins": int(res.n_eta_bins),
            "n_ome_bins": int(res.n_ome_bins),
        },
        n_bins=n_bins,
    )


def _run_pf(ctx: StageContext, started: float, out_dir: Path) -> BinningResult:
    """PF-mode binning: delegate to ``midas_transforms.bin_data_scanning``.

    Inputs:
        - ``InputAllExtraInfoFittingAll{0..n_scans-1}.csv`` per scan.
        - ``paramstest.txt``.

    Outputs (written under ``out_dir``):
        - ``Spots.bin`` (N, 10) float64 — col 9 is scanNr.
        - ``ExtraInfo.bin`` (N, 16).
        - ``IDsMergedScanning.csv``.
        - ``voxel_scan_pos.bin`` — float64 (n_scans,) — 1-D Y per scan.
        - ``positions.csv`` — legacy C-indexer sidecar.
        - ``Data.bin``, ``nData.bin`` (unless ``NoSaveAll==1``).
    """
    from midas_transforms.bin_data import bin_data_scanning

    n_scans = ctx.config.scan.n_scans
    scan_positions = ctx.config.scan.scan_positions

    # Fail-fast when nothing's there yet — pf_MIDAS sometimes runs binning
    # before merge_scans has produced merged per-scan CSVs.
    found = []
    for s in range(n_scans):
        if (ctx.layer_dir / f"InputAllExtraInfoFittingAll{s}.csv").exists():
            found.append(s)
    if not found:
        LOG.warning(
            "binning(pf): no per-scan InputAllExtraInfoFittingAll*.csv files "
            "found in %s — marking stage as skipped",
            ctx.layer_dir,
        )
        finished = time.time()
        return BinningResult(
            stage_name="binning",
            started_at=started,
            finished_at=finished,
            duration_s=finished - started,
            inputs={"layer_dir": str(ctx.layer_dir)},
            outputs={},
            metrics={"scan_mode": "pf", "n_scans": n_scans, "n_csvs_found": 0},
            skipped=True,
        )

    res = bin_data_scanning(
        result_folder=ctx.layer_dir,
        n_scans=n_scans,
        scan_positions=scan_positions,
        out_dir=out_dir,
        device=ctx.config.device,
        dtype=ctx.config.dtype,
        write=True,
        write_positions_csv=True,
        strict_files=False,
    )

    outputs = {
        "spots_bin": str(out_dir / "Spots.bin"),
        "extra_info_bin": str(out_dir / "ExtraInfo.bin"),
        "voxel_scan_pos_bin": str(out_dir / "voxel_scan_pos.bin"),
        "ids_merged_scanning_csv": str(out_dir / "IDsMergedScanning.csv"),
        "positions_csv": str(out_dir / "positions.csv"),
    }
    if (out_dir / "Data.bin").exists():
        outputs["data_bin"] = str(out_dir / "Data.bin")
    if (out_dir / "nData.bin").exists():
        outputs["ndata_bin"] = str(out_dir / "nData.bin")

    n_bins = res.n_ring_bins * res.n_eta_bins * res.n_ome_bins
    finished = time.time()
    return BinningResult(
        stage_name="binning",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        inputs={
            "layer_dir": str(ctx.layer_dir),
            "n_scans": str(n_scans),
        },
        outputs=outputs,
        metrics={
            "scan_mode": "pf",
            "n_spots": int(res.spots.shape[0]),
            "n_scans": int(n_scans),
            "n_ring_bins": int(res.n_ring_bins),
            "n_eta_bins": int(res.n_eta_bins),
            "n_ome_bins": int(res.n_ome_bins),
        },
        n_bins=n_bins,
    )
