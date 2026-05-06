"""Binning via ``midas-bin-data``.

Runs on the merged paramstest (one bin grid for the whole layer).
"""
from __future__ import annotations

import time
from pathlib import Path

from ._base import StageContext, run_subprocess
from .._logging import stage_timer
from ..results import BinningResult


def run(ctx: StageContext) -> BinningResult:
    started = time.time()

    with stage_timer("binning"):
        # One global call against the merged InputAll.csv + paramstest.txt
        # in layer_dir. cross_det_merge has already concatenated per-detector
        # InputAll.csv when running multi-detector.
        cmd = [
            "midas-bin-data",
            "--result-folder", str(ctx.layer_dir),
            "--device", ctx.config.device,
            "--dtype", ctx.config.dtype,
        ]
        run_subprocess(
            cmd,
            cwd=ctx.layer_dir,
            stdout_path=ctx.log_dir / "binning_out.csv",
            stderr_path=ctx.log_dir / "binning_err.csv",
        )

    finished = time.time()
    n_bins = (ctx.layer_dir / "nData.bin").stat().st_size // 8 \
        if (ctx.layer_dir / "nData.bin").exists() else 0

    return BinningResult(
        stage_name="binning",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={
            str(ctx.layer_dir / "Data.bin"): "",
            str(ctx.layer_dir / "nData.bin"): "",
        },
        n_bins=n_bins,
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.layer_dir / "Data.bin", ctx.layer_dir / "nData.bin"]
