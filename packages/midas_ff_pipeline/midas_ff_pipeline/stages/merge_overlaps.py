"""Merge overlapping peaks per detector via ``midas-merge-peaks``."""
from __future__ import annotations

import time
from pathlib import Path

from ._base import StageContext, run_subprocess
from .._logging import stage_timer
from ..results import MergeOverlapsResult


def run(ctx: StageContext) -> MergeOverlapsResult:
    started = time.time()

    with stage_timer("merge_overlaps"):
        for det in ctx.detectors:
            zip_path = Path(det.zarr_path)
            stage_dir = ctx.stage_dir(det)
            cmd = [
                "midas-merge-peaks", str(zip_path),
                "--result-folder", str(stage_dir),
                "--device", ctx.config.device,
                "--dtype", ctx.config.dtype,
            ]
            run_subprocess(
                cmd,
                cwd=stage_dir,
                stdout_path=ctx.log_dir / f"merge_overlaps_det{det.det_id}_out.csv",
                stderr_path=ctx.log_dir / f"merge_overlaps_det{det.det_id}_err.csv",
            )

    finished = time.time()
    return MergeOverlapsResult(
        stage_name="merge_overlaps",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        metrics={"n_detectors": len(ctx.detectors)},
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [Path(d.zarr_path) for d in ctx.detectors]
