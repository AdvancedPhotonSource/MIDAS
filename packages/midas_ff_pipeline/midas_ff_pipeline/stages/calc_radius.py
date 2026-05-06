"""Per-detector calc-radius via ``midas-calc-radius``."""
from __future__ import annotations

import time
from pathlib import Path

from ._base import StageContext, run_subprocess
from .._logging import stage_timer
from ..results import CalcRadiusResult


def run(ctx: StageContext) -> CalcRadiusResult:
    started = time.time()

    with stage_timer("calc_radius"):
        for det in ctx.detectors:
            zip_path = Path(det.zarr_path)
            stage_dir = ctx.stage_dir(det)
            cmd = [
                "midas-calc-radius", str(zip_path),
                "--result-folder", str(stage_dir),
                "--device", ctx.config.device,
                "--dtype", ctx.config.dtype,
            ]
            run_subprocess(
                cmd,
                cwd=stage_dir,
                stdout_path=ctx.log_dir / f"calc_radius_det{det.det_id}_out.csv",
                stderr_path=ctx.log_dir / f"calc_radius_det{det.det_id}_err.csv",
            )

    finished = time.time()
    return CalcRadiusResult(
        stage_name="calc_radius",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        metrics={"n_detectors": len(ctx.detectors)},
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    # calc-radius writes data into the zarr; no explicit file.
    return [Path(d.zarr_path) for d in ctx.detectors]
