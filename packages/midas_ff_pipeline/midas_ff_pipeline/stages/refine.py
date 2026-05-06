"""Grain refinement via ``midas-fit-grain``."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ._base import StageContext, env_for_index_refine, run_subprocess
from .._logging import stage_timer
from ..results import RefineResult


def run(ctx: StageContext) -> RefineResult:
    started = time.time()

    paramstest = ctx.layer_dir / "paramstest.txt"
    spots_to_index = ctx.layer_dir / "SpotsToIndex.csv"
    n_seeds = sum(1 for _ in spots_to_index.open() if _.strip())

    output_dir = ctx.layer_dir / "Output"
    results_dir = ctx.layer_dir / "Results"
    output_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    # Multi-detector pinwheel: pixel loss is per-panel (each panel has its
    # own BC + Lsd), so a global pixel residual is meaningless when the
    # refiner sees obs spots from multiple panels. Switch to ``angular``
    # which compares (η, ω, 2θ) — geometry-independent — when the merged
    # paramstest carries DetParams blocks.
    is_multi_det = "\nDetParams " in ("\n" + paramstest.read_text())
    loss = ctx.config.refine_loss
    if is_multi_det and loss == "pixel":
        loss = "angular"
        from .._logging import LOG
        LOG.info("  multi-detector run → switching refine loss to 'angular'")

    with stage_timer("refinement"):
        cmd = [
            sys.executable, "-m", "midas_fit_grain",
            str(paramstest),
            "0",                                   # block_nr
            "1",                                   # n_blocks
            str(n_seeds),
            str(ctx.config.n_cpus),
            "--solver", ctx.config.refine_solver,
            "--loss", loss,
        ]
        if ctx.config.refine_mode:
            cmd += ["--mode", ctx.config.refine_mode]
        run_subprocess(
            cmd,
            cwd=ctx.layer_dir,
            stdout_path=ctx.log_dir / "refinement_out.csv",
            stderr_path=ctx.log_dir / "refinement_err.csv",
            env=env_for_index_refine(ctx.config),
        )

    finished = time.time()
    # midas-fit-grain honours OutputFolder + ResultFolder from paramstest.txt
    # (set by the transforms stage), so outputs already land in Output/ +
    # Results/ — no relocation needed.
    orient_pos_fit = results_dir / "OrientPosFit.bin"
    n_grains_refined = 0
    if orient_pos_fit.exists():
        # OrientPosFit.bin: per refined seed, several doubles.
        # Just record file size as a proxy for "non-zero output produced."
        n_grains_refined = orient_pos_fit.stat().st_size // 8

    return RefineResult(
        stage_name="refinement",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={
            str(orient_pos_fit): "",
            str(output_dir / "FitBest.bin"): "",
            str(results_dir / "Key.bin"): "",
            str(results_dir / "ProcessKey.bin"): "",
        },
        orient_pos_fit_bin=str(orient_pos_fit),
        n_grains_refined=n_grains_refined,
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [
        ctx.layer_dir / "Results" / "OrientPosFit.bin",
        ctx.layer_dir / "Output" / "FitBest.bin",
    ]
