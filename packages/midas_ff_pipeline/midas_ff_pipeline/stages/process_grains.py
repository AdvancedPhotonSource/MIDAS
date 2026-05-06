"""Grain consolidation via ``midas_process_grains``.

Pure-Python — no C ProcessGrains binary required. Default mode is
``spot_aware`` (the package's tested default; see midas_process_grains
README for the three available modes).
"""
from __future__ import annotations

import time
from pathlib import Path

from ._base import StageContext
from .._logging import LOG, stage_timer
from ..results import ProcessGrainsResult


def run(ctx: StageContext) -> ProcessGrainsResult:
    started = time.time()
    paramstest = ctx.layer_dir / "paramstest.txt"

    # Lazy-import so that import-time failures don't tear down the whole
    # ``midas_ff_pipeline`` package on a system that doesn't have all
    # siblings installed.
    from midas_process_grains import ProcessGrains

    with stage_timer("process_grains"):
        pg = ProcessGrains.from_param_file(str(paramstest))
        result = pg.run(mode=ctx.config.process_grains_mode)
        # ProcessGrains.run returns a ProcessGrainsResult; persist it.
        result.write(str(ctx.layer_dir))

    finished = time.time()
    grains_csv = ctx.layer_dir / "Grains.csv"

    n_grains = int(getattr(result, "n_grains", 0) or 0)
    if not n_grains and grains_csv.exists():
        with grains_csv.open() as fp:
            first = fp.readline()
            if first.startswith("%NumGrains"):
                try:
                    n_grains = int(first.split()[1])
                except (IndexError, ValueError):
                    n_grains = 0
    LOG.info("  ProcessGrains (%s): %d grains", ctx.config.process_grains_mode, n_grains)

    return ProcessGrainsResult(
        stage_name="process_grains",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={
            str(grains_csv): "",
            str(ctx.layer_dir / "SpotMatrix.csv"): "",
            str(ctx.layer_dir / "GrainIDsKey.csv"): "",
        },
        grains_csv=str(grains_csv),
        n_grains=n_grains,
        metrics={"mode": ctx.config.process_grains_mode, "n_grains": n_grains},
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.layer_dir / "Grains.csv"]
