"""Stage: indexing.

PF mode: invokes ``midas_index.Indexer.run_scanning`` on the per-voxel
grid built from ``positions.csv``. Writes the consolidated
``Output/IndexBest_all.bin`` consumed by ``find_grains`` and the
refinement stage.

FF mode: stub — FF callers use the existing ``midas-ff-pipeline``
which shells out to ``midas-index`` directly. Wiring the FF path
through this stage is on the P9 release work.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import IndexResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def run(ctx: StageContext) -> StageResult:
    if ctx.is_ff:
        return stub_run("indexing", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    out_dir = layer_dir / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "IndexBest_all.bin"

    paramstest = layer_dir / "paramstest.txt"
    positions_csv = layer_dir / "positions.csv"
    # Soft fail: when upstream stages haven't produced their artefacts
    # (e.g. running a smoke-test or partial pipeline), skip cleanly so
    # the orchestrator can continue. Hard errors only fire from inside
    # the indexer body once we know we *should* be indexing.
    if not paramstest.exists() or not positions_csv.exists():
        LOG.info("indexing(PF): missing paramstest or positions.csv → skip.")
        return stub_run("indexing", ctx)

    LOG.info("indexing(PF): paramstest=%s positions=%s out=%s",
             paramstest, positions_csv, out_path)

    # Lazy-import so FF runs that never touch this stage don't pay the
    # midas-index import cost.
    from midas_index.indexer import Indexer

    scan_positions = np.loadtxt(positions_csv, dtype=np.float64).reshape(-1)
    n_scans = int(scan_positions.size)
    if n_scans < 2:
        raise ValueError(
            f"indexing(PF): positions.csv has {n_scans} entries; "
            "scan mode needs n_scans >= 2."
        )

    # Change to layer_dir so load_observations resolves hkls.csv etc.
    import os
    cwd0 = Path.cwd()
    os.chdir(layer_dir)
    try:
        ind = Indexer.from_param_file(paramstest, device=ctx.config.device,
                                      dtype=ctx.config.dtype)
        ind.params.multi_solution_output = True
        ind.params.friedel_symmetric_scan_filter = (
            ctx.config.scan.friedel_symmetric_scan_filter
        )
        if ctx.config.scan.scan_pos_tol_um > 0:
            ind.params.scan_pos_tol_um = ctx.config.scan.scan_pos_tol_um
        elif ctx.config.scan.beam_size_um > 0:
            ind.params.scan_pos_tol_um = ctx.config.scan.beam_size_um / 2.0
        ind.params.OutputFolder = str(layer_dir)

        ind.load_observations(cwd=layer_dir)
        n_processed = ind.run_scanning(
            scan_positions=scan_positions,
            out_path=out_path,
            num_procs=ctx.config.n_cpus,
            seed_group_size=ctx.config.indexer_group_size,
        )
    finally:
        os.chdir(cwd0)

    finished = time.time()
    return IndexResult(
        stage_name="indexing",
        started_at=started, finished_at=finished, duration_s=finished - started,
        index_best_bin="",
        index_best_all_bin=str(out_path),
        n_voxels_indexed=int(n_processed),
        outputs={str(out_path): ""},
        metrics={"scan_mode": "pf", "n_voxels": n_processed,
                 "n_scans": n_scans},
    )
