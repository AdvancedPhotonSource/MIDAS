"""Stage: find_grains_stage (PF only).

Invokes the unified ``find_grains_single`` / ``find_grains_multiple``
sub-package on the consolidated indexer output. Emits the unique-grain
table (``UniqueOrientations.csv``) + sinogram binaries that the
reconstruct stage consumes.

FF mode is a no-op (FF has no find_grains step — ``process_grains``
handles that).
"""

from __future__ import annotations

import time
from pathlib import Path

from .._logging import LOG
from ..results import FindGrainsResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def _read_space_group(layer_dir: Path) -> int:
    """Best-effort space-group resolver (mirrors consolidation.py)."""
    p = layer_dir / "paramstest.txt"
    if not p.exists():
        return 225
    for line in p.read_text().splitlines():
        toks = line.split()
        if len(toks) >= 2 and toks[0] == "SpaceGroup":
            try:
                return int(toks[1])
            except ValueError:
                continue
    return 225


def run(ctx: StageContext) -> StageResult:
    if ctx.is_ff:
        # FF doesn't run find_grains; record a clean skipped row.
        return stub_run("find_grains", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    space_group = _read_space_group(layer_dir)

    # Soft skip when indexing hasn't produced the consolidated triplet.
    index_best_all = layer_dir / "Output" / "IndexBest_all.bin"
    if not index_best_all.exists():
        LOG.info("find_grains(PF): missing %s → skip.", index_best_all)
        return stub_run("find_grains", ctx)

    LOG.info("find_grains(PF): layer_dir=%s, space_group=%d",
             layer_dir, space_group)

    from ..find_grains import find_grains_single, find_grains_multiple

    if ctx.config.one_sol_per_vox:
        artifacts = find_grains_single(
            layer_dir,
            space_group=space_group,
            sino_mode=ctx.config.recon.sino_source,    # "tolerance" | "indexing"
            confidence_min=ctx.config.recon.sino_conf_min,
            scan_tolerance_um=ctx.config.recon.sino_scan_tol_um,
            cluster_misorientation_deg=ctx.config.fusion.max_ang_deg,
            n_scans=ctx.config.scan.n_scans,
        )
    else:
        artifacts = find_grains_multiple(
            layer_dir,
            space_group=space_group,
            cluster_misorientation_deg=ctx.config.fusion.max_ang_deg,
        )

    finished = time.time()
    return FindGrainsResult(
        stage_name="find_grains",
        started_at=started, finished_at=finished, duration_s=finished - started,
        unique_orientations_csv=str(getattr(artifacts, "unique_orientations_csv", "")),
        unique_index_single_key_bin=str(getattr(
            artifacts, "unique_index_single_key_bin", "",
        )),
        spots_to_index_csv=str(getattr(artifacts, "spots_to_index_csv", "")),
        n_unique_grains=int(getattr(artifacts, "n_unique_grains", 0)),
        outputs={},
        metrics={"scan_mode": "pf", "space_group": space_group,
                 "one_sol_per_vox": ctx.config.one_sol_per_vox},
    )
