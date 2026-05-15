"""Stage: refinement.

PF mode: invokes ``midas_fit_grain.scan_driver.refine_scanning_block``
on the consolidated ``Output/IndexBest_all.bin`` produced by the
indexing stage. Each voxel's top candidate is refined under the
scan-aware filter; per-voxel ``Results/Result_OrientPos_voxel_N.csv``
emitted for consolidation_pf to aggregate.

FF mode: stub — FF callers use the existing ``midas-ff-pipeline``
which shells out to ``midas-fit-grain`` directly.
"""

from __future__ import annotations

import time
from pathlib import Path

from .._logging import LOG
from ..results import RefineResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def run(ctx: StageContext) -> StageResult:
    if ctx.is_ff:
        return stub_run("refinement", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    results_dir = layer_dir / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    index_best_all = layer_dir / "Output" / "IndexBest_all.bin"
    positions_csv = layer_dir / "positions.csv"
    paramstest = layer_dir / "paramstest.txt"
    # Soft skip when upstream artefacts aren't present (smoke runs /
    # partial pipelines).
    if not index_best_all.exists() or not positions_csv.exists() or not paramstest.exists():
        LOG.info("refinement(PF): missing upstream artefacts → skip.")
        return stub_run("refinement", ctx)

    LOG.info("refinement(PF): index_best_all=%s, results_dir=%s",
             index_best_all, results_dir)

    # Lazy imports to keep FF runs lean.
    from midas_fit_grain.config import FitConfig
    from midas_fit_grain.driver import _build_model, _read_hkls_csv
    from midas_fit_grain.observations import ObservedSpots
    from midas_fit_grain.io_binary import read_extra_info
    from midas_fit_grain.scan_driver import refine_scanning_block
    import torch

    # Build FitConfig from paramstest. The legacy reader in
    # midas-fit-grain.config.from_param_file handles the canonical keys.
    cfg = FitConfig.from_param_file(paramstest)
    cfg.scan_pos_tol_um = (
        ctx.config.scan.scan_pos_tol_um
        if ctx.config.scan.scan_pos_tol_um > 0
        else (ctx.config.scan.beam_size_um / 2.0)
    )
    cfg.friedel_symmetric_scan_filter = ctx.config.scan.friedel_symmetric_scan_filter
    cfg.beam_size_um = ctx.config.scan.beam_size_um
    cfg.position_mode = ctx.config.refinement.position_mode
    cfg.mode = ctx.config.refinement.mode or "all_at_once"
    cfg.solver = ctx.config.refinement.solver
    cfg.loss = ctx.config.refinement.loss

    # Build the forward model + observations once for the whole voxel loop.
    device = torch.device(ctx.config.device)
    dtype = torch.float64 if ctx.config.dtype == "float64" else torch.float32

    extra_info_path = layer_dir / "ExtraInfo.bin"
    if not extra_info_path.exists():
        raise FileNotFoundError(
            f"refinement(PF): missing {extra_info_path}; transforms didn't run."
        )
    extra = read_extra_info(extra_info_path, mmap=True)
    obs = ObservedSpots.from_extra_info(extra, device=device, dtype=dtype)

    hkls_path = layer_dir / "hkls.csv"
    if cfg.RhoD > 0.0 and cfg.Lsd > 0.0:
        import math
        max_two_theta_deg = 2.0 * math.degrees(math.atan(cfg.RhoD / cfg.Lsd))
    else:
        max_two_theta_deg = 180.0
    hkls_int, thetas_deg, ring_nr = _read_hkls_csv(
        hkls_path, cfg.RingNumbers, max_two_theta_deg=max_two_theta_deg,
    )
    model, pred_ring_slot = _build_model(
        cfg, device=device, dtype=dtype,
        hkls_int=hkls_int, thetas_deg=thetas_deg, ring_nr=ring_nr,
    )

    voxel_results = refine_scanning_block(
        cfg,
        index_best_all=index_best_all,
        positions_csv=positions_csv,
        results_dir=results_dir,
        model=model,
        obs=obs,
        pred_ring_slot=pred_ring_slot,
        voxel_block_nr=0, voxel_n_blocks=1,
    )

    finished = time.time()
    return RefineResult(
        stage_name="refinement",
        started_at=started, finished_at=finished, duration_s=finished - started,
        orient_pos_fit_bin="",
        results_dir=str(results_dir),
        n_grains_refined=0,
        n_voxels_refined=int(len(voxel_results)),
        outputs={str(results_dir): ""},
        metrics={"scan_mode": "pf",
                 "n_voxels_processed": len(voxel_results),
                 "position_mode": cfg.position_mode,
                 "mode": cfg.mode},
    )
