"""Stage: potts (PF only, optional).

Runs confidence-weighted Potts ICM smoothing on the per-voxel
posterior produced by the fuse stage. Gated on
``cfg.fusion.cw_potts_lambda > 0``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import PottsResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    if ctx.is_ff or cfg.fusion.cw_potts_lambda <= 0:
        return stub_run("potts", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    recons_dir = layer_dir / "Recons"
    posterior_path = recons_dir / "posterior.npy"
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"potts: missing {posterior_path}. Run fuse first."
        )

    posterior = np.load(str(posterior_path))
    # max_id seed from the argmax of the posterior; voxels with no
    # posterior support are marked -1 (Potts respects this invariant).
    full_max = posterior.max(axis=0)
    max_id = np.where(full_max > 0, posterior.argmax(axis=0).astype(np.int32), -1)

    LOG.info("potts(PF): lambda=%.3f, max_iter=%d, conf_floor=%.3f",
             cfg.fusion.cw_potts_lambda,
             cfg.fusion.cw_potts_max_iter,
             cfg.fusion.cw_potts_conf_floor)

    from ..potts import confidence_weighted_potts
    refined = confidence_weighted_potts(
        posterior=posterior,
        max_id=max_id,
        lam=cfg.fusion.cw_potts_lambda,
        max_iter=cfg.fusion.cw_potts_max_iter,
        conf_floor=cfg.fusion.cw_potts_conf_floor,
    )

    out_path = recons_dir / "Full_recon_max_project_grID_potts.tif"
    try:
        import tifffile
        tifffile.imwrite(str(out_path), refined.astype(np.int32))
    except ImportError:                                  # pragma: no cover
        np.save(str(out_path.with_suffix(".npy")), refined)
        out_path = out_path.with_suffix(".npy")

    n_flips = int(np.sum(refined != max_id))
    finished = time.time()
    return PottsResult(
        stage_name="potts",
        started_at=started, finished_at=finished, duration_s=finished - started,
        grain_id_map_path=str(out_path),
        n_flips=n_flips,
        n_iter=cfg.fusion.cw_potts_max_iter,
        outputs={str(out_path): ""},
        metrics={"lambda": cfg.fusion.cw_potts_lambda,
                 "n_flips": n_flips,
                 "n_iter": cfg.fusion.cw_potts_max_iter},
    )
