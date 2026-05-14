"""Helper for P1 thin-shell stages.

Every stage in P1 starts as a placeholder that records a stage_name +
duration in the provenance ledger and returns a StageResult with
``skipped=True``. As P2-P8 stream owners take over each stage, they
replace the ``run(ctx)`` body in that stage's module — the function
signature is the contract and stays stable.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .._logging import LOG, stage_timer
from ..results import StageResult

if TYPE_CHECKING:
    from ._base import StageContext


def stub_run(stage_name: str, ctx: "StageContext") -> StageResult:
    """Return a skipped StageResult for a stage that hasn't been implemented yet.

    Logs a clear warning so callers know the pipeline is incomplete.
    """
    LOG.warning(
        "stage '%s' is a P1 stub — implementation lands in a later phase. "
        "Skipping in scan_mode=%s.", stage_name, ctx.scan_mode,
    )
    now = time.time()
    return StageResult(
        stage_name=stage_name,
        started_at=now,
        finished_at=now,
        duration_s=0.0,
        inputs={},
        outputs={},
        metrics={"stub": True, "scan_mode": ctx.scan_mode},
        skipped=True,
    )
