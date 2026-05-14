"""Stage: consolidation. P1 thin-shell — implementation lands in P8."""
from __future__ import annotations

from ._base import StageContext
from ._stub import stub_run
from ..results import StageResult


def run(ctx: StageContext) -> StageResult:
    return stub_run("consolidation", ctx)
