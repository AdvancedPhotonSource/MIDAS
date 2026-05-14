"""Pipeline-wide logger + stage timer.

Single logger named ``midas_pipeline`` to keep child stage logs
co-located with the orchestrator's own.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

LOG = logging.getLogger("midas_pipeline")


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent root-logger setup. Safe to call from CLI or notebook."""
    root = logging.getLogger("midas_pipeline")
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


@contextmanager
def stage_timer(stage_name: str) -> Iterator[dict]:
    """Time a stage and yield a dict that gets populated with start/end/duration."""
    info: dict[str, float] = {"started_at": time.time()}
    LOG.info("→ %s", stage_name)
    try:
        yield info
    except Exception:
        info["finished_at"] = time.time()
        info["duration_s"] = info["finished_at"] - info["started_at"]
        LOG.exception("✗ %s failed after %.2fs", stage_name, info["duration_s"])
        raise
    info["finished_at"] = time.time()
    info["duration_s"] = info["finished_at"] - info["started_at"]
    LOG.info("✓ %s — %.2fs", stage_name, info["duration_s"])
