"""Common helpers for stage modules.

The ``StageContext`` is a mutable execution context handed to every
stage. It carries the immutable ``PipelineConfig`` plus the per-layer /
per-run path bookkeeping the stage needs.

P1 keeps the multi-detector machinery (``DetectorConfig`` list) out of
this — single-detector is the common case for the scaffold; multi-det
lands in P1.5 or P3 when the stage-internal logic actually consumes it.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .._logging import LOG
from ..config import PipelineConfig


@dataclass
class StageContext:
    """Mutable execution context handed to each stage."""

    config: PipelineConfig
    layer_nr: int
    layer_dir: Path
    log_dir: Path
    merged_paramstest: Optional[Path] = None

    @property
    def scan_mode(self) -> str:
        return self.config.scan.scan_mode

    @property
    def is_ff(self) -> bool:
        return self.config.is_ff

    @property
    def is_pf(self) -> bool:
        return self.config.is_pf


def run_subprocess(cmd: Sequence[str], *,
                   cwd: str | Path,
                   stdout_path: str | Path,
                   stderr_path: str | Path,
                   env: Optional[dict] = None,
                   timeout: Optional[int] = None) -> int:
    """Run a subprocess and tee stdout/stderr to per-stage log files."""
    cmd_list = [str(c) for c in cmd]
    log_cmd = " ".join(shlex.quote(c) for c in cmd_list)
    LOG.debug("$ %s   (cwd=%s)", log_cmd, cwd)

    Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)

    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        proc = subprocess.run(
            cmd_list, cwd=str(cwd),
            stdout=out, stderr=err,
            env={**os.environ, **(env or {})},
            timeout=timeout,
            check=False,
        )

    if proc.returncode != 0:
        LOG.error("subprocess failed (rc=%d): %s", proc.returncode, log_cmd)
        try:
            tail = Path(stderr_path).read_text().strip().splitlines()[-10:]
            LOG.error("stderr tail:\n  %s", "\n  ".join(tail))
        except Exception:
            pass
        raise subprocess.CalledProcessError(proc.returncode, cmd_list)
    return proc.returncode


def env_for_index_refine(config: PipelineConfig) -> dict:
    """Common env for indexer + refiner: dtype, device, group size."""
    return {
        "MIDAS_INDEX_DTYPE": config.dtype,
        "MIDAS_INDEX_DEVICE": config.device,
        "MIDAS_INDEX_GROUP_SIZE": str(config.indexer_group_size),
        "MIDAS_FIT_GRAIN_DTYPE": config.dtype,
        "MIDAS_FIT_GRAIN_DEVICE": config.device,
    }


def hash_inputs(paths: Iterable[Path]) -> dict[str, str]:
    from ..provenance import file_sha256
    return {str(p): file_sha256(p) for p in paths}
