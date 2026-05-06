"""Common helpers for stage modules."""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .._logging import LOG, stage_timer
from ..config import PipelineConfig
from ..detector import DetectorConfig


@dataclass
class StageContext:
    """Mutable execution context handed to each stage.

    Holds the paths and config the stage needs. Pipeline updates
    fields (e.g., ``layer_nr``, ``layer_dir``) before each call.
    """

    config: PipelineConfig
    detectors: List[DetectorConfig]
    layer_nr: int
    layer_dir: Path                       # result_dir/LayerNr_<N>
    log_dir: Path                         # layer_dir/midas_log

    # Set by the cross-det merge stage — used by binning/index/refine downstream.
    merged_paramstest: Optional[Path] = None

    @property
    def is_multi_detector(self) -> bool:
        return len(self.detectors) > 1

    def detector_dir(self, det: DetectorConfig) -> Path:
        """Per-detector working dir (only used for multi-det runs)."""
        return self.layer_dir / f"Det_{det.det_id}"

    def stage_dir(self, det: Optional[DetectorConfig] = None) -> Path:
        """Where a stage writes its outputs.

        Single-det runs write directly into ``layer_dir``. Multi-det
        per-detector stages write into ``layer_dir/Det_<id>/``.
        """
        if det is None or not self.is_multi_detector:
            return self.layer_dir
        return self.detector_dir(det)


# --- Subprocess helper ---


def run_subprocess(cmd: Sequence[str], *,
                   cwd: str | Path,
                   stdout_path: str | Path,
                   stderr_path: str | Path,
                   env: Optional[dict] = None,
                   timeout: Optional[int] = None) -> int:
    """Run a subprocess and tee stdout/stderr to per-stage log files.

    Returns the exit code. Logs exit code at INFO; raises on non-zero.
    """
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
        # Surface the last few lines of stderr so callers see the failure
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
    """Convenience wrapper for the provenance store."""
    from ..provenance import file_sha256
    return {str(p): file_sha256(p) for p in paths}
