"""Incremental HDF5 checkpoint tracking for the NF pipeline.

This is a pip-installable port of ``utils/pipeline_state.py``: same
public API, but self-contained (no ``utils/version.py`` dependency —
provenance is stamped from this package's ``__version__`` plus
``midas_nf_preprocess``, ``midas_nf_fitorientation``, ``midas_hkls``,
and ``midas_stress`` versions).

Usage::

    from midas_nf_pipeline.state import PipelineH5

    with PipelineH5(h5_path, "nf_midas", args_namespace, param_text) as ph5:
        run_preprocessing(...)
        ph5.mark("preprocessing")
        run_image_processing(...)
        ph5.mark("image_processing", data={"voxels/position": pos_array})
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Default gzip compression for all datasets
COMPRESSION = {"compression": "gzip", "compression_opts": 4}


# ---------------------------------------------------------------------------
#  Provenance helpers (no `utils/version.py` dependency)
# ---------------------------------------------------------------------------

def _midas_versions() -> dict[str, str]:
    """Collect versions of every MIDAS package we depend on."""
    out: dict[str, str] = {}
    try:
        from . import __version__ as v
        out["midas_nf_pipeline"] = v
    except Exception:
        pass
    for name in (
        "midas_nf_preprocess", "midas_nf_fitorientation",
        "midas_hkls", "midas_stress", "midas_diffract",
    ):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return out


def stamp_provenance(grp: h5py.Group) -> None:
    """Write package versions + creation timestamp into ``grp.attrs``."""
    grp.attrs["created"] = datetime.datetime.now().isoformat()
    for k, v in _midas_versions().items():
        grp.attrs[k] = str(v)


# ---------------------------------------------------------------------------
#  PipelineH5 class
# ---------------------------------------------------------------------------

class PipelineH5:
    """Context manager for incremental pipeline state tracking in HDF5.

    On enter:
      - creates or opens the H5 file
      - writes ``/provenance/`` and ``/pipeline_state/`` if this is a fresh file
      - stores the parameter file text and CLI args for reproducibility

    Methods
    -------
    mark(stage_name, data=None) :
        Record a completed stage, optionally write datasets at the same time.
    write_dataset(path, array, attrs=None) :
        Convenience for writing compressed datasets.
    completed :
        Property returning the list of completed stage names.
    is_complete(stage_name) :
        Check if a stage was already completed.
    reset_from(stage_name, stage_order) :
        Clear ``stage_name`` and everything after it (for ``--restart-from``).
    """

    def __init__(
        self,
        h5_path: str | os.PathLike,
        workflow_type: str,
        args_namespace: Any = None,
        param_text: str = "",
    ):
        self.h5_path = str(h5_path)
        self.workflow_type = workflow_type
        self.param_text = param_text
        self._h5: Optional[h5py.File] = None

        if args_namespace is None:
            self.args_json = "{}"
        elif isinstance(args_namespace, dict):
            self.args_json = json.dumps(args_namespace, default=str)
        else:
            self.args_json = json.dumps(vars(args_namespace), default=str)

    # ----- context manager -----------------------------------------------

    def __enter__(self) -> "PipelineH5":
        os.makedirs(os.path.dirname(os.path.abspath(self.h5_path)) or ".", exist_ok=True)
        is_new = not os.path.exists(self.h5_path)
        self._h5 = h5py.File(self.h5_path, "a")
        if is_new:
            self._init_fresh()
        else:
            if "provenance" in self._h5:
                self._h5["provenance"].attrs["last_opened"] = (
                    datetime.datetime.now().isoformat()
                )
            logger.info(
                f"Resumed pipeline H5: {self.h5_path} "
                f"({len(self.completed)} stages complete)"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._h5 is not None:
            try:
                if "pipeline_state" in self._h5:
                    self._h5["pipeline_state"].attrs["last_update"] = (
                        datetime.datetime.now().isoformat()
                    )
                self._h5.flush()
                self._h5.close()
            except Exception:
                pass
            self._h5 = None
        return False

    # ----- public API ----------------------------------------------------

    def mark(self, stage_name: str, data: Optional[dict] = None) -> None:
        """Record a pipeline stage as complete."""
        ps = self._h5.require_group("pipeline_state")
        stages_grp = ps.require_group("completed_stages")
        idx = len(stages_grp)
        stages_grp.create_dataset(str(idx), data=stage_name)
        ps.attrs["current_stage"] = stage_name
        ts_grp = ps.require_group("timestamps/per_stage")
        ts_grp.attrs[stage_name] = datetime.datetime.now().isoformat()
        if data:
            for path, value in data.items():
                self.write_dataset(path, value)
        self._h5.flush()
        logger.info(f"Pipeline stage complete: {stage_name}")

    def write_dataset(
        self, path: str, value: Any, attrs: Optional[dict] = None,
        overwrite: bool = True,
    ) -> None:
        if overwrite and path in self._h5:
            del self._h5[path]
        if isinstance(value, np.ndarray):
            ds = self._h5.create_dataset(path, data=value, **COMPRESSION)
        elif isinstance(value, str):
            ds = self._h5.create_dataset(path, data=value)
        elif isinstance(value, (int, float, bool)):
            ds = self._h5.create_dataset(path, data=value)
        else:
            arr = np.asarray(value)
            ds = self._h5.create_dataset(path, data=arr, **COMPRESSION)
        if attrs:
            for k, v in attrs.items():
                ds.attrs[k] = v

    def write_group_attrs(self, group_path: str, attrs: dict) -> None:
        grp = self._h5.require_group(group_path)
        for k, v in attrs.items():
            grp.attrs[k] = v

    @property
    def completed(self) -> list[str]:
        if self._h5 is None or "pipeline_state/completed_stages" not in self._h5:
            return []
        grp = self._h5["pipeline_state/completed_stages"]
        out: list[str] = []
        for i in range(len(grp)):
            key = str(i)
            if key in grp:
                v = grp[key][()]
                out.append(v.decode() if isinstance(v, bytes) else v)
        return out

    def is_complete(self, stage_name: str) -> bool:
        return stage_name in self.completed

    @property
    def h5(self) -> h5py.File:
        return self._h5

    # ----- internals -----------------------------------------------------

    def _init_fresh(self) -> None:
        prov = self._h5.require_group("provenance")
        stamp_provenance(prov)
        prov.attrs["parameter_file"] = self.param_text

        ps = self._h5.require_group("pipeline_state")
        ps.attrs["workflow_type"] = self.workflow_type
        ps.attrs["command_line_args"] = self.args_json
        ps.attrs["start"] = datetime.datetime.now().isoformat()
        ps.attrs["last_update"] = ps.attrs["start"]
        ps.attrs["current_stage"] = ""
        ps.require_group("completed_stages")
        ps.require_group("timestamps/per_stage")
        self._h5.flush()
        logger.info(
            f"Initialized pipeline H5: {self.h5_path} "
            f"(workflow={self.workflow_type})"
        )

    def reset_from(self, stage_name: str, stage_order: list[str]) -> None:
        """Clear ``stage_name`` and everything after it."""
        if stage_name not in stage_order:
            logger.warning(f"Stage '{stage_name}' not in stage_order, skipping reset")
            return
        target_idx = stage_order.index(stage_name)
        current = self.completed
        keep = [s for s in current if s in stage_order and stage_order.index(s) < target_idx]
        keep += [s for s in current if s not in stage_order]
        if "pipeline_state/completed_stages" in self._h5:
            del self._h5["pipeline_state/completed_stages"]
        grp = self._h5.require_group("pipeline_state/completed_stages")
        for i, s in enumerate(keep):
            grp.create_dataset(str(i), data=s)
        self._h5["pipeline_state"].attrs["current_stage"] = keep[-1] if keep else ""
        self._h5.flush()

    def restore_dataset(self, path: str):
        if path not in self._h5:
            raise KeyError(f"Dataset '{path}' not found in {self.h5_path}")
        v = self._h5[path][()]
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return v.item()
        return v


# ---------------------------------------------------------------------------
#  Standalone helpers
# ---------------------------------------------------------------------------

def get_completed_stages(h5_path: str | os.PathLike) -> list[str]:
    """Read completed stages from an existing pipeline H5."""
    if not os.path.exists(h5_path):
        return []
    with h5py.File(h5_path, "r") as h5:
        if "pipeline_state/completed_stages" not in h5:
            return []
        grp = h5["pipeline_state/completed_stages"]
        out: list[str] = []
        for i in range(len(grp)):
            key = str(i)
            if key in grp:
                v = grp[key][()]
                out.append(v.decode() if isinstance(v, bytes) else v)
        return out


def can_skip_to(h5_path: str | os.PathLike, target_stage: str, stage_order: list[str]) -> bool:
    if target_stage not in stage_order:
        return False
    completed = set(get_completed_stages(h5_path))
    target_idx = stage_order.index(target_stage)
    for i in range(target_idx):
        if stage_order[i] not in completed:
            return False
    return True


def load_resume_info(h5_path: str | os.PathLike) -> dict:
    info = {
        "completed_stages": [], "workflow_type": "",
        "args_json": "{}", "param_text": "",
    }
    if not os.path.exists(h5_path):
        return info
    with h5py.File(h5_path, "r") as h5:
        if "pipeline_state/completed_stages" in h5:
            grp = h5["pipeline_state/completed_stages"]
            for i in range(len(grp)):
                key = str(i)
                if key in grp:
                    v = grp[key][()]
                    info["completed_stages"].append(
                        v.decode() if isinstance(v, bytes) else v
                    )
        if "pipeline_state" in h5:
            info["workflow_type"] = h5["pipeline_state"].attrs.get("workflow_type", "")
            info["args_json"] = h5["pipeline_state"].attrs.get("command_line_args", "{}")
        if "provenance" in h5:
            info["param_text"] = h5["provenance"].attrs.get("parameter_file", "")
    return info


def find_resume_stage(h5_path: str | os.PathLike, stage_order: list[str]) -> str:
    """Determine the first incomplete stage in the pipeline."""
    completed = set(get_completed_stages(h5_path))
    for stage in stage_order:
        if stage not in completed:
            return stage
    return ""
