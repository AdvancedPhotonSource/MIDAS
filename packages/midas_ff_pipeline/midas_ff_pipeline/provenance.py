"""Per-layer provenance ledger ``midas_state.h5``.

Fresh schema (not ff_MIDAS-compatible) — single h5 file per layer with
a ``stages/<name>`` group per stage. Each group stores:

  * status        — "complete" | "incomplete" | "failed"
  * started_at    — UNIX seconds, float
  * finished_at   — UNIX seconds, float
  * duration_s    — float
  * inputs        — JSON-serialised {path: sha256}
  * outputs       — JSON-serialised {path: sha256}
  * metrics       — JSON-serialised dict (n_grains, n_spots, etc.)

The Pipeline writes after each successful stage and reads at startup
to pick up where it left off when ``resume="auto"``. Hashes are
sha256 of file contents at the time of write.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py


PROVENANCE_FILENAME = "midas_state.h5"


# ----- Hash helpers ---------------------------------------------------


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file into a hex digest. Empty/missing → 'missing'."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths: Iterable[str | Path]) -> Dict[str, str]:
    return {str(Path(p)): file_sha256(p) for p in paths}


# ----- Provenance store -----------------------------------------------


class ProvenanceStore:
    """Layer-scoped ledger of completed stages.

    One ``midas_state.h5`` per ``LayerNr_N`` directory.
    """

    def __init__(self, layer_dir: str | Path) -> None:
        self.layer_dir = Path(layer_dir)
        self.layer_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.layer_dir / PROVENANCE_FILENAME

    # --- record / read ---

    def record(self, stage_name: str, *,
               status: str = "complete",
               started_at: Optional[float] = None,
               finished_at: Optional[float] = None,
               duration_s: Optional[float] = None,
               inputs: Optional[Dict[str, str]] = None,
               outputs: Optional[Dict[str, str]] = None,
               metrics: Optional[Dict[str, Any]] = None) -> None:
        """Write/update a stage entry."""
        if started_at is None:
            started_at = time.time()
        if finished_at is None:
            finished_at = started_at
        if duration_s is None:
            duration_s = finished_at - started_at
        with h5py.File(self.path, "a") as f:
            grp_name = f"stages/{stage_name}"
            if grp_name in f:
                del f[grp_name]
            grp = f.create_group(grp_name)
            grp.attrs["status"] = status
            grp.attrs["started_at"] = float(started_at)
            grp.attrs["finished_at"] = float(finished_at)
            grp.attrs["duration_s"] = float(duration_s)
            grp.create_dataset("inputs",
                               data=json.dumps(inputs or {}, default=_json_default))
            grp.create_dataset("outputs",
                               data=json.dumps(outputs or {}, default=_json_default))
            grp.create_dataset("metrics",
                               data=json.dumps(metrics or {}, default=_json_default))

    def read(self, stage_name: str) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        with h5py.File(self.path, "r") as f:
            grp_name = f"stages/{stage_name}"
            if grp_name not in f:
                return None
            grp = f[grp_name]
            return {
                "status": grp.attrs.get("status", b"unknown").decode()
                          if isinstance(grp.attrs.get("status"), bytes)
                          else grp.attrs.get("status"),
                "started_at": float(grp.attrs.get("started_at", 0.0)),
                "finished_at": float(grp.attrs.get("finished_at", 0.0)),
                "duration_s": float(grp.attrs.get("duration_s", 0.0)),
                "inputs": _safe_loads(grp["inputs"][()]),
                "outputs": _safe_loads(grp["outputs"][()]),
                "metrics": _safe_loads(grp["metrics"][()]),
            }

    def all_stages(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        with h5py.File(self.path, "r") as f:
            if "stages" not in f:
                return {}
            for name in f["stages"]:
                grp = f[f"stages/{name}"]
                out[name] = {
                    "status": grp.attrs.get("status"),
                    "started_at": float(grp.attrs.get("started_at", 0.0)),
                    "finished_at": float(grp.attrs.get("finished_at", 0.0)),
                    "duration_s": float(grp.attrs.get("duration_s", 0.0)),
                    "inputs": _safe_loads(grp["inputs"][()]),
                    "outputs": _safe_loads(grp["outputs"][()]),
                    "metrics": _safe_loads(grp["metrics"][()]),
                }
                if isinstance(out[name]["status"], bytes):
                    out[name]["status"] = out[name]["status"].decode()
        return out

    # --- resume helpers ---

    def is_complete(self, stage_name: str,
                    *, expected_outputs: Optional[List[str | Path]] = None) -> bool:
        """Was the stage recorded as complete AND outputs still present + matching hash?

        If ``expected_outputs`` is given, every listed path must exist on
        disk and its hash must match the recorded one. If omitted, we
        only check status.
        """
        rec = self.read(stage_name)
        if rec is None or rec["status"] != "complete":
            return False
        recorded = rec.get("outputs") or {}
        if not expected_outputs:
            # Trust the recorded outputs — verify all listed files exist + hash matches.
            for p, h in recorded.items():
                if h == "missing":
                    continue
                if file_sha256(p) != h:
                    return False
            return True
        for p in expected_outputs:
            p_str = str(Path(p))
            if file_sha256(p) != recorded.get(p_str):
                return False
        return True

    def invalidate(self, stage_name: str) -> None:
        """Remove a stage entry (used when resuming from that stage)."""
        if not self.path.exists():
            return
        with h5py.File(self.path, "a") as f:
            grp_name = f"stages/{stage_name}"
            if grp_name in f:
                del f[grp_name]


# ----- json helpers ---------------------------------------------------


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):  # numpy/torch scalars
        return o.tolist()
    return str(o)


def _safe_loads(blob: Any) -> Any:
    if isinstance(blob, bytes):
        blob = blob.decode()
    if not blob:
        return {}
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return {}
