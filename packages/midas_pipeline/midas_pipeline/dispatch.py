"""Cluster dispatch.

Loads a Parsl config via the standalone ``midas_parsl_configs`` package.
Mirrors ``midas_ff_pipeline.dispatch`` — same defaults, same fallback to
synchronous execution when parsl/midas-parsl-configs aren't installed.
"""

from __future__ import annotations

import os
from typing import Tuple

from ._logging import LOG


_DEFAULTS: dict[str, tuple[int, int]] = {
    "local":      (8, 1),
    "orthrosnew": (32, 11),
    "orthrosall": (64, 5),
    "umich":      (36, 1),
    "marquette":  (36, 1),
    "purdue":     (128, 1),
    "polaris":    (32, 1),
    "adhoc":      (8, 1),
}


def configure_dispatch(machine: str, n_nodes: int, n_cpus: int) -> Tuple[int, int]:
    """Resolve ``(n_cpus, n_nodes)`` and best-effort load parsl config."""
    machine = (machine or "local").lower()
    default_cpus, default_nodes = _DEFAULTS.get(machine, (8, 1))
    final_cpus = n_cpus if n_cpus and n_cpus > 0 else default_cpus
    final_nodes = n_nodes if n_nodes and n_nodes > 0 else default_nodes

    os.environ["nNodes"] = str(final_nodes)
    os.environ.setdefault("MIDAS_SCRIPT_DIR", os.getcwd())

    try:
        from midas_parsl_configs import load_config, available_configs
    except ImportError:
        LOG.warning(
            "dispatch: midas_parsl_configs not installed — running "
            "synchronously (machine=%s ignored).", machine,
        )
        return final_cpus, final_nodes

    if machine not in _DEFAULTS and machine not in available_configs():
        raise ValueError(
            f"Unknown machine {machine!r}. Bundled: {sorted(_DEFAULTS)}; "
            f"all visible: {sorted(available_configs())}."
        )

    try:
        import parsl  # type: ignore
    except ImportError:
        LOG.warning("dispatch: parsl not installed — synchronous fallback.")
        return final_cpus, final_nodes

    try:
        cfg = load_config(machine, n_nodes=final_nodes, n_cpus=final_cpus,
                          script_dir=os.environ.get("MIDAS_SCRIPT_DIR"))
    except Exception as e:
        LOG.warning("dispatch: failed to resolve parsl config: %s", e)
        return final_cpus, final_nodes

    try:
        parsl.load(config=cfg)
        LOG.info("dispatch: parsl loaded (machine=%s, n_nodes=%d, n_cpus=%d)",
                 machine, final_nodes, final_cpus)
    except Exception as e:
        LOG.warning("dispatch: parsl.load failed: %s", e)
    return final_cpus, final_nodes
