"""Cluster dispatch (gap #8).

Loads a Parsl config via the standalone ``midas_parsl_configs`` package.
That package owns the bundled cluster configs (formerly under
``FF_HEDM/workflows/*Config.py``) and the
SLURM/PBS-submit-script-to-config generator. User-defined configs in
``~/.midas/parsl_configs/`` are picked up automatically and override
bundled configs of the same name.

``configure_dispatch`` is called at the start of ``Pipeline.run()``.
Stages still shell out via ``run_subprocess`` so the parsl load is
mainly so ``@python_app`` decorators in optional cluster code paths
(future work) start working without further plumbing.

Supported short names (bundled):
``local``, ``orthrosnew``, ``orthrosall``, ``umich``, ``marquette``,
``purdue``, ``polaris``, ``adhoc``. Anything else falls through to the
midas_parsl_configs registry, which means user-installed configs in
``~/.midas/parsl_configs/<name>Config.py`` work transparently.
"""
from __future__ import annotations

import os
from typing import Tuple

from ._logging import LOG


# Sensible (n_cpus, n_nodes) defaults per known cluster — used only when
# the user passes 0 for either.
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
    """Load the parsl config for ``machine`` and return ``(n_cpus, n_nodes)``.

    Resolution flow:

      1. Pick (n_cpus, n_nodes) defaults if the caller passed 0/-1.
      2. Surface ``MIDAS_SCRIPT_DIR`` and ``nNodes`` so the bundled +
         user configs can read them at import time.
      3. Ask :py:func:`midas_parsl_configs.load_config` for a Config.
      4. ``parsl.load(config=…)`` — best-effort. If parsl or
         midas_parsl_configs is missing, we log a warning and fall
         back to synchronous execution.
    """
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
            "synchronously (machine=%s ignored). `pip install midas-parsl-configs`.",
            machine,
        )
        return final_cpus, final_nodes

    if machine not in _DEFAULTS and machine not in available_configs():
        raise ValueError(
            f"Unknown machine {machine!r}. Bundled: {sorted(_DEFAULTS)}; "
            f"all visible: {sorted(available_configs())}. "
            f"Generate one via `midas-parsl-configs generate <submit-script> --name {machine}`."
        )

    try:
        import parsl  # type: ignore
    except ImportError:
        LOG.warning(
            "dispatch: parsl not installed — running synchronously "
            "(machine=%s ignored).", machine,
        )
        return final_cpus, final_nodes

    try:
        cfg = load_config(machine, n_nodes=final_nodes, n_cpus=final_cpus,
                          script_dir=os.environ.get("MIDAS_SCRIPT_DIR"))
    except Exception as e:
        LOG.warning("dispatch: failed to resolve parsl config for %s: %s",
                    machine, e)
        return final_cpus, final_nodes

    try:
        parsl.load(config=cfg)
        LOG.info("dispatch: parsl loaded for machine=%s, n_nodes=%d, n_cpus=%d",
                 machine, final_nodes, final_cpus)
    except Exception as e:
        LOG.warning("dispatch: parsl.load failed for %s: %s", machine, e)
    return final_cpus, final_nodes
