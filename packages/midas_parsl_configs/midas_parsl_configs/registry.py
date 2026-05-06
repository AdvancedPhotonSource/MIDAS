"""Config registry — discovers builtin + user configs, returns parsl Config.

Discovery order:

  1. ``$MIDAS_PARSL_CONFIGS_DIR`` (if set)
  2. ``~/.midas/parsl_configs/`` (user-installed configs)
  3. ``midas_parsl_configs.builtin`` (bundled, FF_HEDM/workflows lineage)

User configs win over builtin when names collide. Names are
case-insensitive and the trailing ``Config.py`` suffix is optional in
``load_config(name)``.

Each config module is expected to define a ``parsl.config.Config``
attribute. We pick it by trying these names, in order:

  ``config``, ``<name>Config``, ``<Name>Config`` (capitalised),
  ``orthrosAllConfig``, ``orthrosNewConfig`` (legacy aliases) …

The first match wins.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional


# Map short names → bundled module name (in ``builtin/``). Keep the keys
# lowercase; ``load_config`` lower()s the user-supplied name before lookup.
AVAILABLE_BUILTIN: dict[str, str] = {
    "local":      "localConfig",
    "adhoc":      "adhocConfig",
    "orthrosnew": "orthrosAllConfig",   # both Orthros configs live in this module
    "orthrosall": "orthrosAllConfig",
    "umich":      "uMichConfig",
    "marquette":  "marquetteConfig",
    "purdue":     "purdueConfig",
    "polaris":    "polarisConfig",
}


def user_configs_dir() -> Path:
    """User-writable config dir, env override beats the default."""
    env = os.environ.get("MIDAS_PARSL_CONFIGS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".midas" / "parsl_configs"


def _normalize(name: str) -> str:
    n = name.strip().lower()
    if n.endswith("config"):
        n = n[: -len("config")]
    if n.endswith("config.py"):
        n = n[: -len("config.py")]
    return n


def _user_module_path(name: str) -> Optional[Path]:
    """Locate ``<name>Config.py`` in the user dir, case-insensitively."""
    udir = user_configs_dir()
    if not udir.is_dir():
        return None
    for p in udir.iterdir():
        if p.suffix != ".py":
            continue
        stem = p.stem
        if stem.lower() == f"{name}config":
            return p
    return None


def _load_user_module(path: Path, name: str) -> ModuleType:
    """Import a user-config module from a filesystem path."""
    spec = importlib.util.spec_from_file_location(
        f"midas_parsl_configs._user_{name}", str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load user parsl config: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_builtin_module(builtin_name: str) -> ModuleType:
    return importlib.import_module(f"midas_parsl_configs.builtin.{builtin_name}")


def _candidate_attr_names(name: str, module: ModuleType) -> list[str]:
    out: list[str] = ["config"]
    out.append(f"{name}Config")
    out.append(f"{name.capitalize()}Config")
    out.append(f"{name.upper()}Config")
    # legacy aliases — orthros has two configs in one module
    if name == "orthrosnew":
        out.append("orthrosNewConfig")
    if name == "orthrosall":
        out.append("orthrosAllConfig")
    if name == "umich":
        out.append("uMichConfig")
    if name == "marquette":
        out.append("marquetteConfig")
    if name == "purdue":
        out.append("purdueConfig")
    if name == "polaris":
        out.append("config")
    if name == "local":
        out.append("localConfig")
    if name == "adhoc":
        out.append("config")
    # finally, anything ending with "Config" defined at module level
    for attr in dir(module):
        if attr.endswith("Config") and attr not in out:
            out.append(attr)
    return out


def _set_required_env(machine: str, *, n_nodes: int, n_cpus: int,
                      script_dir: Optional[str]) -> None:
    """Populate the env vars the bundled configs read at import time."""
    if "MIDAS_SCRIPT_DIR" not in os.environ:
        os.environ["MIDAS_SCRIPT_DIR"] = script_dir or os.getcwd()
    elif script_dir:
        os.environ["MIDAS_SCRIPT_DIR"] = script_dir
    os.environ["nNodes"] = str(max(1, int(n_nodes)))
    os.environ.setdefault("PROJECT_NAME", "default-project")
    os.environ.setdefault("QUEUE_NAME", "default-queue")
    os.environ.setdefault("CONDA_LOC", os.environ.get("CONDA_PREFIX", ""))


def available_configs() -> dict[str, str]:
    """Return ``{name: 'builtin' | 'user:<path>'}`` for everything visible."""
    out: dict[str, str] = {}
    udir = user_configs_dir()
    if udir.is_dir():
        for p in sorted(udir.iterdir()):
            if p.suffix != ".py":
                continue
            stem = p.stem
            if not stem.endswith("Config"):
                continue
            name = stem[: -len("Config")].lower()
            out[name] = f"user:{p}"
    for name in AVAILABLE_BUILTIN:
        out.setdefault(name, "builtin")
    return out


def register_config(name: str, source_module: str) -> None:
    """Register an additional builtin under ``name``. Mostly for tests."""
    AVAILABLE_BUILTIN[name.lower()] = source_module


def load_config(
    name: str,
    *,
    n_nodes: int = 1,
    n_cpus: int = 8,
    script_dir: Optional[str] = None,
) -> Any:
    """Resolve ``name`` to an instantiated ``parsl.config.Config``.

    User configs in ``user_configs_dir()`` win over builtins of the same name.
    Sets the env vars ``MIDAS_SCRIPT_DIR`` / ``nNodes`` / ``PROJECT_NAME``
    / ``QUEUE_NAME`` / ``CONDA_LOC`` first because the bundled FF/NF
    configs read those at import time.
    """
    norm = _normalize(name)
    _set_required_env(norm, n_nodes=n_nodes, n_cpus=n_cpus,
                      script_dir=script_dir)

    user_path = _user_module_path(norm)
    if user_path is not None:
        mod = _load_user_module(user_path, norm)
    else:
        builtin = AVAILABLE_BUILTIN.get(norm)
        if builtin is None:
            raise KeyError(
                f"Unknown parsl config {name!r}. "
                f"Known: {sorted(available_configs())}"
            )
        mod = _load_builtin_module(builtin)

    for attr in _candidate_attr_names(norm, mod):
        cfg = getattr(mod, attr, None)
        if cfg is not None:
            return cfg
    raise AttributeError(
        f"No Config object found in {mod.__name__}. "
        f"Tried: {_candidate_attr_names(norm, mod)}"
    )
