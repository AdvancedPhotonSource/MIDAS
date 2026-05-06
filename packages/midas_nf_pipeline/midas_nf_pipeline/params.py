"""Canonical NF-HEDM parameter-file parser for the pipeline.

Mirrors the behaviour of ``parse_parameters`` in
``NF_HEDM/workflows/nf_MIDAS.py`` and ``nf_MIDAS_Multiple_Resolutions.py``:

  - whitespace-delimited ``Key Value [Value ...]`` lines
  - ``#``-prefix and blank-line skipping
  - special multi-value keys (LatticeParameter, BC, OmegaRange, GridMask,
    BoxSize, BCTol, GridPoints, GridRefactor)
  - integer-coerced keys (nDistances, RawStartNr) and float-coerced keys
    (TomoPixelSize)
  - everything else stored as the **first** token (string)
  - duplicate key occurrences: last one wins (matches the C
    ``MIDAS_ParamParser`` behaviour the rest of the pipeline relies on
    — e.g. multi-Lsd lines)

We also expose :func:`update_param_file` which appends/overwrites a key
in-place (used by the multi-resolution loop to bump ``GridSize`` /
``MicFileText`` between iterations) and :func:`format_param_value`
helpers for writing keys back consistently.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)


# Multi-value keys → number of float values to read from the line.
MULTI_VALUE_FLOAT_KEYS: Dict[str, int] = {
    "LatticeParameter": 6,
    "GridMask": 4,
    "BC": 2,
    "OmegaRange": 2,
    "BoxSize": 4,
    "BCTol": 2,
    "GridPoints": 12,
    "GridRefactor": 3,            # StartingGridSize, ScalingFactor, NumLoops
}

INT_KEYS = {"nDistances", "RawStartNr", "NrFilesPerDistance"}
FLOAT_KEYS = {"TomoPixelSize"}


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

def parse_parameters(param_file: str | Path) -> Dict[str, Any]:
    """Read an NF-HEDM parameter file into a dict.

    Returns
    -------
    dict
        - Multi-value keys (see :data:`MULTI_VALUE_FLOAT_KEYS`) → list of floats.
        - INT_KEYS → int.
        - FLOAT_KEYS → float.
        - Everything else → first token as ``str``.

    Notes
    -----
    Duplicate occurrences of the same key are intentionally allowed —
    the last one wins, matching the C ``MIDAS_ParamParser``'s overwrite
    semantics. This is important e.g. for multi-Lsd lines where each
    detector distance gets its own ``Lsd <value>`` line and the writer
    needs to read all of them.

    To collect all occurrences of a multi-line key (e.g. all Lsd
    values), use :func:`collect_multiline` instead.
    """
    out: Dict[str, Any] = {}
    path = Path(param_file)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            key, vals = tokens[0], tokens[1:]
            if key in MULTI_VALUE_FLOAT_KEYS:
                count = MULTI_VALUE_FLOAT_KEYS[key]
                if len(vals) < count:
                    raise ValueError(
                        f"{path}: key {key!r} expects {count} values, got "
                        f"{len(vals)} in line {raw.rstrip()!r}"
                    )
                out[key] = [float(v) for v in vals[:count]]
            elif key in INT_KEYS:
                out[key] = int(vals[0])
            elif key in FLOAT_KEYS:
                out[key] = float(vals[0])
            else:
                out[key] = vals[0]
    return out


def collect_multiline(param_file: str | Path, key: str) -> list[str]:
    """Return every value of ``key``, in file order.

    For multi-distance NF setups the param file has

        Lsd 8289.154576
        Lsd 10290.724494
        BC 985.41 17.51
        BC 985.16 24.51

    Use ``collect_multiline(p, "Lsd")`` to get both Lsd values.
    """
    out: list[str] = []
    path = Path(param_file)
    if not path.exists():
        return out
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0] == key:
                out.append(" ".join(tokens[1:]))
    return out


# ---------------------------------------------------------------------------
#  Writer (in-place key update)
# ---------------------------------------------------------------------------

def update_param_file(param_file: str | Path, updates: Dict[str, str]) -> None:
    """Overwrite values for ``updates``-keys in-place; append unknown keys.

    Mirrors ``update_param_file`` in nf_MIDAS_Multiple_Resolutions.py.
    """
    path = Path(param_file)
    pending = dict(updates)
    if path.exists():
        with open(path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    out_lines: list[str] = []
    for line in lines:
        tokens = line.split()
        key = tokens[0] if tokens else ""
        if key in pending:
            out_lines.append(f"{key} {pending.pop(key)}\n")
        else:
            out_lines.append(line)
    for key, val in pending.items():
        out_lines.append(f"{key} {val}\n")

    with open(path, "w") as f:
        f.writelines(out_lines)


def append_param_line(param_file: str | Path, key: str, value: str) -> None:
    """Append ``Key Value`` as a new line at the end of the file."""
    with open(param_file, "a") as f:
        f.write(f"\n{key} {value}\n")
