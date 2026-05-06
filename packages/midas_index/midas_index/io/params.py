"""paramstest.txt parser.

Mirrors `ReadParams` from `FF_HEDM/src/IndexerOMP.c:1281-1547`.

Format: one key per line, "Key value [value...]". Repeated keys
(RingNumbers, RingsToExcludeFraction, RingRadii, OmegaRange, BoxSize)
accumulate. Aliases collapse to the canonical field. Unknown keys
emit a warning then are skipped (matches C printf at line 1527).
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ..params import IndexerParams

# Canonical alias map: source key -> field on IndexerParams.
# The order in C uses startswith on a trailing space, so longer
# names are checked before substrings (e.g. "Distance " before any
# would-be "Dist*" prefix).  We replicate that with a single dict
# keyed by the *exact prefix* the C parser uses.
_FLOAT_KEYS: dict[str, str] = {
    "px": "px",
    "Wavelength": "Wavelength",
    "Distance": "Distance",
    "Lsd": "Distance",
    "Rsample": "Rsample",
    "Hbeam": "Hbeam",
    "StepsizePos": "StepsizePos",
    "StepSizePos": "StepsizePos",     # alias used by midas-fit-setup writer
    "StepsizeOrient": "StepsizeOrient",
    "StepSizeOrient": "StepsizeOrient",
    "MarginOme": "MarginOme",
    "MarginRadius": "MarginRad",
    "MarginRadial": "MarginRadial",
    "MarginEta": "MarginEta",
    "EtaBinSize": "EtaBinSize",
    "OmeBinSize": "OmeBinSize",
    "MinMatchesToAcceptFrac": "MinMatchesToAcceptFrac",
    "Completeness": "MinMatchesToAcceptFrac",
    "ExcludePoleAngle": "ExcludePoleAngle",
    "MinEta": "ExcludePoleAngle",   # IndexerOMP.c:1454 — aliases ExcludePoleAngle
}

_INT_KEYS: dict[str, str] = {
    "SpaceGroup": "SpaceGroup",
    "UseFriedelPairs": "UseFriedelPairs",
}

_STR_KEYS: dict[str, str] = {
    "SpotsFileName": "SpotsFileName",
    "IDsFileName": "IDsFileName",
    "OutputFolder": "OutputFolder",
}


def read_params(path: str | Path) -> IndexerParams:
    """Parse a paramstest.txt and return an IndexerParams.

    Mirrors `ReadParams` semantics line-for-line.
    """
    p = IndexerParams()
    ring_radii_user: list[float] = []  # parallel to ring_numbers_in_order

    with open(path, "r") as fp:
        for raw in fp:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # The C ``FitSetupZarr`` writer emits ``key value;`` lines (trailing
            # semicolons on numerics). Strip ``;`` from each token so the float
            # / int conversions below work transparently for both writers.
            tokens = [t.rstrip(";") for t in stripped.split()]
            tokens = [t for t in tokens if t]
            if not tokens:
                continue
            key = tokens[0]
            args = tokens[1:]

            # --- repeated/structured keys first ---
            if key == "RingNumbers":
                p.RingNumbers.append(int(args[0]))
                continue
            if key == "RingsToExcludeFraction":
                p.RingsToReject.append(int(args[0]))
                continue
            if key == "RingRadii":
                ring_radii_user.append(float(args[0]))
                continue
            if key == "OmegaRange":
                p.OmegaRanges.append((float(args[0]), float(args[1])))
                continue
            if key == "BoxSize":
                p.BoxSizes.append(
                    (float(args[0]), float(args[1]), float(args[2]), float(args[3]))
                )
                continue
            if key == "GrainsFile":
                p.isGrainsInput = True
                p.GrainsFileName = args[0]
                continue
            if key in ("LatticeParameter", "LatticeConstant"):
                # 6 floats expected; if fewer present, fall back to scalar a (rare).
                if len(args) >= 6:
                    p.LatticeConstant = (
                        float(args[0]), float(args[1]), float(args[2]),
                        float(args[3]), float(args[4]), float(args[5]),
                    )
                else:
                    a = float(args[0])
                    p.LatticeConstant = (a, a, a, 90.0, 90.0, 90.0)
                continue
            if key == "BigDetSize":
                # Deprecated; parsed for backward compat then ignored.
                continue

            # --- multi-detector (pinwheel) per-panel blocks ---
            if key == "DetParams":
                # DetParams det_id Lsd y_bc z_bc tx ty tz p0..p10
                if len(args) >= 7:
                    det_id = int(float(args[0]))
                    p.DetParams[det_id] = {
                        "lsd": float(args[1]),
                        "y_bc": float(args[2]),
                        "z_bc": float(args[3]),
                        "tx": float(args[4]),
                        "ty": float(args[5]),
                        "tz": float(args[6]),
                        "p_distortion": [float(v) for v in args[7:7 + 11]],
                    }
                continue
            if key.startswith("RingRadii_Det"):
                # RingRadii_Det<det_id> ring_nr radius_um
                try:
                    det_id = int(key[len("RingRadii_Det"):])
                except ValueError:
                    continue
                if len(args) >= 2:
                    p.RingRadiiPerDet.setdefault(det_id, {})[
                        int(float(args[0]))
                    ] = float(args[1])
                continue
            if key.startswith("EtaCoverage_Det"):
                # EtaCoverage_Det<det_id> ring_nr eta_lo_deg eta_hi_deg
                try:
                    det_id = int(key[len("EtaCoverage_Det"):])
                except ValueError:
                    continue
                if len(args) >= 3:
                    p.EtaCoverage.setdefault(det_id, []).append((
                        int(float(args[0])),
                        float(args[1]),
                        float(args[2]),
                    ))
                continue

            # --- scalar typed keys ---
            if key in _FLOAT_KEYS:
                setattr(p, _FLOAT_KEYS[key], float(args[0]))
                continue
            if key in _INT_KEYS:
                setattr(p, _INT_KEYS[key], int(args[0]))
                continue
            if key in _STR_KEYS:
                setattr(p, _STR_KEYS[key], args[0])
                continue

            warnings.warn(
                f"Unknown key '{key}' in paramstest at line: {line!r}",
                stacklevel=2,
            )

    # IndexerOMP.c:1535-1538 — fold parallel (RingNumbers, RingRadiiUser) into sparse map
    for ring_nr, radius in zip(p.RingNumbers, ring_radii_user):
        p.RingRadii[ring_nr] = radius

    return p
