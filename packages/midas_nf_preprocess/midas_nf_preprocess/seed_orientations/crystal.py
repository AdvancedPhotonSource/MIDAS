"""Crystal-system <-> space-group <-> Laue-group mapping.

Mirrors:

  - ``utils/extract_seed_orientations.sg_to_lookup_type`` (the SG -> 12 MIDAS
    symmetry-type bucket, used by the cached lookup files).
  - ``NF_HEDM/src/GenerateSeedLookupTables.c`` SymType table (representative
    space group per bucket).

A "lookup type" is the canonical short name MIDAS uses for one of its 12
symmetry buckets; a "crystal system" is the friendly name (cubic, hexagonal,
...). Both map to a representative integer space group that
``midas_stress.fundamental_zone`` understands.
"""

from __future__ import annotations

from typing import Iterable

# 12 MIDAS lookup types in the order GenerateSeedLookupTables emits them.
LOOKUP_TYPES = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal_low",
    "tetragonal_high",
    "trigonal_low",
    "trigonal_type1",
    "trigonal_type2",
    "hexagonal_low",
    "hexagonal_high",
    "cubic_low",
    "cubic_high",
)

# Representative space group per lookup type (matches GenerateSeedLookupTables.c
# SymType table). Either of the SGs in a bucket would give identical symmetry
# operators; the canonical choice is the highest-symmetry SG in the bucket.
REPRESENTATIVE_SG = {
    "triclinic":       1,
    "monoclinic":      15,
    "orthorhombic":    74,
    "tetragonal_low":  88,
    "tetragonal_high": 142,
    "trigonal_low":    148,
    "trigonal_type1":  155,
    "trigonal_type2":  149,
    "hexagonal_low":   176,
    "hexagonal_high":  194,
    "cubic_low":       206,
    "cubic_high":      225,
}

# Number of proper rotations in each MIDAS Laue group. Used by the from-scratch
# resolution heuristic (more symmetries -> smaller FZ -> need fewer master
# samples to fill it at a given resolution).
SYM_ORDER = {
    "triclinic":       2,
    "monoclinic":      4,
    "orthorhombic":    8,
    "tetragonal_low":  8,
    "tetragonal_high": 16,
    "trigonal_low":    6,
    "trigonal_type1":  12,
    "trigonal_type2":  12,
    "hexagonal_low":   12,
    "hexagonal_high":  24,
    "cubic_low":       24,
    "cubic_high":      48,
}

# Friendly crystal-system names -> the highest-symmetry MIDAS bucket in that
# system. Pass as ``crystal_system="cubic"``.
CRYSTAL_SYSTEM_TO_LOOKUP = {
    "triclinic":     "triclinic",
    "monoclinic":    "monoclinic",
    "orthorhombic":  "orthorhombic",
    "tetragonal":    "tetragonal_high",
    "trigonal":      "trigonal_type1",
    "hexagonal":     "hexagonal_high",
    "cubic":         "cubic_high",
}

# Trigonal type-2 SGs (matches utils/extract_seed_orientations._TrigType2SGs).
_TRIG_TYPE2_SGS = frozenset({149, 151, 153, 157, 159, 162, 163})


def space_group_to_lookup_type(sg: int) -> str:
    """Map a space group (1-230) to its MIDAS lookup-type bucket.

    Direct port of ``utils/extract_seed_orientations.sg_to_lookup_type``.
    """
    if not (1 <= sg <= 230):
        raise ValueError(f"space_group must be in [1, 230], got {sg}")
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 88:
        return "tetragonal_low"
    if sg <= 142:
        return "tetragonal_high"
    if sg <= 148:
        return "trigonal_low"
    if sg <= 167:
        return "trigonal_type2" if sg in _TRIG_TYPE2_SGS else "trigonal_type1"
    if sg <= 176:
        return "hexagonal_low"
    if sg <= 194:
        return "hexagonal_high"
    if sg <= 206:
        return "cubic_low"
    return "cubic_high"


def crystal_system_to_space_group(crystal_system: str) -> int:
    """Map a friendly crystal-system name to a representative space group.

    >>> crystal_system_to_space_group("cubic")
    225
    """
    cs = crystal_system.lower()
    try:
        return REPRESENTATIVE_SG[CRYSTAL_SYSTEM_TO_LOOKUP[cs]]
    except KeyError:
        raise ValueError(
            f"Unknown crystal_system '{crystal_system}'; "
            f"expected one of {sorted(CRYSTAL_SYSTEM_TO_LOOKUP)}"
        )


def space_group_to_sym_order(sg: int) -> int:
    """Number of proper rotations in the Laue group for a given SG."""
    return SYM_ORDER[space_group_to_lookup_type(sg)]
