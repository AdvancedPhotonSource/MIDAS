"""Stage wrappers for the FF-HEDM pipeline.

Each module exposes a single ``run(...)`` function that takes a
``StageContext`` and returns a typed ``StageResult``.
"""

from ._base import StageContext, run_subprocess
from . import (
    zip_convert,
    hkl,
    peakfit,
    merge_overlaps,
    calc_radius,
    transforms,
    cross_det_merge,
    binning,
    index as index_stage,
    refine,
    process_grains,
    consolidation,
)

__all__ = [
    "StageContext",
    "run_subprocess",
    "zip_convert",
    "hkl",
    "peakfit",
    "merge_overlaps",
    "calc_radius",
    "transforms",
    "cross_det_merge",
    "binning",
    "index_stage",
    "refine",
    "process_grains",
    "consolidation",
]
