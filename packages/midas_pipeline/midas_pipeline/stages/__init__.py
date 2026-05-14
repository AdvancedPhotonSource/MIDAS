"""Pipeline stages.

Each stage exposes ``run(ctx: StageContext) -> StageResult``. P1 ships
thin shells that mark themselves ``skipped=True`` and emit a warning.
Real implementations land in P2–P8 — see the relevant phase in the plan
file for the ownership boundary.

The thin shells exist so:
- ``Pipeline._run_layer`` has something to call per stage.
- Parallel-stream developers (P2–P8) have a clear single file to fill in.
- Resume / provenance integration is testable from day one.
"""

from . import (
    binning,
    calc_radius,
    consolidation,
    cross_det_merge,
    em_refine,
    find_grains_stage,
    fuse,
    global_powder,
    hkl,
    indexing,
    merge_overlaps,
    merge_scans,
    peakfit,
    potts,
    process_grains,
    reconstruct,
    refinement,
    sinogen,
    transforms,
    zip_convert,
)

__all__ = [
    "binning",
    "calc_radius",
    "consolidation",
    "cross_det_merge",
    "em_refine",
    "find_grains_stage",
    "fuse",
    "global_powder",
    "hkl",
    "indexing",
    "merge_overlaps",
    "merge_scans",
    "peakfit",
    "potts",
    "process_grains",
    "reconstruct",
    "refinement",
    "sinogen",
    "transforms",
    "zip_convert",
]
