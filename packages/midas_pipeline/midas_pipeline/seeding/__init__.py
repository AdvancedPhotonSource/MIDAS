"""Seeding modes for the per-voxel indexer (plan §9).

Three modes:

- ``unseeded``  : brute-force orientation search per voxel (current
                  default). No external seed file; the indexer's
                  orientation grid drives candidate enumeration.
- ``ff``        : caller supplies a ``GrainsFile`` from a prior FF run.
                  ``midas-index`` reads ``UniqueOrientations.csv`` and
                  only checks orientations near those seeds.
- ``merged-ff`` : merge all per-scan spot lists into a single FF-style
                  spot file, run the FF indexer on it, then hand the
                  resulting grains off as seeds for the per-voxel
                  scanning indexer. Higher recall than ``ff`` on
                  small grains that a single-FF run misses; faster
                  than ``unseeded``.

The merged-FF path has four sub-stages (plan §9c):

  A. align     → per-scan rotation-axis alignment (ring-center fit).
  B. merge_all → :func:`midas_pipeline.stages.merge_scans` with
                 ``n_merges == n_scans`` → one InputAllExtraInfoFittingAll.csv.
  C. ff_index  → ``midas-index`` (FF mode, no scan filter) on the
                 merged spot file → ``Grains.csv``.
  D. handoff   → ``Grains.csv`` → ``UniqueOrientations.csv`` for the
                 per-voxel pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SeedingResult:
    """Per-mode summary returned by the top-level orchestrator."""

    mode: str                              # "unseeded" | "ff" | "merged-ff"
    unique_orientations_csv: Optional[Path] = None
    align_diagnostics_csv: Optional[Path] = None    # only set for "merged-ff"
    merge_summary: Dict = field(default_factory=dict)
    ff_index_summary: Dict = field(default_factory=dict)
    n_seed_grains: int = 0


# Public entry points (each sub-module exposes a small, testable
# function — see the individual modules for details).
from .handoff import grains_csv_to_unique_orientations
from .merge_all import merge_all_scans

__all__ = [
    "SeedingResult",
    "grains_csv_to_unique_orientations",
    "merge_all_scans",
]
