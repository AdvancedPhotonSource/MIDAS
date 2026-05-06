"""Per-stage and per-layer result containers.

Each stage's ``run()`` returns a dataclass with the *paths* of files
it produced + small summary metrics. The Pipeline holds the latest
result per stage and writes them to ``midas_state.h5`` for resume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    """Common fields across stages."""

    stage_name: str
    started_at: float
    finished_at: float
    duration_s: float
    inputs: Dict[str, str] = field(default_factory=dict)        # path → sha256 or marker
    outputs: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False                                       # set True if resume hit


@dataclass
class HKLResult(StageResult):
    hkls_csv: str = ""


@dataclass
class PerDetPeakFitResult:
    det_id: int
    n_peaks: int
    zip_path: str


@dataclass
class PeakFitResult(StageResult):
    per_detector: List[PerDetPeakFitResult] = field(default_factory=list)


@dataclass
class MergeOverlapsResult(StageResult):
    n_peaks_after_merge: int = 0


@dataclass
class CalcRadiusResult(StageResult):
    n_spots: int = 0


@dataclass
class TransformsResult(StageResult):
    paramstest_path: str = ""


@dataclass
class CrossDetMergeResult(StageResult):
    n_total_spots: int = 0
    n_per_detector: List[int] = field(default_factory=list)
    spots_bin: str = ""
    spots_det_bin: str = ""        # the side-car (det_id per spot)


@dataclass
class BinningResult(StageResult):
    n_bins: int = 0


@dataclass
class IndexResult(StageResult):
    index_best_bin: str = ""
    n_seeds_attempted: int = 0
    n_seeds_indexed: int = 0


@dataclass
class RefineResult(StageResult):
    orient_pos_fit_bin: str = ""
    n_grains_refined: int = 0


@dataclass
class ProcessGrainsResult(StageResult):
    grains_csv: str = ""
    n_grains: int = 0


@dataclass
class LayerResult:
    """All-stage roll-up for one layer."""

    layer_nr: int
    layer_dir: str
    zip_convert: Optional[StageResult] = None
    hkl: Optional[HKLResult] = None
    peakfit: Optional[PeakFitResult] = None
    merge_overlaps: Optional[MergeOverlapsResult] = None
    calc_radius: Optional[CalcRadiusResult] = None
    transforms: Optional[TransformsResult] = None
    cross_det_merge: Optional[CrossDetMergeResult] = None
    global_powder: Optional[StageResult] = None
    binning: Optional[BinningResult] = None
    indexing: Optional[IndexResult] = None
    refinement: Optional[RefineResult] = None
    process_grains: Optional[ProcessGrainsResult] = None
    consolidation: Optional[StageResult] = None

    @property
    def n_grains(self) -> int:
        return self.process_grains.n_grains if self.process_grains else 0

    @property
    def grains_csv(self) -> str:
        return self.process_grains.grains_csv if self.process_grains else ""

    def grains_df(self):  # pragma: no cover — needs pandas at call time
        """Return Grains.csv as a DataFrame, dropping the ``%``-prefixed header lines."""
        import pandas as pd
        if not self.grains_csv:
            return pd.DataFrame()
        path = Path(self.grains_csv)
        if not path.exists():
            return pd.DataFrame()
        # Find the header line (starts with "%GrainID")
        header_idx = None
        with path.open() as fp:
            lines = fp.readlines()
        for i, line in enumerate(lines):
            if line.startswith("%GrainID"):
                header_idx = i
                break
        if header_idx is None:
            return pd.DataFrame()
        cols = lines[header_idx][1:].split()
        rows = [
            line.split() for line in lines[header_idx + 1:]
            if line.strip() and not line.startswith("%")
        ]
        df = pd.DataFrame(rows, columns=cols)
        for c in cols:
            if c == "GrainID":
                df[c] = df[c].astype(int)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def all_stage_results(self) -> List[StageResult]:
        out: list[StageResult] = []
        for f in (self.zip_convert, self.hkl, self.peakfit, self.merge_overlaps,
                  self.calc_radius, self.transforms, self.cross_det_merge,
                  self.global_powder, self.binning,
                  self.indexing, self.refinement, self.process_grains,
                  self.consolidation):
            if f is not None:
                out.append(f)
        return out

    def total_duration_s(self) -> float:
        return sum(r.duration_s for r in self.all_stage_results())
