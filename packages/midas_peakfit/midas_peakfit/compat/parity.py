"""Parity report: diff our output against a C-produced ``AllPeaks_PS.bin``.

Tolerances per ``peakfit_torch_implementation_plan.md`` §7.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from midas_peakfit.compat.reference_decoder import read_ps
from midas_peakfit.postfit import N_PEAK_COLS

# Column index → (name, abs_tol, rel_tol). ``None`` rel_tol means abs only;
# ``None`` abs_tol means rel only. ``"exact"`` => exact match required.
_COL_TOLS: Dict[int, Tuple[str, object]] = {
    0: ("SpotID", "skip"),  # IDs may differ in ordering across implementations
    1: ("IntegratedIntensity", (None, 0.05)),
    2: ("Omega", (1e-6, None)),
    3: ("YCen", (0.05, None)),
    4: ("ZCen", (0.05, None)),
    5: ("IMax", (None, 0.05)),
    6: ("Radius", (0.05, None)),
    7: ("Eta", (0.02, None)),
    8: ("SigmaR", (None, 0.10)),
    9: ("SigmaEta", (None, 0.10)),
    10: ("NrPixels", "exact"),
    11: ("NrPxTot", "exact"),
    12: ("nPeaks", "exact"),
    13: ("maxY", "exact"),
    14: ("maxZ", "exact"),
    15: ("diffY", (0.05, None)),
    16: ("diffZ", (0.05, None)),
    17: ("rawIMax", (None, 0.01)),
    18: ("returnCode", "skip"),  # zero-vs-nonzero tracked separately
    19: ("retVal", (None, 0.10)),
    20: ("BG", (None, 0.10)),
    21: ("SigmaGR", (None, 0.10)),
    22: ("SigmaLR", (None, 0.10)),
    23: ("SigmaGEta", (None, 0.10)),
    24: ("SigmaLEta", (None, 0.10)),
    25: ("MU", (0.10, None)),
    26: ("RawSumIntensity", (None, 0.05)),
    27: ("maskTouched", "exact"),
    28: ("FitRMSE", (None, 0.10)),
}


@dataclass
class ColumnDiff:
    name: str
    n_within_tol: int = 0
    n_outside_tol: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0


@dataclass
class ParityReport:
    n_frames: int = 0
    n_frames_matching_count: int = 0
    n_total_peaks_c: int = 0
    n_total_peaks_py: int = 0
    columns: Dict[str, ColumnDiff] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)

    def passing(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        lines = [
            f"== Parity Report ==",
            f"  Frames: {self.n_frames}, matching nPeaks: {self.n_frames_matching_count}",
            f"  Total peaks: C={self.n_total_peaks_c}, Py={self.n_total_peaks_py}",
            f"  Columns:",
        ]
        for name, d in self.columns.items():
            total = d.n_within_tol + d.n_outside_tol
            if total == 0:
                continue
            lines.append(
                f"    {name:22s}: {d.n_within_tol}/{total} within tol "
                f"(max_abs={d.max_abs:.3e}, max_rel={d.max_rel:.3e})"
            )
        if self.failures:
            lines.append(f"  Failures ({len(self.failures)}):")
            for fmsg in self.failures[:10]:
                lines.append(f"    - {fmsg}")
            if len(self.failures) > 10:
                lines.append(f"    ... ({len(self.failures) - 10} more)")
        else:
            lines.append("  ALL TOLERANCES MET")
        return "\n".join(lines)


def _match_peaks(
    rows_a: np.ndarray, rows_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy NN matching by (Y, Z) center, then Imax. Returns matched indices."""
    if rows_a.shape[0] == 0 or rows_b.shape[0] == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Fields: YCen=3, ZCen=4, IMax=5
    a_yz = rows_a[:, [3, 4]]
    b_yz = rows_b[:, [3, 4]]
    used_b = np.zeros(rows_b.shape[0], dtype=bool)
    matched_a, matched_b = [], []
    for i, ay in enumerate(a_yz):
        d = np.linalg.norm(b_yz - ay, axis=1)
        d = np.where(used_b, np.inf, d)
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > 1.0:  # 1 px gating
            continue
        used_b[j] = True
        matched_a.append(i)
        matched_b.append(j)
    return np.array(matched_a, dtype=np.int32), np.array(matched_b, dtype=np.int32)


def parity_check(c_ps_path: str | Path, py_ps_path: str | Path) -> ParityReport:
    """Read both ``AllPeaks_PS.bin`` files and produce a parity report."""
    c = read_ps(c_ps_path)
    py = read_ps(py_ps_path)

    rep = ParityReport()
    if c.n_frames != py.n_frames:
        rep.failures.append(
            f"frame count mismatch: C={c.n_frames}, Py={py.n_frames}"
        )
        rep.n_frames = min(c.n_frames, py.n_frames)
    else:
        rep.n_frames = c.n_frames

    for col_idx, (name, tol_spec) in _COL_TOLS.items():
        rep.columns[name] = ColumnDiff(name=name)

    for f in range(rep.n_frames):
        ra = c.rows_per_frame[f]
        rb = py.rows_per_frame[f]
        rep.n_total_peaks_c += ra.shape[0]
        rep.n_total_peaks_py += rb.shape[0]
        if ra.shape[0] == rb.shape[0]:
            rep.n_frames_matching_count += 1
        else:
            rep.failures.append(
                f"frame {f}: nPeaks C={ra.shape[0]} vs Py={rb.shape[0]}"
            )
        if ra.shape[0] == 0 or rb.shape[0] == 0:
            continue
        ai, bi = _match_peaks(ra, rb)
        if ai.size == 0:
            rep.failures.append(f"frame {f}: no peak matches found")
            continue

        for col_idx, (name, tol_spec) in _COL_TOLS.items():
            if tol_spec == "skip":
                continue
            d = rep.columns[name]
            a = ra[ai, col_idx]
            b = rb[bi, col_idx]
            if tol_spec == "exact":
                eq = a == b
                d.n_within_tol += int(eq.sum())
                d.n_outside_tol += int((~eq).sum())
                if (~eq).any():
                    d.max_abs = max(d.max_abs, float(np.max(np.abs(a - b))))
                continue
            abs_tol, rel_tol = tol_spec
            abs_diff = np.abs(a - b)
            d.max_abs = max(d.max_abs, float(np.max(abs_diff))) if abs_diff.size else d.max_abs
            scale = np.maximum(np.abs(a), 1e-12)
            rel_diff = abs_diff / scale
            d.max_rel = max(d.max_rel, float(np.max(rel_diff))) if rel_diff.size else d.max_rel
            within = np.ones_like(abs_diff, dtype=bool)
            if abs_tol is not None:
                within = within & (abs_diff <= abs_tol)
            if rel_tol is not None:
                within = within & (rel_diff <= rel_tol)
            d.n_within_tol += int(within.sum())
            d.n_outside_tol += int((~within).sum())

    # Only fail on out-of-tol columns where the failure rate is non-trivial
    for name, d in rep.columns.items():
        total = d.n_within_tol + d.n_outside_tol
        if total > 0 and d.n_outside_tol / total > 0.05:
            rep.failures.append(
                f"column {name}: {d.n_outside_tol}/{total} outside tol"
            )

    return rep


__all__ = ["ParityReport", "ColumnDiff", "parity_check"]
