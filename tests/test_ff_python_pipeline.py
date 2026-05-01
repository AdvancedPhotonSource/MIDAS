#!/usr/bin/env python
"""End-to-end test of the MIDAS pure-Python FF-HEDM pipeline against the
C-binary reference outputs.

Stages exercised (drop-in replacements for the C tools):
    hkl              midas-hkls zarr             vs  GetHKLListZarr
    peak_search      peakfit_torch               vs  PeaksFittingOMPZarrRefactor
    merge_overlaps   midas-merge-peaks           vs  MergeOverlappingPeaksAllZarr
    calc_radius     midas-calc-radius           vs  CalcRadiusAllZarr
    data_transform   midas-fit-setup             vs  FitSetupZarr
    binning          midas-bin-data              vs  SaveBinData
    indexing         python -m midas_index       vs  IndexerOMP

Two modes:
    isolated  Each stage runs against C goldens for the *previous* stage.
              Catches per-stage regressions cleanly.
    chained   Each stage consumes the previous Python stage's output.

Usage::

    python tests/test_ff_python_pipeline.py \
        --data-dir /Users/hsharma/Desktop/analysis/mpe_feb21/ff_recon/LayerNr_1 \
        --scratch-dir /Users/hsharma/Desktop/analysis/mpe_feb21/ff_recon/LayerNr_1_pyrun \
        --device cpu --mode isolated --ncpus 8
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np


STAGES = ["hkl", "peak_search", "merge_overlaps", "calc_radius",
          "data_transform", "binning", "indexing"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rewrite_paramstest(src: Path, dst: Path, *, output_folder: Path,
                        result_folder: Path) -> None:
    """Copy paramstest.txt from src to dst, replacing OutputFolder/ResultFolder.

    Also strips trailing semicolons (the C writer emits ``key value;`` lines;
    midas-index's parser cannot tolerate the trailing ``;`` on numeric values).
    """
    text = src.read_text()
    new_lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith(";"):
            line = line[:-1].rstrip()
        if line.startswith("OutputFolder "):
            new_lines.append(f"OutputFolder {output_folder}")
        elif line.startswith("ResultFolder "):
            new_lines.append(f"ResultFolder {result_folder}")
        else:
            new_lines.append(line)
    dst.write_text("\n".join(new_lines) + "\n")


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _set_device_env(env: dict, device: str, dtype: str) -> dict:
    env = dict(env)
    env["MIDAS_TRANSFORMS_DEVICE"] = device
    env["MIDAS_TRANSFORMS_DTYPE"] = dtype
    env["MIDAS_INDEX_DEVICE"] = device
    env["MIDAS_INDEX_DTYPE"] = dtype
    # OpenMP runtime conflict workaround: peakfit_torch + numpy/scipy can
    # double-link libomp on macOS conda. Without this the peakfit binary
    # aborts with OMP Error #15 / SIGABRT.
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # MPS-specific: peakfit_torch's LM step uses cholesky_solve, which is not
    # yet implemented in PyTorch's MPS backend. Allow the rare op to fall back
    # to CPU; the rest of the kernel still runs on MPS.
    if device == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def _run(cmd: List[str], cwd: Path, env: dict, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f,
                           stderr=subprocess.STDOUT)
    return r.returncode


def _find_pair(rf: Path, pattern: str) -> Optional[Path]:
    """Find first file matching pattern in rf."""
    matches = sorted(rf.glob(pattern))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# comparators
# ---------------------------------------------------------------------------


@dataclass
class CmpResult:
    ok: bool
    rows: int
    detail: str


def _cmp_hkls(new: Path, ref: Path) -> CmpResult:
    """Sort by (h,k,l), compare numeric columns with allclose."""
    cols = ("h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius").split()
    a = np.loadtxt(new, skiprows=1)
    b = np.loadtxt(ref, skiprows=1)
    if a.shape != b.shape:
        return CmpResult(False, len(a), f"shape mismatch {a.shape} vs {b.shape}")
    # Sort by (h,k,l) — deterministic order independent of qsort stability.
    ka = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
    kb = np.lexsort((b[:, 2], b[:, 1], b[:, 0]))
    a = a[ka]
    b = b[kb]
    diffs = []
    for i, name in enumerate(cols):
        if i in (0, 1, 2):  # int cols: h, k, l
            if not np.array_equal(a[:, i], b[:, i]):
                return CmpResult(False, len(a), f"{name} integer rows differ")
        else:
            d = np.max(np.abs(a[:, i] - b[:, i]))
            diffs.append((name, d))
    bad = [(n, d) for n, d in diffs if d > 1e-9]
    if bad:
        return CmpResult(False, len(a),
                         "diffs >1e-9: " + ", ".join(f"{n}={d:.2e}" for n, d in bad))
    return CmpResult(True, len(a),
                     f"max|Δ|={max(d for _, d in diffs):.2e}")


def _cmp_csv(new: Path, ref: Path, *, sort_cols: List[int],
             atol: float = 1e-6, rtol: float = 1e-9) -> CmpResult:
    # MIDAS "csvs" are mostly whitespace-separated; some start with `%`.
    a = np.loadtxt(new, comments="%", skiprows=1, ndmin=2)
    b = np.loadtxt(ref, comments="%", skiprows=1, ndmin=2)
    if a.shape != b.shape:
        return CmpResult(False, len(a), f"shape mismatch {a.shape} vs {b.shape}")
    if a.size == 0:
        return CmpResult(True, 0, "empty")
    ka = np.lexsort([a[:, c] for c in reversed(sort_cols)])
    kb = np.lexsort([b[:, c] for c in reversed(sort_cols)])
    a = a[ka]; b = b[kb]
    if np.allclose(a, b, atol=atol, rtol=rtol):
        return CmpResult(True, len(a),
                         f"max|Δ|={np.max(np.abs(a - b)):.2e}")
    diffs = np.abs(a - b)
    worst = np.argmax(diffs.max(axis=0))
    return CmpResult(False, len(a),
                     f"col {worst} max|Δ|={diffs.max():.2e} (atol={atol})")


def _cmp_text(new: Path, ref: Path,
              *, normalize: Optional[Callable[[str], str]] = None) -> CmpResult:
    a = new.read_text().splitlines()
    b = ref.read_text().splitlines()
    if normalize:
        a = [normalize(s) for s in a]
        b = [normalize(s) for s in b]
    sa, sb = sorted(a), sorted(b)
    if sa == sb:
        return CmpResult(True, len(a), "byte-equal modulo line order")
    extra_a = set(sa) - set(sb)
    extra_b = set(sb) - set(sa)
    return CmpResult(False, len(a),
                     f"+{len(extra_a)} -{len(extra_b)} lines differ "
                     f"(showing first: {next(iter(extra_a), '')!r})")


def _cmp_paramstest(new: Path, ref: Path) -> CmpResult:
    """Compare paramstest.txt: relaxed key-by-key compare. C writes ``key value;``
    while Python omits the trailing semicolon and uses different precision; we
    compare on key presence + numeric closeness, ignoring path-dependent keys."""
    skip_keys = {"OutputFolder", "ResultFolder", "SpotsFileName",
                 "RefinementFileName", "IDsFileName"}

    def _parse(p: Path) -> dict:
        d: dict = {}
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.endswith(";"):
                line = line[:-1].strip()
            parts = line.split(None, 1)
            if not parts:
                continue
            key = parts[0]
            if key in skip_keys:
                continue
            val = parts[1] if len(parts) > 1 else ""
            d.setdefault(key, []).append(val)
        return d

    a = _parse(new)
    b = _parse(ref)
    only_b = sorted(set(b) - set(a))
    only_a = sorted(set(a) - set(b))
    diffs = []
    for k in set(a) & set(b):
        if a[k] != b[k]:
            # Try numeric compare per-line.
            try:
                aa = [list(map(float, x.split())) for x in a[k]]
                bb = [list(map(float, x.split())) for x in b[k]]
                if len(aa) != len(bb):
                    diffs.append(f"{k}(rows {len(aa)}/{len(bb)})")
                    continue
                arr_a = np.array(aa); arr_b = np.array(bb)
                if not np.allclose(arr_a, arr_b, atol=1e-3, rtol=1e-6):
                    diffs.append(f"{k}|Δ|={np.max(np.abs(arr_a-arr_b)):.1e}")
            except ValueError:
                if a[k] != b[k]:
                    diffs.append(f"{k}{a[k]!r}!={b[k]!r}")
    ok = (not only_b) and (not diffs)
    detail = []
    if only_b:
        detail.append(f"missing in py: {','.join(only_b)}")
    if only_a:
        detail.append(f"extra in py: {','.join(only_a)}")
    if diffs:
        detail.append("; ".join(diffs))
    if not detail:
        detail.append("keys+values match")
    return CmpResult(ok, len(b), "; ".join(detail))


def _cmp_bin(new: Path, ref: Path, *, ncols: int, dtype: str = "float64") -> CmpResult:
    a = np.fromfile(str(new), dtype=dtype)
    b = np.fromfile(str(ref), dtype=dtype)
    if a.size != b.size:
        return CmpResult(False, a.size,
                         f"size mismatch {a.size} vs {b.size}")
    if a.size % ncols == 0:
        a = a.reshape(-1, ncols)
        b = b.reshape(-1, ncols)
    if np.array_equal(a, b):
        return CmpResult(True, len(a) if a.ndim > 1 else a.size,
                         "byte-exact")
    diff = np.max(np.abs(a - b))
    rel = diff / max(np.max(np.abs(b)), 1e-30)
    if diff < 1e-9 or rel < 1e-12:
        return CmpResult(True, len(a) if a.ndim > 1 else a.size,
                         f"max|Δ|={diff:.2e} (within fp64 noise)")
    return CmpResult(False, len(a) if a.ndim > 1 else a.size,
                     f"max|Δ|={diff:.2e}")


def _cmp_indexbest(new: Path, ref: Path) -> CmpResult:
    """IndexBest.bin layout (per IndexerOMP.c): 16 doubles per record:
    Confidence, NrSpots, NrSpotsBest (3 ints stored as doubles), then
    9 OM doubles, then 3 pos doubles + last as IA. This is the legacy
    record format. We treat it as an N x 16 float64 matrix and compare
    grain-by-grain after sorting on completeness desc."""
    a = np.fromfile(str(new), dtype="float64")
    b = np.fromfile(str(ref), dtype="float64")
    rec = 16
    if a.size % rec != 0 or b.size % rec != 0:
        return CmpResult(False, 0,
                         f"record-size mismatch (size {a.size}, {b.size})")
    a = a.reshape(-1, rec)
    b = b.reshape(-1, rec)
    if a.shape != b.shape:
        return CmpResult(False, len(a),
                         f"grain-count mismatch {len(a)} vs {len(b)}")
    # Sort by IA ascending (col 15) then completeness desc (col 0) for stable order
    ka = np.lexsort((a[:, 15], -a[:, 0]))
    kb = np.lexsort((b[:, 15], -b[:, 0]))
    a = a[ka]; b = b[kb]
    if np.allclose(a, b, atol=1e-3, rtol=1e-3):
        return CmpResult(True, len(a), f"max|Δ|={np.max(np.abs(a - b)):.2e}")
    diff = np.abs(a - b)
    return CmpResult(False, len(a),
                     f"col {np.argmax(diff.max(axis=0))} max|Δ|={diff.max():.2e}")


# ---------------------------------------------------------------------------
# stage runners
# ---------------------------------------------------------------------------


@dataclass
class StageReport:
    stage: str
    ok: bool
    secs: float
    detail: str


def stage_hkl(workdir: Path, zip_path: Path, golden: Path, env: dict,
              ncpus: int) -> StageReport:
    log = workdir / "log_hkl.txt"
    t0 = time.time()
    rc = _run([sys.executable, "-m", "midas_hkls", "zarr", str(zip_path),
               "--result-folder", str(workdir)], workdir, env, log)
    secs = time.time() - t0
    if rc != 0:
        return StageReport("hkl", False, secs, f"rc={rc} (see {log})")
    new = workdir / "hkls.csv"
    ref = golden / "hkls.csv"
    r = _cmp_hkls(new, ref)
    return StageReport("hkl", r.ok, secs, f"rows={r.rows} {r.detail}")


def stage_peak_search(workdir: Path, zip_path: Path, golden: Path, env: dict,
                      ncpus: int, device: str) -> StageReport:
    """peakfit_torch — write AllPeaks_PS.bin to workdir/Temp/."""
    log = workdir / "log_peak_search.txt"
    if device not in ("cpu", "cuda", "mps"):
        return StageReport("peak_search", False, 0.0,
                           f"device {device!r} not supported by peakfit_torch")
    cmd = ["peakfit_torch", str(zip_path), "0", "1", str(ncpus),
           str(workdir), "1",
           "--device", device, "--dtype", "float64"]
    t0 = time.time()
    rc = _run(cmd, workdir, env, log)
    secs = time.time() - t0
    if rc != 0:
        return StageReport("peak_search", False, secs,
                           f"rc={rc} (see {log})")
    new = workdir / "Temp" / "AllPeaks_PS.bin"
    ref = golden / "Temp" / "AllPeaks_PS.bin"
    if not new.exists():
        return StageReport("peak_search", False, secs,
                           f"missing AllPeaks_PS.bin in {workdir}/Temp")
    # Per-peak record: AllPeaks_PS layout is 9 doubles per peak per the C tool.
    a = np.fromfile(str(new), dtype="float64")
    b = np.fromfile(str(ref), dtype="float64")
    rec = 9
    da = a.size // rec; db = b.size // rec
    rel = abs(da - db) / max(db, 1)
    if rel > 0.01:
        return StageReport("peak_search", False, secs,
                           f"peak count diff {da-db:+d} / {rel:+.2%}")
    return StageReport("peak_search", True, secs,
                       f"npeaks_new={da} npeaks_ref={db} (Δ {da-db:+d})")


def stage_merge(workdir: Path, zip_path: Path, golden: Path, env: dict,
                ncpus: int) -> StageReport:
    log = workdir / "log_merge.txt"
    t0 = time.time()
    rc = _run(["midas-merge-peaks", str(zip_path),
               "--result-folder", str(workdir)], workdir, env, log)
    secs = time.time() - t0
    if rc != 0:
        return StageReport("merge_overlaps", False, secs,
                           f"rc={rc} (see {log})")
    new = _find_pair(workdir, "Result_StartNr_*_EndNr_*.csv")
    ref = _find_pair(golden, "Result_StartNr_*_EndNr_*.csv")
    if not new or not ref:
        return StageReport("merge_overlaps", False, secs,
                           f"missing Result_*.csv (new={new}, ref={ref})")
    # SpotID is col 0 in Result_*.csv
    r = _cmp_csv(new, ref, sort_cols=[0], atol=1e-6, rtol=1e-9)
    return StageReport("merge_overlaps", r.ok, secs,
                       f"rows={r.rows} {r.detail}")


def stage_radius(workdir: Path, zip_path: Path, golden: Path, env: dict,
                 ncpus: int) -> StageReport:
    log = workdir / "log_radius.txt"
    t0 = time.time()
    rc = _run(["midas-calc-radius", str(zip_path),
               "--result-folder", str(workdir)], workdir, env, log)
    secs = time.time() - t0
    if rc != 0:
        return StageReport("calc_radius", False, secs,
                           f"rc={rc} (see {log})")
    new = _find_pair(workdir, "Radius_StartNr_*_EndNr_*.csv")
    ref = _find_pair(golden, "Radius_StartNr_*_EndNr_*.csv")
    if not new or not ref:
        return StageReport("calc_radius", False, secs,
                           f"missing Radius_*.csv")
    r = _cmp_csv(new, ref, sort_cols=[0], atol=1e-6, rtol=1e-9)
    return StageReport("calc_radius", r.ok, secs,
                       f"rows={r.rows} {r.detail}")


def stage_fit_setup(workdir: Path, zip_path: Path, golden: Path, env: dict,
                    ncpus: int) -> StageReport:
    log = workdir / "log_fit_setup.txt"
    t0 = time.time()
    rc = _run(["midas-fit-setup", str(zip_path),
               "--result-folder", str(workdir)], workdir, env, log)
    secs = time.time() - t0
    if rc != 0:
        return StageReport("data_transform", False, secs,
                           f"rc={rc} (see {log})")
    # Rewrite OutputFolder/ResultFolder in the Python-written paramstest.txt
    # to point at workdir so downstream binning/indexing finds the right paths
    # (the zarr-stored ResultFolder is the original Argonne path).
    pt_path = workdir / "paramstest.txt"
    if pt_path.exists():
        out_dir = workdir / "Output"; out_dir.mkdir(exist_ok=True)
        res_dir = workdir / "Results"; res_dir.mkdir(exist_ok=True)
        text = pt_path.read_text().splitlines()
        seen_out = seen_res = False
        new_lines = []
        for line in text:
            if line.startswith("OutputFolder "):
                new_lines.append(f"OutputFolder {out_dir}"); seen_out = True
            elif line.startswith("ResultFolder "):
                new_lines.append(f"ResultFolder {res_dir}"); seen_res = True
            else:
                new_lines.append(line)
        if not seen_out:
            new_lines.append(f"OutputFolder {out_dir}")
        if not seen_res:
            new_lines.append(f"ResultFolder {res_dir}")
        pt_path.write_text("\n".join(new_lines) + "\n")
    # Compare InputAll.csv + paramstest.txt. C writes %.5f so atol=1e-5
    # is the achievable precision floor (not a tolerance bug).
    new_ia = workdir / "InputAll.csv"
    ref_ia = golden / "InputAll.csv"
    r = _cmp_csv(new_ia, ref_ia, sort_cols=[4], atol=1e-5, rtol=1e-9)
    new_pt = workdir / "paramstest.txt"
    ref_pt = golden / "paramstest.txt"
    r2 = _cmp_paramstest(new_pt, ref_pt)
    ok = r.ok and r2.ok
    detail = f"InputAll rows={r.rows} {r.detail}; paramstest {r2.detail}"
    return StageReport("data_transform", ok, secs, detail)


def stage_binning(workdir: Path, zip_path: Path, golden: Path, env: dict,
                  ncpus: int) -> StageReport:
    log = workdir / "log_binning.txt"
    t0 = time.time()
    rc = _run(["midas-bin-data", "--result-folder", str(workdir),
               "--device", env.get("MIDAS_TRANSFORMS_DEVICE", "cpu"),
               "--dtype", env.get("MIDAS_TRANSFORMS_DTYPE", "float64")],
              workdir, env, log)
    secs = time.time() - t0
    if rc != 0:
        return StageReport("binning", False, secs, f"rc={rc} (see {log})")
    parts = []
    okall = True
    # Spots.bin — 9 cols
    rs = _cmp_bin(workdir / "Spots.bin", golden / "Spots.bin", ncols=9)
    parts.append(f"Spots.bin {rs.detail}"); okall &= rs.ok
    # ExtraInfo.bin — 16 cols
    re_ = _cmp_bin(workdir / "ExtraInfo.bin", golden / "ExtraInfo.bin", ncols=16)
    parts.append(f"ExtraInfo {re_.detail}"); okall &= re_.ok
    return StageReport("binning", okall, secs, "; ".join(parts))


def stage_indexing(workdir: Path, zip_path: Path, golden: Path, env: dict,
                   ncpus: int, max_spots_to_index: Optional[int] = None) -> StageReport:
    """midas-index needs paramstest.txt + bin files; outputs Output/IndexBest.bin."""
    output_dir = workdir / "Output"
    output_dir.mkdir(exist_ok=True)
    # Count SpotsToIndex
    sti = workdir / "SpotsToIndex.csv"
    if not sti.exists():
        return StageReport("indexing", False, 0.0,
                           "SpotsToIndex.csv missing — wire bin_data first")
    nlines = sum(1 for _ in sti.open())
    if max_spots_to_index is not None and max_spots_to_index < nlines:
        nlines = max_spots_to_index
    log = workdir / "log_indexing.txt"
    t0 = time.time()
    device = env.get("MIDAS_INDEX_DEVICE", "cpu")
    dtype = env.get("MIDAS_INDEX_DTYPE", "float64")
    cmd = [sys.executable, "-m", "midas_index", "paramstest.txt",
           "0", "1", str(nlines), str(ncpus),
           "--device", device, "--dtype", dtype]
    grp = env.get("MIDAS_INDEX_GROUP_SIZE")
    if grp:
        cmd.extend(["--group-size", grp])
    rc = _run(cmd, workdir, env, log)
    secs = time.time() - t0
    if rc != 0:
        return StageReport("indexing", False, secs, f"rc={rc} (see {log})")
    new = workdir / "Output" / "IndexBest.bin"
    ref = golden / "Output" / "IndexBest.bin"
    if not new.exists():
        return StageReport("indexing", False, secs,
                           f"IndexBest.bin missing in {output_dir}")
    if max_spots_to_index is not None:
        a = np.fromfile(str(new), dtype="float64")
        rec = 16
        nrec = a.size // rec
        return StageReport("indexing", True, secs,
                           f"partial-N={max_spots_to_index} records={nrec} "
                           f"(comparison skipped — ran on subset)")
    r = _cmp_indexbest(new, ref)
    return StageReport("indexing", r.ok, secs, f"records={r.rows} {r.detail}")


# ---------------------------------------------------------------------------
# isolation harness — copies just-enough C goldens into a fresh stage dir
# ---------------------------------------------------------------------------


def stage_inputs_isolated(stage: str, stage_dir: Path, golden: Path,
                          zip_path: Path) -> None:
    """Populate stage_dir with the C goldens that this stage needs as inputs."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    # zip is always available
    _link(zip_path, stage_dir / zip_path.name)

    if stage == "hkl":
        return  # no extra inputs

    # All non-hkl stages eventually want hkls.csv
    if (golden / "hkls.csv").exists():
        shutil.copy2(golden / "hkls.csv", stage_dir / "hkls.csv")

    if stage == "peak_search":
        return

    if stage in ("merge_overlaps",):
        # peakfit output → Temp/AllPeaks_PS.bin
        (stage_dir / "Temp").mkdir(exist_ok=True)
        for fn in ("AllPeaks_PS.bin", "AllPeaks_PX.bin"):
            src = golden / "Temp" / fn
            if src.exists():
                shutil.copy2(src, stage_dir / "Temp" / fn)
        return

    if stage in ("calc_radius",):
        for p in golden.glob("Result_StartNr_*_EndNr_*.csv"):
            shutil.copy2(p, stage_dir / p.name)
        return

    if stage in ("data_transform",):
        for pat in ("Result_StartNr_*_EndNr_*.csv",
                    "Radius_StartNr_*_EndNr_*.csv"):
            for p in golden.glob(pat):
                shutil.copy2(p, stage_dir / p.name)
        return

    if stage in ("binning",):
        for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv",
                   "InputAllNoHeader.csv"):
            if (golden / fn).exists():
                shutil.copy2(golden / fn, stage_dir / fn)
        # Rewrite paramstest.txt with local Output/Result paths
        out_dir = stage_dir / "Output"
        res_dir = stage_dir / "Results"
        out_dir.mkdir(exist_ok=True); res_dir.mkdir(exist_ok=True)
        _rewrite_paramstest(golden / "paramstest.txt",
                            stage_dir / "paramstest.txt",
                            output_folder=out_dir, result_folder=res_dir)
        return

    if stage in ("indexing",):
        # Indexing reads bin files + paramstest + SpotsToIndex/IDsHash/IDRings
        for fn in ("Spots.bin", "ExtraInfo.bin", "Data.bin", "nData.bin",
                   "SpotsToIndex.csv", "IDsHash.csv", "IDRings.csv",
                   "InputAll.csv", "InputAllExtraInfoFittingAll.csv"):
            if (golden / fn).exists():
                shutil.copy2(golden / fn, stage_dir / fn)
        out_dir = stage_dir / "Output"
        res_dir = stage_dir / "Results"
        out_dir.mkdir(exist_ok=True); res_dir.mkdir(exist_ok=True)
        _rewrite_paramstest(golden / "paramstest.txt",
                            stage_dir / "paramstest.txt",
                            output_folder=out_dir, result_folder=res_dir)
        return


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


STAGE_RUNNERS = {
    "hkl": stage_hkl,
    "peak_search": stage_peak_search,
    "merge_overlaps": stage_merge,
    "calc_radius": stage_radius,
    "data_transform": stage_fit_setup,
    "binning": stage_binning,
    "indexing": stage_indexing,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Source LayerNr_1 dir with C goldens + zip")
    ap.add_argument("--scratch-dir", required=True, type=Path,
                    help="Scratch dir (will create iso_*/ and chained/ inside)")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--mode", choices=["isolated", "chained"],
                    default="isolated")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help="Comma-separated list of stages to run")
    ap.add_argument("--ncpus", type=int, default=8)
    ap.add_argument("--max-spots-to-index", type=int, default=None,
                    help="Cap n_spots_to_index for the indexer stage (memory work-around).")
    args = ap.parse_args()

    golden: Path = args.data_dir.resolve()
    scratch: Path = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    # Find the zip
    zips = sorted(golden.glob("*.MIDAS.zip"))
    if not zips:
        print(f"ERROR: no .MIDAS.zip in {golden}", file=sys.stderr)
        return 2
    zip_path = zips[0]

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in requested:
        if s not in STAGES:
            print(f"ERROR: unknown stage {s!r}", file=sys.stderr)
            return 2

    env = _set_device_env(os.environ.copy(), args.device, args.dtype)

    print(f"=== FF-HEDM Python pipeline test ===")
    print(f"data-dir   : {golden}")
    print(f"scratch    : {scratch}")
    print(f"mode       : {args.mode}")
    print(f"device     : {args.device}  dtype: {args.dtype}  ncpus: {args.ncpus}")
    print(f"stages     : {','.join(requested)}")
    print()

    reports: List[StageReport] = []
    chained_dir = scratch / "chained"
    if args.mode == "chained":
        chained_dir.mkdir(exist_ok=True)
        # In chained mode the zip lives in chained_dir
        _link(zip_path, chained_dir / zip_path.name)

    for s in requested:
        if args.mode == "isolated":
            stage_dir = scratch / f"iso_{s}"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_inputs_isolated(s, stage_dir, golden, zip_path)
        else:
            stage_dir = chained_dir

        runner = STAGE_RUNNERS[s]
        if s == "peak_search":
            rep = runner(stage_dir, stage_dir / zip_path.name, golden, env,
                         args.ncpus, args.device)
        elif s == "indexing":
            rep = runner(stage_dir, stage_dir / zip_path.name, golden, env,
                         args.ncpus, max_spots_to_index=args.max_spots_to_index)
        else:
            rep = runner(stage_dir, stage_dir / zip_path.name, golden, env,
                         args.ncpus)
        reports.append(rep)
        flag = "PASS" if rep.ok else "FAIL"
        print(f"[ {requested.index(s)+1:>1}/{len(requested)} ] "
              f"{rep.stage:<14}  {flag}  {rep.secs:6.2f} s   {rep.detail}")
        sys.stdout.flush()
        # Note: in chained mode we continue past failures so the report covers
        # the full pipeline. Downstream stages may cascade-fail; that is
        # informative, not a bug in the driver.

    print()
    n_ok = sum(1 for r in reports if r.ok)
    total_t = sum(r.secs for r in reports)
    print(f"Summary: {n_ok}/{len(reports)} stages passed, "
          f"wall time {total_t:.1f} s")
    return 0 if n_ok == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
