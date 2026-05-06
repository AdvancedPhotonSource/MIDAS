"""End-to-end parity test: ``fit_orientation_run`` vs the C
``FitOrientationOMP`` binary on the bundled NF-HEDM example.

Pipeline
--------
1. Replicate ``tests/test_nf_hedm.py``'s setup (workspace, simulate,
   patch the paramfile) and invoke ``nf_MIDAS.py``. This produces:
     - ``SpotsInfo.bin``, ``OrientMat.bin``, ``Key.bin``,
       ``DiffractionSpots.bin`` (inputs the screen + fit consume),
     - ``MicFileBinary`` (the C ``FitOrientationOMP`` output we treat
       as reference).
2. Move the C ``MicFileBinary`` aside as the reference.
3. Call :func:`midas_nf_fitorientation.fit_orientation_run` with the
   same parameter file, ``blockNr=0 nBlocks=1``. This overwrites the
   in-place ``MicFileBinary`` with our Python-produced version.
4. Read both ``MicFileBinary`` files (11 doubles per voxel, the layout
   defined in :class:`midas_nf_fitorientation.output.MicRecord`) and
   compare the per-voxel best Euler angles via the symmetry-aware
   misorientation from :mod:`midas_stress.orientation`.

Pass criteria — the new package replaces NM with L-BFGS over a soft
Gaussian-splat surrogate, so it converges to a slightly different
point than the C reference even on textbook voxels. Empirically on a
30-voxel stratified sample of the bundled Au example:

    - Median miso ≈ 0.12°, max ≈ 0.27°
    - 93% of voxels < 0.25°, 100% < 0.5°

So the gates are:
    - Median miso < 0.5°
    - 90% of voxels < 0.5°

This test is **slow** (≈60–120 s) and depends on the full MIDAS C
build being present plus the bundled ``NF_HEDM/Example`` data, so it
is gated behind an env var:

    MIDAS_RUN_INTEGRATION=1 pytest tests/integration/

Run it directly as a script too::

    python tests/integration/test_vs_c_fit_orientation.py

The script form prints the full diagnostic table and is useful for
debugging.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


MIDAS_HOME = Path(__file__).resolve().parents[4]   # …/MIDAS
EXAMPLE_DIR = MIDAS_HOME / "NF_HEDM" / "Example"
SIM_DIR = EXAMPLE_DIR / "sim"
PARAM_FN = EXAMPLE_DIR / "ps_au.txt"
MIC_FN = EXAMPLE_DIR / "Au_txt.mic"

# Pass thresholds (see module docstring).
MAX_MEDIAN_MISO_DEG = 0.5
MIN_FRAC_BELOW_HALF_DEG = 0.90

INTEGRATION = os.environ.get("MIDAS_RUN_INTEGRATION") == "1"


# ---------------------------------------------------------------------------
#  Setup: simulate, patch paramfile, run nf_MIDAS.py
# ---------------------------------------------------------------------------

def _simulate(local_param: Path, local_mic: Path, n_cpus: int, work_dir: Path) -> None:
    """Call the bundled ``simulateNF`` C executable.

    Mirrors ``tests.test_nf_hedm.run_simulation``.
    """
    bin_path = MIDAS_HOME / "NF_HEDM" / "bin" / "simulateNF"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"simulateNF not built at {bin_path}; run the C build first"
        )
    cmd = [
        str(bin_path), str(local_param), str(local_mic),
        "recon", str(n_cpus),
    ]
    print("Step 1 — simulateNF:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(work_dir))
    if res.returncode != 0:
        raise RuntimeError(f"simulateNF failed (exit {res.returncode})")
    if not (work_dir / "SpotsInfo.bin").exists():
        raise RuntimeError("simulateNF did not produce SpotsInfo.bin")


def _patch_paramfile(local_param: Path, work_dir: Path) -> Path:
    """Add the workflow-required keys (DataDirectory, GrainsFile,
    SeedOrientations, PrecomputedSpotsInfo) and write to
    ``test_<name>.txt`` next to ``local_param``. Mirrors
    ``tests.test_nf_hedm.prepare_param_file``.
    """
    grains = work_dir / "grs.csv"
    seeds = work_dir / "seedOrientations.txt"
    skip_keys = {"DataDirectory", "SeedOrientations", "GrainsFile"}
    out = work_dir / f"test_{local_param.name}"
    with open(local_param) as fin, open(out, "w") as fout:
        for line in fin:
            stripped = line.split("#", 1)[0].strip()
            tokens = stripped.split() if stripped else []
            if tokens and tokens[0] in skip_keys:
                continue
            fout.write(line)
        fout.write(f"\nDataDirectory {work_dir}\n")
        fout.write(f"GrainsFile {grains}\n")
        fout.write(f"SeedOrientations {seeds}\n")
        fout.write(f"PrecomputedSpotsInfo 1\n")
    return out


def _run_nf_midas(test_param: Path, n_cpus: int, work_dir: Path) -> None:
    """Run ``nf_MIDAS.py`` to completion. This drives all upstream
    stages and finally calls ``FitOrientationOMP`` to write
    ``MicFileBinary``.
    """
    workflow = MIDAS_HOME / "NF_HEDM" / "workflows" / "nf_MIDAS.py"
    if not workflow.exists():
        raise FileNotFoundError(workflow)
    cmd = [
        sys.executable, str(workflow),
        "-paramFN", test_param.name,
        "-nCPUs", str(n_cpus),
        "-ffSeedOrientations", "1",
        "-doImageProcessing", "0",
        "-refineParameters", "0",
        "-multiGridPoints", "0",
    ]
    print("Step 2 — nf_MIDAS.py:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(work_dir))
    if res.returncode != 0:
        raise RuntimeError(f"nf_MIDAS.py failed (exit {res.returncode})")


def _setup_workspace() -> Tuple[Path, Path]:
    """Build the ``sim`` workspace from scratch and return
    ``(work_dir, test_paramfile)``.
    """
    grains_src = EXAMPLE_DIR / "grs.csv"
    if not grains_src.exists():
        raise FileNotFoundError(f"missing grains seed {grains_src}")
    if SIM_DIR.exists():
        shutil.rmtree(SIM_DIR)
    SIM_DIR.mkdir(parents=True)
    shutil.copy2(PARAM_FN, SIM_DIR / PARAM_FN.name)
    shutil.copy2(MIC_FN, SIM_DIR / MIC_FN.name)
    shutil.copy2(grains_src, SIM_DIR / "grs.csv")
    return SIM_DIR, SIM_DIR / PARAM_FN.name


def _resolve_mic_binary(test_param: Path, work_dir: Path) -> Path:
    """Look up the ``MicFileBinary`` key from the paramfile and
    resolve to a path in ``work_dir``.
    """
    name = None
    with open(test_param) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 2 and tokens[0] == "MicFileBinary":
                name = tokens[1]
                break
    if name is None:
        raise ValueError(f"MicFileBinary missing in {test_param}")
    return work_dir / Path(name).name


def _read_mic_binary(path: Path, n_voxels: int) -> np.ndarray:
    """Decode an 11-doubles-per-voxel ``MicFileBinary`` into ``(N, 11)``.

    The columns match :class:`midas_nf_fitorientation.output.MicRecord`:
    ``[bestRowNr, nWinners, fitTime, xs, ys, gridSize, ud,
     eulA, eulB, eulC, fracOverlap]``.
    """
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size != 11 * n_voxels:
        raise ValueError(
            f"{path}: expected {11 * n_voxels} doubles "
            f"({n_voxels} voxels × 11), got {raw.size}"
        )
    return raw.reshape(n_voxels, 11)


# ---------------------------------------------------------------------------
#  Comparison
# ---------------------------------------------------------------------------

@dataclass
class ParityReport:
    n_total: int
    n_compared: int
    n_below_quarter_deg: int
    n_below_half_deg: int
    miso_mean_deg: float
    miso_median_deg: float
    miso_max_deg: float
    frac_diff_mean: float
    frac_diff_max: float


def _compare(c_mic: np.ndarray, py_mic: np.ndarray, space_group: int) -> ParityReport:
    """Compare per-voxel Eulers via symmetry-aware misorientation.

    Voxels with no winners in either run (``nWinners == 0``) are
    excluded — they have no meaningful Euler in the C output and so
    cannot disagree with our Python output regardless of correctness.
    """
    from midas_stress.orientation import misorientation

    if c_mic.shape != py_mic.shape:
        raise ValueError(
            f"shape mismatch: C {c_mic.shape} vs Py {py_mic.shape}"
        )

    n_winners_c = c_mic[:, 1].astype(np.int64)
    n_winners_py = py_mic[:, 1].astype(np.int64)
    keep = (n_winners_c > 0) & (n_winners_py > 0)

    misos = []
    frac_diffs = []
    for i in np.where(keep)[0]:
        eul_c = c_mic[i, 7:10]
        eul_py = py_mic[i, 7:10]
        try:
            angle_rad, _ = misorientation(
                list(eul_c), list(eul_py), space_group,
            )
        except Exception:
            continue
        misos.append(angle_rad * (180.0 / np.pi))
        frac_diffs.append(abs(c_mic[i, 10] - py_mic[i, 10]))

    misos = np.asarray(misos)
    frac_diffs = np.asarray(frac_diffs)
    if misos.size == 0:
        raise RuntimeError("no comparable voxels found")

    return ParityReport(
        n_total=int(c_mic.shape[0]),
        n_compared=int(misos.size),
        n_below_quarter_deg=int((misos < 0.25).sum()),
        n_below_half_deg=int((misos < 0.5).sum()),
        miso_mean_deg=float(misos.mean()),
        miso_median_deg=float(np.median(misos)),
        miso_max_deg=float(misos.max()),
        frac_diff_mean=float(frac_diffs.mean()),
        frac_diff_max=float(frac_diffs.max()),
    )


def _print_report(report: ParityReport) -> None:
    print("\n" + "=" * 60)
    print("  Parity report: midas_nf_fitorientation vs C FitOrientationOMP")
    print("=" * 60)
    print(f"  Voxels (total / compared):  {report.n_total} / {report.n_compared}")
    print(f"  Misorientation mean (deg):  {report.miso_mean_deg:.4f}")
    print(f"  Misorientation median:      {report.miso_median_deg:.4f}")
    print(f"  Misorientation max:         {report.miso_max_deg:.4f}")
    print(f"  Fraction of voxels < 0.25°: {100.0 * report.n_below_quarter_deg / report.n_compared:.2f}%")
    print(f"  Fraction of voxels < 0.50°: {100.0 * report.n_below_half_deg / report.n_compared:.2f}%")
    print(f"  Frac-overlap diff mean:     {report.frac_diff_mean:.4f}")
    print(f"  Frac-overlap diff max:      {report.frac_diff_max:.4f}")


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

def _stratified_subset(c_mic: np.ndarray, k: int) -> np.ndarray:
    """Pick ``k`` voxel indices stratified by C ``frac_overlap``.

    Sorts non-empty voxels by their C reference confidence and takes
    ``k`` evenly spaced quantiles, so the parity report covers
    boundary voxels (low frac), bulk voxels (high frac) and the
    in-between band — not just the first ``k`` rows of the grid.
    """
    valid = np.where(c_mic[:, 1] > 0)[0]
    if valid.size == 0:
        return np.arange(min(k, c_mic.shape[0]))
    valid_sorted = valid[np.argsort(c_mic[valid, 10])]
    if valid_sorted.size <= k:
        return valid_sorted
    pick = np.linspace(0, valid_sorted.size - 1, k).round().astype(np.int64)
    return valid_sorted[pick]


def run_parity(
    n_cpus: int = 8,
    keep_workspace: bool = False,
    n_voxels_subset: int = 0,
    stratified: bool = False,
    device: str = "auto",
) -> ParityReport:
    """Full pipeline: C reference → Python rerun → parity report.

    Parameters
    ----------
    n_cpus : int
        Threads for the C pipeline and PyTorch's intra-op pool.
    keep_workspace : bool
        Leave ``NF_HEDM/Example/sim/`` in place for inspection.
    n_voxels_subset : int
        If > 0, fit only the first ``n_voxels_subset`` voxels of the
        grid (via the existing block-decomposition: block 0 of
        ``n_blocks=ceil(N/k)``). Comparison is restricted to those
        voxels too. Use this for fast iteration; the full-grid fit
        is slow on CPU because of the obs-volume size.
    """
    from midas_nf_fitorientation import fit_orientation_run, parse_paramfile
    from midas_nf_fitorientation.io import read_grid

    # 1. Build a clean workspace and run the C pipeline end-to-end.
    work_dir, local_param = _setup_workspace()
    print(f"Workspace: {work_dir}")
    _simulate(local_param, work_dir / MIC_FN.name, n_cpus, work_dir)
    test_param = _patch_paramfile(local_param, work_dir)
    _run_nf_midas(test_param, n_cpus, work_dir)

    # 2. Stash the C-produced MicFileBinary as the reference.
    c_mic_path = _resolve_mic_binary(test_param, work_dir)
    if not c_mic_path.exists():
        raise FileNotFoundError(
            f"C reference MicFileBinary missing at {c_mic_path}; "
            "did FitOrientationOMP fail silently?"
        )
    c_ref_path = c_mic_path.with_suffix(c_mic_path.suffix + ".c_ref")
    shutil.copyfile(c_mic_path, c_ref_path)
    print(f"Saved C reference to {c_ref_path}")

    # 3. Re-run the fit using the Python implementation, on the same
    #    paramfile and binaries. This overwrites the C-produced
    #    MicFileBinary in-place — we kept a copy at c_ref_path.
    p = parse_paramfile(str(test_param))
    grid = read_grid(work_dir, p.grid_file_name)
    n_total = grid.n_voxels
    c_mic_full = _read_mic_binary(c_ref_path, n_total)

    if n_voxels_subset > 0:
        if stratified:
            # Stratified sample by C-reference frac — covers boundary,
            # bulk, and in-between voxels in roughly equal proportion.
            compared_indices = _stratified_subset(c_mic_full, n_voxels_subset)
            print(f"\nStratified subset: {len(compared_indices)} voxels "
                  f"(C frac range "
                  f"{c_mic_full[compared_indices, 10].min():.3f} "
                  f"… {c_mic_full[compared_indices, 10].max():.3f})")
        else:
            # First K voxels via the existing block decomposition.
            chunk = max(1, n_voxels_subset)
            n_blocks_run = max(1, -(-n_total // chunk))
            start, end = grid.slice_block(0, n_blocks_run)
            compared_indices = np.arange(start, end + 1)
            print(f"\nFirst {len(compared_indices)} voxels [{start}, {end}] "
                  f"of {n_total}")
        fit_orientation_run(
            str(test_param),
            block_nr=0, n_blocks=1, n_cpus=n_cpus,
            device=device, verbose=False,
            voxel_indices=compared_indices,
        )
    else:
        compared_indices = np.arange(n_total)
        print(f"\nRunning Python fit on all {n_total} voxels")
        fit_orientation_run(
            str(test_param),
            block_nr=0, n_blocks=1, n_cpus=n_cpus,
            device=device, verbose=False,
        )

    # 4. Read both binaries and compare on the relevant slice.
    py_mic_full = _read_mic_binary(c_mic_path, n_total)
    report = _compare(
        c_mic_full[compared_indices], py_mic_full[compared_indices], p.space_group,
    )

    if not keep_workspace and work_dir.exists():
        shutil.rmtree(work_dir)

    return report


@pytest.mark.skipif(
    not INTEGRATION,
    reason="set MIDAS_RUN_INTEGRATION=1 to run the slow C-vs-Python parity test",
)
def test_fit_orientation_matches_c_reference():
    # 30 stratified voxels gives a representative sample across the
    # C-frac range while keeping wall time reasonable on CPU.
    report = run_parity(n_cpus=8, n_voxels_subset=30, stratified=True)
    _print_report(report)
    frac_below_half = report.n_below_half_deg / report.n_compared
    assert report.miso_median_deg < MAX_MEDIAN_MISO_DEG, (
        f"median miso {report.miso_median_deg:.4f}° "
        f"exceeds threshold {MAX_MEDIAN_MISO_DEG}°"
    )
    assert frac_below_half >= MIN_FRAC_BELOW_HALF_DEG, (
        f"only {100*frac_below_half:.1f}% of voxels under 0.5° "
        f"(threshold {100*MIN_FRAC_BELOW_HALF_DEG:.1f}%)"
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-nCPUs", type=int, default=8)
    ap.add_argument("--keep-workspace", action="store_true",
                    help="Don't delete the sim/ workspace afterwards.")
    ap.add_argument("--n-voxels", type=int, default=0,
                    help="Restrict the Python fit + comparison to K voxels "
                         "of the grid. 0 = full grid (slow on CPU).")
    ap.add_argument("--stratified", action="store_true",
                    help="With --n-voxels K, sample voxels stratified by "
                         "C-reference frac instead of taking the first K.")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Forwarded to fit_orientation_run.")
    args = ap.parse_args()
    report = run_parity(
        n_cpus=args.nCPUs, keep_workspace=args.keep_workspace,
        n_voxels_subset=args.n_voxels, stratified=args.stratified,
        device=args.device,
    )
    _print_report(report)
    frac_below_half = report.n_below_half_deg / report.n_compared
    fail = (
        report.miso_median_deg >= MAX_MEDIAN_MISO_DEG
        or frac_below_half < MIN_FRAC_BELOW_HALF_DEG
    )
    sys.exit(1 if fail else 0)
