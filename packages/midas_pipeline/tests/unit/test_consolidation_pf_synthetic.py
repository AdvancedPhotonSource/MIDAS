"""Smoke + correctness tests for the PF consolidation port.

Synthesizes a 2-grain × 4-voxel × 1-scan-position fixture, runs
``consolidate_pf``, and asserts the produced ``microstrFull.csv`` has
the expected shape, column count, and per-voxel grain assignment.

The fixture is intentionally minimal — we only need the per-voxel
``Results/*.csv`` files plus a ``UniqueOrientations.csv``. The legacy
inline code doesn't actually consume ``UniqueOrientations.csv`` for
``microstrFull.csv`` generation (it's used in the *reconstruct* stage
upstream), so the file presence is sufficient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from midas_pipeline.stages.consolidation_pf import (
    MICROSTR_HEADER,
    N_COLS,
    consolidate_pf,
)


def _write_voxel_result(results_dir: Path, vox_nr: int, *, completeness: float,
                        om: np.ndarray, posXY: tuple[float, float] = (0.0, 0.0),
                        lattice=(3.6, 3.6, 3.6, 90.0, 90.0, 90.0)) -> None:
    """Write one per-voxel result CSV in the legacy 43-col format.

    Filename pattern matches the C refiner: ``FitBest_<voxNr:06d>_<SpId:09d>.csv``.
    Column layout follows ``pf_MIDAS.py:2462-2467``:
      [0]      SpotID
      [1:10]   orientation matrix (flat 9)
      [11:14]  position (x,y,z)
      [15:21]  lattice (a,b,c,alpha,beta,gamma)
      [22]     PosErr
      [23]     OmeErr
      [24]     InternalAngle
      [25]     Radius
      [26]     Completeness  ← acceptance gate
      [27:36]  strain (E11..E33)
      [36:39]  euler (Eul1..Eul3)
      [39:43]  quaternion (filled by consolidate_pf)
    """
    row = np.zeros(N_COLS, dtype=np.float64)
    row[0] = float(vox_nr)
    row[1:10] = om.ravel()
    row[11], row[12], row[13] = posXY[0], posXY[1], 0.0
    row[15:21] = lattice
    row[22] = 0.5
    row[23] = 0.1
    row[24] = 1.5
    row[25] = 100.0
    row[26] = completeness
    row[27:36] = 0.0
    row[36:39] = 0.0
    fname = results_dir / f"FitBest_{vox_nr:06d}_{vox_nr+1:09d}.csv"
    with open(fname, "w") as f:
        f.write("# Header line ignored by parser\n")
        f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def _identity_om() -> np.ndarray:
    return np.eye(3, dtype=np.float64).ravel()


def _ry90_om() -> np.ndarray:
    """Orientation matrix for a 90-deg rotation about Y."""
    return np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64).ravel()


def _setup_fixture(tmp_path: Path, n_scans: int = 2) -> Path:
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir()
    (layer_dir / "Results").mkdir()
    # 4 voxels in a 2×2 grid: 2 grains, 2 voxels each
    _write_voxel_result(layer_dir / "Results", 0,
                        completeness=0.95, om=_identity_om())
    _write_voxel_result(layer_dir / "Results", 1,
                        completeness=0.92, om=_identity_om())
    _write_voxel_result(layer_dir / "Results", 2,
                        completeness=0.88, om=_ry90_om())
    _write_voxel_result(layer_dir / "Results", 3,
                        completeness=0.91, om=_ry90_om())

    # Minimal UniqueOrientations.csv (1 row per grain, 14 cols)
    uo = layer_dir / "UniqueOrientations.csv"
    uo.write_text(
        " ".join(["0"] * 5 + [f"{v:.6f}" for v in _identity_om()]) + "\n"
        + " ".join(["1"] * 5 + [f"{v:.6f}" for v in _ry90_om()]) + "\n"
    )
    return layer_dir


def test_consolidate_pf_returns_4_rows(tmp_path):
    layer_dir = _setup_fixture(tmp_path)
    result = consolidate_pf(layer_dir, n_grains=2, n_scans=2, space_group=225)

    csv_path = Path(result.microstr_full_csv)
    assert csv_path.exists()
    data = np.loadtxt(csv_path, delimiter=",")
    assert data.shape == (4, N_COLS)
    # Completeness column should hit the 4 values we wrote.
    np.testing.assert_allclose(np.sort(data[:, 26]),
                               np.array([0.88, 0.91, 0.92, 0.95]),
                               rtol=0, atol=1e-5)


def test_consolidate_pf_rejects_invalid_completeness(tmp_path):
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir()
    (layer_dir / "Results").mkdir()

    # vox 0 accepted; vox 1 negative completeness; vox 2 >1.1; vox 3 NaN
    _write_voxel_result(layer_dir / "Results", 0,
                        completeness=0.5, om=_identity_om())
    _write_voxel_result(layer_dir / "Results", 1,
                        completeness=-0.1, om=_identity_om())
    _write_voxel_result(layer_dir / "Results", 2,
                        completeness=2.0, om=_identity_om())
    _write_voxel_result(layer_dir / "Results", 3,
                        completeness=float("nan"), om=_identity_om())

    result = consolidate_pf(layer_dir, n_grains=1, n_scans=2, space_group=225)
    data = np.loadtxt(Path(result.microstr_full_csv), delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape == (1, N_COLS)


def test_consolidate_pf_header_byte_identical(tmp_path):
    """The CSV header MUST match the legacy production string."""
    layer_dir = _setup_fixture(tmp_path)
    result = consolidate_pf(layer_dir, n_grains=2, n_scans=2, space_group=225)
    first_line = Path(result.microstr_full_csv).read_text().splitlines()[0]
    # np.savetxt prefixes the header with '# '
    assert first_line == f"# {MICROSTR_HEADER}"


def test_consolidate_pf_writes_quat_columns(tmp_path):
    """After consolidation the [39:43] quaternion block should be filled
    (FZ-reduced) — i.e. not all zeros, since the per-voxel input has
    zeros there."""
    layer_dir = _setup_fixture(tmp_path)
    result = consolidate_pf(layer_dir, n_grains=2, n_scans=2, space_group=225)
    data = np.loadtxt(Path(result.microstr_full_csv), delimiter=",")
    quats = data[:, 39:43]
    # Each row should be a unit quaternion (norm 1, up to fp roundoff).
    norms = np.linalg.norm(quats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_consolidate_pf_metrics(tmp_path):
    layer_dir = _setup_fixture(tmp_path)
    result = consolidate_pf(layer_dir, n_grains=2, n_scans=2, space_group=225)
    assert result.metrics["n_voxels_accepted"] == 4
    assert result.metrics["n_voxels_total"] == 4
    assert result.metrics["n_files_read"] == 4
    assert result.metrics["space_group"] == 225
    assert result.metrics["n_sym"] == 24  # cubic m-3m → 24 ops
