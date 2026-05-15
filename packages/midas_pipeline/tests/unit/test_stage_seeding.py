"""Smoke tests for the seeding stage wiring (P7 → orchestrator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.config import (
    PipelineConfig, ScanGeometry, SeedingConfig,
)
from midas_pipeline.stages._base import StageContext
from midas_pipeline.stages import seeding


def _ctx(tmp_path: Path, *, mode: str = "unseeded", scan_mode: str = "pf",
         **seeding_overrides) -> StageContext:
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    scan = (
        ScanGeometry.ff() if scan_mode == "ff"
        else ScanGeometry.pf_uniform(n_scans=4, scan_step_um=2.0, beam_size_um=4.0)
    )
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=scan,
        device="cpu", dtype="float64",
        seeding=SeedingConfig(mode=mode, **seeding_overrides),
    )
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    return StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir,
                        log_dir=log_dir)


# ---------------------------------------------------------------------------
# Skip / no-op contracts
# ---------------------------------------------------------------------------


def test_unseeded_mode_returns_skipped_stub(tmp_path: Path):
    result = seeding.run(_ctx(tmp_path, mode="unseeded"))
    assert result.skipped is True


def test_ff_mode_in_ff_scan_returns_skipped(tmp_path: Path):
    result = seeding.run(_ctx(tmp_path, scan_mode="ff", mode="ff"))
    assert result.skipped is True


def test_ff_mode_without_grains_file_skips(tmp_path: Path):
    """seeding(ff) with no grains_file → no-op (indexer falls back to unseeded)."""
    result = seeding.run(_ctx(tmp_path, mode="ff", grains_file=None))
    assert result.skipped is True


def test_ff_mode_missing_grains_file_raises(tmp_path: Path):
    """seeding(ff) with a configured-but-missing GrainsFile raises clearly."""
    with pytest.raises(FileNotFoundError, match="grains_file"):
        seeding.run(_ctx(tmp_path, mode="ff", grains_file="/nonexistent.csv"))


# ---------------------------------------------------------------------------
# FF handoff path runs end-to-end on a synthetic Grains.csv
# ---------------------------------------------------------------------------


def test_ff_handoff_writes_unique_orientations(tmp_path: Path):
    """ff-mode + valid Grains.csv → UniqueOrientations.csv produced."""
    grains = tmp_path / "Grains.csv"
    grains.write_text(
        "%GrainID O11 O12 O13 O21 O22 O23 O31 O32 O33\n"
        "1 1 0 0 0 1 0 0 0 1\n"
        "2 -1 0 0 0 -1 0 0 0 1\n"
    )
    ctx = _ctx(tmp_path, mode="ff", grains_file=str(grains))
    result = seeding.run(ctx)
    assert result.skipped is False
    seed_csv = ctx.layer_dir / "UniqueOrientations.csv"
    assert seed_csv.exists()
    arr = np.loadtxt(seed_csv)
    assert arr.ndim == 2 and arr.shape == (2, 14)
    assert result.metrics["n_seed_grains"] == 2
    assert result.metrics["mode"] == "ff"


# ---------------------------------------------------------------------------
# merged-ff mode dispatch threads all four sub-stages together
# ---------------------------------------------------------------------------


def test_merged_ff_mode_invokes_ff_index_and_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Merged-FF path: align + merge_all + ff_index (subprocess) + handoff.

    Patches ``subprocess.run`` inside ``ff_index`` so the test doesn't
    actually run midas-index, then plants a fake ``Grains.csv`` to drive
    the handoff stage. Verifies:
    - All four sub-stages are visited in order.
    - The ff_index subprocess command line is right (``python -m
      midas_index ... paramstest_merged.txt ...``).
    - The handoff produces ``UniqueOrientations.csv``.
    """
    # Set up minimal merge_all inputs: original_positions.csv + per-scan CSVs.
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    np.savetxt(layer_dir / "original_positions.csv",
               np.array([0.0, 1.0, 2.0, 3.0]), fmt="%.4f")
    for i in range(4):
        header = " ".join(f"col{j}" for j in range(18))
        row = " ".join(["0.0"] * 18)
        (layer_dir / f"original_InputAllExtraInfoFittingAll{i}.csv").write_text(
            header + "\n" + row + "\n"
        )
    (layer_dir / "paramstest.txt").write_text("MinNHKLs 6\n")

    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(tmp_path / "P.txt"),
        scan=ScanGeometry.pf_uniform(n_scans=4, scan_step_um=2.0, beam_size_um=4.0),
        device="cpu", dtype="float64",
        seeding=SeedingConfig(mode="merged-ff", merged_align_method="none"),
    )
    (tmp_path / "P.txt").write_text("SpaceGroup 225\n")
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    ctx = StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir,
                       log_dir=log_dir)

    seen = {"cmds": []}

    def fake_run(cmd, **kwargs):
        from types import SimpleNamespace
        seen["cmds"].append((list(cmd), kwargs.get("cwd")))
        # Plant a minimal Grains.csv so handoff has something to read.
        # 1 grain row, columns matching `grains_csv_to_unique_orientations`
        # input expectations (9-element OM in cols 1..9 after GrainID).
        # Use identity OM.
        gains_path = layer_dir / "Grains.csv"
        gains_path.write_text(
            "%GrainID O11 O12 O13 O21 O22 O23 O31 O32 O33\n"
            "1 1 0 0 0 1 0 0 0 1\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    result = seeding.run(ctx)
    assert result.skipped is False
    # Merge ran:
    assert (layer_dir / "InputAllExtraInfoFittingAll.csv").exists()
    # ff_index ran:
    assert (layer_dir / "paramstest_merged.txt").exists()
    # Subprocess was called with the right command:
    assert len(seen["cmds"]) == 1
    cmd, cwd = seen["cmds"][0]
    assert cmd[1:3] == ["-m", "midas_index"]
    assert cmd[3].endswith("paramstest_merged.txt")
    assert cwd == str(layer_dir)
    # Handoff produced the seed CSV:
    seed_csv = layer_dir / "UniqueOrientations.csv"
    assert seed_csv.exists()
    arr = np.loadtxt(seed_csv)
    assert arr.ndim == 1 or arr.shape[0] >= 1


def test_ff_index_paramstest_rewrite_halves_min_nhkls(tmp_path: Path):
    """Direct unit on the paramstest rewriter inside run_ff_indexer_on_merged."""
    from midas_pipeline.seeding.ff_index import _rewrite_paramstest

    src = tmp_path / "paramstest.txt"
    src.write_text(
        "RingNumbers 1\n"
        "MinNHKLs 8\n"
        "OutputFolder /should/be/replaced\n"
        "nScans 15\n"
        "BeamSize 5.0\n"
        "ScanPosTol 2.5\n"
    )
    dst = tmp_path / "paramstest_merged.txt"
    out_folder = tmp_path / "Output_MergedFFSeeding"
    resolved = _rewrite_paramstest(
        src, dst, min_n_hkls_override=None, output_folder=out_folder,
    )
    assert resolved == 4  # 8 // 2
    body = dst.read_text()
    assert "nScans" not in body
    assert "BeamSize" not in body
    assert "ScanPosTol" not in body
    assert "RingNumbers 1" in body
    assert "MinNHKLs 4" in body
    assert f"OutputFolder {out_folder}" in body
