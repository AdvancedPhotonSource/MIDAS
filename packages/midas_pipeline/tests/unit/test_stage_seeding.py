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
# merged-ff mode dispatch reaches the ff_index NotImplementedError gate
# ---------------------------------------------------------------------------


def test_merged_ff_mode_reaches_ff_index_gate(tmp_path: Path):
    """The merged-ff path runs align (method='none') + merge_all, then
    raises NotImplementedError from ff_index until that stage is wired.
    Verifies the orchestrator threads all four sub-stages together.
    """
    # Set up minimal merge_all inputs: original_positions.csv + per-scan CSVs.
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    np.savetxt(layer_dir / "original_positions.csv",
               np.array([0.0, 1.0, 2.0, 3.0]), fmt="%.4f")
    for i in range(4):
        header = " ".join(f"col{j}" for j in range(18))
        # 1 spot per scan; col 3 (weight) > 0.01.
        row = " ".join(["0.0"] * 18)
        (layer_dir / f"original_InputAllExtraInfoFittingAll{i}.csv").write_text(
            header + "\n" + row + "\n"
        )

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
    # ff_index is the not-yet-wired sub-stage — the orchestrator should
    # raise on entering it after align + merge_all succeed.
    with pytest.raises(NotImplementedError, match="ff_index"):
        seeding.run(ctx)
    # Merge actually ran (merge_all writes positions.csv + the FF input
    # file before ff_index is invoked).
    assert (layer_dir / "InputAllExtraInfoFittingAll.csv").exists()
