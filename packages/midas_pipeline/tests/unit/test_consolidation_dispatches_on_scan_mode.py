"""Dispatcher test for ``stages.consolidation.run``.

FF mode → returns the legacy stub (skipped=True).
PF mode → invokes the new ``consolidation_pf.consolidate_pf`` port.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from midas_pipeline.config import PipelineConfig, ScanGeometry
from midas_pipeline.stages import consolidation
from midas_pipeline.stages._base import StageContext

from .test_consolidation_pf_synthetic import _setup_fixture


def _ctx(tmp_path: Path, *, scan_mode: str, n_scans: int) -> StageContext:
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    if scan_mode == "ff":
        scan = ScanGeometry.ff()
    else:
        scan = ScanGeometry.pf_uniform(
            n_scans=n_scans, scan_step_um=2.0, beam_size_um=4.0,
        )
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=scan,
        device="cpu",
        dtype="float64",
    )
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    return StageContext(
        config=cfg,
        layer_nr=1,
        layer_dir=layer_dir,
        log_dir=log_dir,
    )


def test_ff_mode_returns_skipped_stub(tmp_path):
    ctx = _ctx(tmp_path, scan_mode="ff", n_scans=1)
    result = consolidation.run(ctx)
    assert result.skipped is True
    assert result.stage_name == "consolidation"
    # FF mode must not produce PF artefacts.
    assert not (ctx.layer_dir / "Recons" / "microstrFull.csv").exists()


def test_pf_mode_calls_consolidate_pf(tmp_path):
    # Build the PF fixture inside the layer dir the ctx will point at.
    layer_dir = _setup_fixture(tmp_path)
    # The ctx config result_dir is somewhere else; we just rebuild ctx
    # with layer_dir pointing where the fixture lives.
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.pf_uniform(n_scans=2, scan_step_um=2.0,
                                     beam_size_um=4.0),
        device="cpu",
        dtype="float64",
    )
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    ctx = StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir,
                       log_dir=log_dir)
    # Stash paramstest.txt with SpaceGroup so the dispatcher picks it up.
    (layer_dir / "paramstest.txt").write_text("SpaceGroup 225\n")

    result = consolidation.run(ctx)
    assert result.skipped is False
    assert result.stage_name == "consolidation"
    assert (layer_dir / "Recons" / "microstrFull.csv").exists()
    assert (layer_dir / "Recons" / "microstructure.hdf").exists()


def test_pf_dispatcher_resolves_space_group_default(tmp_path):
    """No paramstest.txt → defaults to space group 225 (FCC).

    Must not crash on the missing file.
    """
    layer_dir = _setup_fixture(tmp_path)
    params = tmp_path / "P.txt"
    params.write_text("")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.pf_uniform(n_scans=2, scan_step_um=2.0,
                                     beam_size_um=4.0),
        device="cpu",
        dtype="float64",
    )
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    ctx = StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir,
                       log_dir=log_dir)
    result = consolidation.run(ctx)
    assert result.metrics["space_group"] == 225
