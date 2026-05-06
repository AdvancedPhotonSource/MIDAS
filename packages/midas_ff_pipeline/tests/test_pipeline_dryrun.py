"""Pipeline orchestration: skip-stages + resume work without running stages.

These are pure orchestration checks — no zarr, no subprocess, no GPU.
We use ``only_stages=[]`` + monkey-patched stage modules to verify the
state-machine wiring: status/resume/skip behavior.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from midas_ff_pipeline.config import LayerSelection, PipelineConfig
from midas_ff_pipeline.detector import DetectorConfig
from midas_ff_pipeline.pipeline import STAGE_NAMES, STAGE_ORDER, Pipeline
from midas_ff_pipeline.results import LayerResult, ProcessGrainsResult


def _fake_stage_module(stage_name: str, *, outputs: list[Path] = None):
    """Build a minimal stage module fake."""
    outs = outputs or []

    def _run(ctx):
        for o in outs:
            o.parent.mkdir(parents=True, exist_ok=True)
            o.write_text("dummy")
        # Use the matching result dataclass — we don't bother per-stage,
        # the LayerResult attribute setter just checks it isn't None.
        return ProcessGrainsResult(
            stage_name=stage_name,
            started_at=time.time(),
            finished_at=time.time(),
            duration_s=0.0,
            outputs={str(o): "" for o in outs},
            grains_csv=str(outs[-1]) if outs else "",
            n_grains=0,
        )

    def _expected(ctx):
        return outs

    mod = SimpleNamespace(run=_run, expected_outputs=_expected)
    return mod


@pytest.fixture
def stub_pipeline(tmp_path: Path, monkeypatch):
    params = tmp_path / "ps.txt"
    params.write_text("Lsd 1000000\nBC 1024 1024\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        layer_selection=LayerSelection(start=1, end=1),
        device="cpu",
        log_level="WARNING",
    )
    det = DetectorConfig(
        det_id=1, zarr_path="x.zip", lsd=1e6, y_bc=1024, z_bc=1024,
    )
    pipe = Pipeline(config=cfg, detectors=[det])

    # Replace each stage module with a fake that creates a sentinel file.
    layer_dir = cfg.layer_dir(1)
    for name, _ in STAGE_ORDER:
        sentinel = layer_dir / f"{name}.done"
        fake = _fake_stage_module(name, outputs=[sentinel])
        # Find the slot in STAGE_ORDER that holds this stage and replace it.
        for i, (n2, _) in enumerate(STAGE_ORDER):
            if n2 == name:
                STAGE_ORDER[i] = (n2, fake)
    monkeypatch.setattr("midas_ff_pipeline.pipeline.STAGE_ORDER", STAGE_ORDER)
    return pipe, layer_dir


def test_full_run_creates_provenance_for_every_stage(stub_pipeline):
    pipe, layer_dir = stub_pipeline
    pipe.run()
    # All stage sentinel files should now exist.
    for name in STAGE_NAMES:
        assert (layer_dir / f"{name}.done").exists()
    # Provenance ledger covers every stage.
    from midas_ff_pipeline.provenance import ProvenanceStore
    stages = ProvenanceStore(layer_dir).all_stages()
    assert set(stages) == set(STAGE_NAMES)
    assert all(s["status"] == "complete" for s in stages.values())


def test_resume_auto_skips_complete_stages(stub_pipeline):
    pipe, layer_dir = stub_pipeline
    pipe.run()                                 # first pass — everything runs

    # Re-build a Pipeline with the same config — auto-resume should skip all.
    pipe2 = Pipeline(config=pipe.config, detectors=pipe.detectors)
    pipe2.run()
    # Layer result still has correct path
    assert pipe2.layer_results[-1].layer_dir == str(layer_dir)


def test_only_and_skip(stub_pipeline):
    pipe, layer_dir = stub_pipeline
    pipe.config.only_stages = ["hkl", "binning"]
    pipe.run()
    assert (layer_dir / "hkl.done").exists()
    assert (layer_dir / "binning.done").exists()
    assert not (layer_dir / "indexing.done").exists()


def test_status_reports_layers(stub_pipeline):
    pipe, _ = stub_pipeline
    pipe.run()
    status = pipe.status()
    assert status["layers"][0]["complete"] is True
