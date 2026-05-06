"""Unit tests for the zip_convert stage (gap #1).

The actual ``ffGenerateZipRefactor.py`` invocation is heavy-weight and
needs real raw data, so we only cover the no-op paths here:

  - ``--no-convert`` is honoured.
  - When every detector already has an existing zarr_path the stage
    short-circuits to a "skipped" result.

End-to-end raw → zarr is exercised by the existing integration tests
(``tests/test_pipeline_dryrun.py`` once a fixture lands).
"""
from __future__ import annotations

from pathlib import Path

from midas_ff_pipeline.config import LayerSelection, MachineConfig, PipelineConfig
from midas_ff_pipeline.detector import DetectorConfig
from midas_ff_pipeline.stages._base import StageContext
from midas_ff_pipeline.stages import zip_convert


def _fake_ctx(tmp_path: Path, *, convert_files: bool, zarr_exists: bool) -> StageContext:
    params = tmp_path / "params.txt"
    params.write_text("FileStem sample\nPadding 6\nStartFileNrFirstLayer 1\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "out"),
        params_file=str(params),
        layer_selection=LayerSelection(start=1, end=1),
        machine=MachineConfig(name="local"),
        convert_files=convert_files,
    )
    layer_dir = Path(cfg.result_dir) / "LayerNr_1"
    layer_dir.mkdir(parents=True, exist_ok=True)

    zp = layer_dir / "sample_000001.MIDAS.zip"
    if zarr_exists:
        zp.write_bytes(b"")  # treat presence as "already converted"

    det = DetectorConfig(det_id=1, zarr_path=str(zp) if zarr_exists else "",
                        lsd=1.0, y_bc=1024.0, z_bc=1024.0)
    return StageContext(
        config=cfg, detectors=[det],
        layer_nr=1, layer_dir=layer_dir, log_dir=layer_dir / "midas_log",
    )


def test_zip_convert_skipped_when_no_convert(tmp_path: Path):
    ctx = _fake_ctx(tmp_path, convert_files=False, zarr_exists=False)
    res = zip_convert.run(ctx)
    assert res.metrics.get("skipped") is True
    assert res.duration_s == 0.0


def test_zip_convert_skipped_when_zarr_exists(tmp_path: Path):
    ctx = _fake_ctx(tmp_path, convert_files=True, zarr_exists=True)
    res = zip_convert.run(ctx)
    assert res.metrics.get("skipped") is True
