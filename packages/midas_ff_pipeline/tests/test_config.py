"""PipelineConfig + LayerSelection unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from midas_ff_pipeline.config import LayerSelection, PipelineConfig


def test_layer_selection_basic():
    sel = LayerSelection(start=2, end=5)
    assert sel.layers() == [2, 3, 4, 5]


def test_layer_selection_single():
    sel = LayerSelection(start=1, end=1)
    assert sel.layers() == [1]


def test_layer_selection_validates():
    with pytest.raises(ValueError):
        LayerSelection(start=0)
    with pytest.raises(ValueError):
        LayerSelection(start=3, end=2)


def test_pipeline_config_resolves_paths(tmp_path: Path):
    params = tmp_path / "ps.txt"
    params.write_text("Lsd 1000000\nBC 1024 1024\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
    )
    assert Path(cfg.params_file).is_absolute()
    assert Path(cfg.result_dir).is_absolute()


def test_pipeline_config_resume_from_requires_stage(tmp_path: Path):
    params = tmp_path / "ps.txt"
    params.write_text("Lsd 1000000\n")
    with pytest.raises(ValueError):
        PipelineConfig(
            result_dir=str(tmp_path / "run"),
            params_file=str(params),
            resume="from",
            resume_from_stage=None,
        )


def test_pipeline_config_layer_dir_helper(tmp_path: Path):
    params = tmp_path / "ps.txt"
    params.write_text("Lsd 1\n")
    cfg = PipelineConfig(result_dir=str(tmp_path / "run"), params_file=str(params))
    assert cfg.layer_dir(3).name == "LayerNr_3"
    assert cfg.layer_dir(3).parent == cfg.result_path
