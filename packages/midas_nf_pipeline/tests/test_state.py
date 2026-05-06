"""Tests for the pipeline-state H5 tracker."""
from __future__ import annotations

import tempfile
from pathlib import Path

from midas_nf_pipeline.state import (
    PipelineH5,
    can_skip_to,
    find_resume_stage,
    get_completed_stages,
    load_resume_info,
)


def test_fresh_h5_initialises_provenance_and_state():
    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "p.h5"
        with PipelineH5(h5_path, "nf_midas", {"foo": 1}, "param-text") as ph5:
            assert ph5.completed == []
            ph5.mark("preprocessing")
            ph5.mark("fitting")
        info = load_resume_info(h5_path)
        assert info["completed_stages"] == ["preprocessing", "fitting"]
        assert info["workflow_type"] == "nf_midas"
        assert info["param_text"] == "param-text"


def test_marks_persist_across_reopens():
    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "p.h5"
        with PipelineH5(h5_path, "nf_midas") as ph5:
            ph5.mark("a")
            ph5.mark("b")
        # Reopen
        with PipelineH5(h5_path, "nf_midas") as ph5:
            assert ph5.completed == ["a", "b"]
            ph5.mark("c")
        assert get_completed_stages(h5_path) == ["a", "b", "c"]


def test_find_resume_stage_returns_first_incomplete():
    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "p.h5"
        order = ["a", "b", "c", "d"]
        with PipelineH5(h5_path, "nf_midas") as ph5:
            ph5.mark("a")
            ph5.mark("b")
        assert find_resume_stage(h5_path, order) == "c"


def test_find_resume_stage_empty_when_all_complete():
    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "p.h5"
        order = ["a", "b"]
        with PipelineH5(h5_path, "nf_midas") as ph5:
            ph5.mark("a")
            ph5.mark("b")
        assert find_resume_stage(h5_path, order) == ""


def test_can_skip_to():
    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "p.h5"
        order = ["a", "b", "c"]
        with PipelineH5(h5_path, "nf_midas") as ph5:
            ph5.mark("a")
        assert can_skip_to(h5_path, "b", order) is True
        assert can_skip_to(h5_path, "c", order) is False


def test_reset_from_clears_target_and_after():
    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "p.h5"
        order = ["a", "b", "c", "d"]
        with PipelineH5(h5_path, "nf_midas") as ph5:
            ph5.mark("a")
            ph5.mark("b")
            ph5.mark("c")
            ph5.mark("d")
            ph5.reset_from("b", order)
            assert ph5.completed == ["a"]
        # Persisted across reopens.
        assert get_completed_stages(h5_path) == ["a"]


def test_write_dataset_roundtrip():
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "p.h5"
        with PipelineH5(h5_path, "nf_midas") as ph5:
            arr = np.arange(10, dtype=np.float64)
            ph5.write_dataset("voxels/position", arr)
            ph5.write_dataset("scalars/n", 42)
            ph5.write_dataset("scalars/name", "au_test")
        with PipelineH5(h5_path, "nf_midas") as ph5:
            assert np.array_equal(ph5.restore_dataset("voxels/position"), arr)
            assert ph5.restore_dataset("scalars/n") == 42
            assert ph5.restore_dataset("scalars/name") == "au_test"
