"""CLI argv parsing — smoke checks each subcommand parses without erroring."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from midas_ff_pipeline.cli import _build_parser


def test_run_parser_accepts_minimum(tmp_path: Path):
    parser = _build_parser()
    args = parser.parse_args([
        "run",
        "--params", "p.txt",
        "--result", str(tmp_path),
    ])
    assert args.cmd == "run"
    assert args.layers == "1-1"
    assert args.device == "cuda"
    assert args.solver == "lbfgs"


def test_run_parser_propagates_overrides():
    parser = _build_parser()
    args = parser.parse_args([
        "run",
        "--params", "p.txt", "--result", "/tmp/x",
        "--layers", "2-7",
        "--device", "cpu", "--dtype", "float32",
        "--solver", "lm", "--loss", "angular", "--mode", "iterative",
        "--group-size", "8", "--pg-mode", "legacy",
        "--only", "indexing", "--only", "refinement",
    ])
    assert args.layers == "2-7"
    assert args.device == "cpu" and args.dtype == "float32"
    assert args.solver == "lm" and args.loss == "angular" and args.mode == "iterative"
    # ``--group-size`` is now string-typed at the argparse layer to allow
    # the ``auto`` sentinel; CLI resolver converts to int before
    # PipelineConfig sees it.
    assert args.group_size == "8" and args.pg_mode == "legacy"
    assert args.only == ["indexing", "refinement"]


def test_status_parser():
    parser = _build_parser()
    args = parser.parse_args(["status", "/tmp/run", "--json"])
    assert args.cmd == "status"
    assert args.json is True


def test_resume_parser_requires_from():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["resume", "/tmp/run"])
    args = parser.parse_args(["resume", "/tmp/run", "--from", "indexing"])
    assert args.from_stage == "indexing"


def test_inspect_parser():
    parser = _build_parser()
    args = parser.parse_args(["inspect", "/tmp/run/LayerNr_1", "--json"])
    assert args.layer_dir == "/tmp/run/LayerNr_1"
    assert args.json is True


def test_simulate_parser():
    parser = _build_parser()
    args = parser.parse_args([
        "simulate", "--out", "/tmp/sim",
        "--params", "Parameters.txt",
        "--n-grains", "100",
    ])
    assert args.n_grains == 100


def test_status_runs_on_empty_dir(tmp_path: Path, capsys):
    """status prints something useful even when no LayerNr_* exists."""
    from midas_ff_pipeline.cli import _cmd_status

    class _A:
        result_dir = str(tmp_path)
        layers = None
        json = False
    rc = _cmd_status(_A())
    assert rc == 2  # no LayerNr_* dirs


# ---- auto-resolver tests --------------------------------------------------

def test_resolve_dtype_auto():
    from midas_ff_pipeline.cli import _resolve_dtype
    assert _resolve_dtype("cuda", "auto") == "float32"
    assert _resolve_dtype("mps", "auto") == "float32"
    assert _resolve_dtype("cpu", "auto") == "float64"
    # explicit values pass through
    assert _resolve_dtype("cuda", "float64") == "float64"
    assert _resolve_dtype("cpu", "float32") == "float32"


def test_resolve_shard_gpus():
    from midas_ff_pipeline.cli import _resolve_shard_gpus
    assert _resolve_shard_gpus("cpu", "auto") is None
    assert _resolve_shard_gpus("cuda", "none") is None
    assert _resolve_shard_gpus("cuda", "") is None
    # explicit comma list
    assert _resolve_shard_gpus("cuda", "0,1") == "0,1"


def test_resolve_group_size_explicit():
    from midas_ff_pipeline.cli import _resolve_group_size
    assert _resolve_group_size("cuda", None, "1") == 1
    assert _resolve_group_size("cuda", "0,1", "16") == 16
    # CPU device → falls back to 4 regardless
    assert _resolve_group_size("cpu", None, "auto") == 4
