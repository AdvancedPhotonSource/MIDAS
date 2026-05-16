"""Smoke tests for CLI argument parsing + config building.

We don't run a full pipeline here — that's covered by integration tests.
The goal is to catch regressions in the argparse wiring + the
``build_config`` mapping from argparse Namespace → PipelineConfig.
"""

from __future__ import annotations

import pytest

from midas_pipeline.cli import _build_parser, build_config
from midas_pipeline.ff_shim import _inject_ff


def test_help_runs(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    captured = capsys.readouterr()
    assert "midas-pipeline" in captured.out


def test_run_ff_minimal(tmp_path):
    params = tmp_path / "P.txt"
    params.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "ff",
        "--params", str(params),
        "--result", str(tmp_path / "out"),
    ])
    cfg = build_config(args)
    assert cfg.is_ff
    assert cfg.scan.n_scans == 1


def test_run_pf_requires_n_scans(tmp_path):
    params = tmp_path / "P.txt"
    params.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "pf",
        "--params", str(params), "--result", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit, match="--n-scans"):
        build_config(args)


def test_run_pf_reads_geometry_from_paramfile(tmp_path):
    """When --n-scans / --scan-step are omitted, fall back to paramstest."""
    params = tmp_path / "P.txt"
    params.write_text(
        "nScans 15\n"
        "ScanStep 5.0\n"
        "BeamSize 5.0\n"
        "ScanPosTol 2.5\n"
    )
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "pf",
        "--params", str(params), "--result", str(tmp_path / "out"),
    ])
    cfg = build_config(args)
    assert cfg.is_pf
    assert cfg.scan.n_scans == 15
    # scan_positions reflects scan_step×n_scans, centered on 0.
    import numpy as np
    assert cfg.scan.scan_positions.shape == (15,)
    assert np.isclose(cfg.scan.scan_positions[1] - cfg.scan.scan_positions[0], 5.0)
    assert cfg.scan.beam_size_um == 5.0
    assert cfg.scan.scan_pos_tol_um == 2.5


def test_run_pf_cli_overrides_paramfile(tmp_path):
    """CLI flags must override values sniffed from the paramfile."""
    params = tmp_path / "P.txt"
    params.write_text("nScans 15\nScanStep 5.0\nBeamSize 5.0\n")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "pf",
        "--params", str(params), "--result", str(tmp_path / "out"),
        "--n-scans", "31",       # override sniffed 15
        "--scan-step", "2.0",    # override sniffed 5.0
    ])
    cfg = build_config(args)
    assert cfg.scan.n_scans == 31
    import numpy as np
    assert np.isclose(cfg.scan.scan_positions[1] - cfg.scan.scan_positions[0], 2.0)


def test_run_pf_clear_error_when_both_missing(tmp_path):
    """Empty paramstest + no CLI flags → error mentions both sources."""
    params = tmp_path / "P.txt"
    params.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "pf",
        "--params", str(params), "--result", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as excinfo:
        build_config(args)
    msg = str(excinfo.value)
    assert "n-scans" in msg
    assert "scan-step" in msg


def test_run_pf_full(tmp_path):
    params = tmp_path / "P.txt"
    params.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "pf",
        "--params", str(params), "--result", str(tmp_path / "out"),
        "--n-scans", "15", "--scan-step", "5.0", "--beam-size", "5.0",
        "--seeding-mode", "merged-ff",
        "--pf-refine-mode", "voxel_bounded",
        "--cw-potts-lambda", "0.5",
    ])
    cfg = build_config(args)
    assert cfg.is_pf
    assert cfg.scan.n_scans == 15
    assert cfg.seeding.mode == "merged-ff"
    assert cfg.refinement.position_mode == "voxel_bounded"
    assert cfg.fusion.cw_potts_lambda == 0.5


def test_friedel_default_single_sided(tmp_path):
    """Default is single-sided (matches C physics)."""
    params = tmp_path / "P.txt"
    params.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "pf",
        "--params", str(params), "--result", str(tmp_path / "out"),
        "--n-scans", "5", "--scan-step", "2.0", "--beam-size", "4.0",
    ])
    cfg = build_config(args)
    assert cfg.scan.friedel_symmetric_scan_filter is False


def test_friedel_or_form_opt_in(tmp_path):
    """``--friedel-symmetric-scan-filter`` enables the OR-form (experimental)."""
    params = tmp_path / "P.txt"
    params.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "pf",
        "--params", str(params), "--result", str(tmp_path / "out"),
        "--n-scans", "5", "--scan-step", "2.0", "--beam-size", "4.0",
        "--friedel-symmetric-scan-filter",
    ])
    cfg = build_config(args)
    assert cfg.scan.friedel_symmetric_scan_filter is True


def test_no_friedel_flag_still_accepted_as_noop(tmp_path):
    """Backwards-compat: the OLD ``--no-friedel-symmetric-scan-filter``
    flag is still accepted (it's now a no-op since False is the default)."""
    params = tmp_path / "P.txt"
    params.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "pf",
        "--params", str(params), "--result", str(tmp_path / "out"),
        "--n-scans", "5", "--scan-step", "2.0", "--beam-size", "4.0",
        "--no-friedel-symmetric-scan-filter",
    ])
    cfg = build_config(args)
    assert cfg.scan.friedel_symmetric_scan_filter is False


def test_auto_scan_mode_sniffs_pf(tmp_path):
    params = tmp_path / "P.txt"
    params.write_text("nScans 10\nBeamSize 5.0\n")
    parser = _build_parser()
    args = parser.parse_args([
        "run",  # default --scan-mode auto
        "--params", str(params), "--result", str(tmp_path / "out"),
        "--n-scans", "10", "--scan-step", "5.0", "--beam-size", "5.0",
    ])
    cfg = build_config(args)
    assert cfg.is_pf


def test_layers_range(tmp_path):
    params = tmp_path / "P.txt"
    params.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "run", "--scan-mode", "ff",
        "--params", str(params), "--result", str(tmp_path / "out"),
        "--layers", "1-3",
    ])
    cfg = build_config(args)
    assert cfg.layer_selection.start == 1
    assert cfg.layer_selection.end == 3


# ---------------------------------------------------------------------------
# ff_shim back-compat
# ---------------------------------------------------------------------------


class TestFfShim:
    def test_injects_scan_mode_after_subcommand(self):
        assert _inject_ff(["run", "--params", "P.txt"]) == \
               ["run", "--scan-mode", "ff", "--params", "P.txt"]

    def test_respects_existing_scan_mode(self):
        # If the user already passed --scan-mode, leave it alone.
        out = _inject_ff(["run", "--scan-mode", "pf", "--params", "P.txt"])
        assert "--scan-mode" in out and "pf" in out
        # Doesn't double-inject ff:
        assert out.count("ff") == 0

    def test_empty_argv_defaults_to_run_ff(self):
        assert _inject_ff([]) == ["run", "--scan-mode", "ff"]
