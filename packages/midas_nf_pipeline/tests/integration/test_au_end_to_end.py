"""End-to-end integration test: run the full NF pipeline on the bundled
Au example with ``NumLoops=0`` (single-resolution) and verify the
consolidated H5 contains everything we expect (voxels, maps, grains).

This test is gated by ``MIDAS_RUN_INTEGRATION=1`` because it triggers
~5+ minutes of work (image processing + fitting) on a CPU.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

from midas_nf_pipeline.workflows import run_layer_pipeline


_AU_DIR = Path(__file__).resolve().parents[4] / "NF_HEDM/Example/sim"
_AU_PARAM = _AU_DIR / "test_ps_au.txt"


@pytest.mark.skipif(
    os.environ.get("MIDAS_RUN_INTEGRATION") != "1",
    reason="set MIDAS_RUN_INTEGRATION=1 to run the slow Au end-to-end test",
)
def test_au_single_resolution_end_to_end(tmp_path):
    """Single-layer, single-resolution (NumLoops=0). No GridRefactor key."""
    if not _AU_PARAM.exists():
        pytest.skip(f"{_AU_PARAM} not found — bundled Au example missing")

    # Copy the param file into the temp workspace and force OutputDirectory.
    pf = tmp_path / "test_ps_au.txt"
    shutil.copy2(_AU_PARAM, pf)
    # Force OutputDirectory to the temp dir; remove any existing
    # GridRefactor so single-resolution mode kicks in.
    text = pf.read_text()
    lines = [
        line for line in text.splitlines()
        if not line.strip().startswith("GridRefactor")
    ]
    lines.append(f"OutputDirectory {tmp_path}")
    pf.write_text("\n".join(lines) + "\n")

    args = Namespace(
        paramFN=str(pf),
        nCPUs=4, device="auto",
        ffSeedOrientations=False, doImageProcessing=1,
        startLayerNr=1, endLayerNr=1, resultFolder=str(tmp_path),
        minConfidence=0.6, resume="", restartFrom="",
        install_dir=str(_AU_DIR.parents[1]),
    )
    h5_path = run_layer_pipeline(args, install_dir=args.install_dir)
    assert os.path.exists(h5_path), f"consolidated H5 missing: {h5_path}"

    with h5py.File(h5_path, "r") as h5:
        assert "voxels/position" in h5 or "multi_resolution/loop_0_unseeded/voxels/position" in h5
        # Either grains/ at root or loop_0_unseeded — the test allows both.
        has_grains = "grains/grain_id" in h5
        has_loop = "multi_resolution/loop_0_unseeded" in h5
        assert has_grains or has_loop


@pytest.mark.skipif(
    os.environ.get("MIDAS_RUN_INTEGRATION") != "1",
    reason="set MIDAS_RUN_INTEGRATION=1 to run the slow Au end-to-end test",
)
def test_cli_smoke():
    """``midas-nf-pipeline --help`` and subcommand help should exit cleanly."""
    proc = subprocess.run(
        ["python", "-m", "midas_nf_pipeline.cli", "--help"],
        check=True, capture_output=True, env={**os.environ, "PYTHONPATH":
            str(Path(__file__).resolve().parents[3])},
    )
    out = proc.stdout.decode()
    assert "midas-nf-pipeline" in out
    for sub in ("run", "parse-mic", "mic2grains", "consolidate", "refine-params"):
        assert sub in out
