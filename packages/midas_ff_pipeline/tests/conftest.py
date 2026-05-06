"""Shared fixtures.

The expensive synthetic-data fixture (`au_synthetic_50`) is gated on
the env var ``MIDAS_FF_PIPELINE_E2E=1`` so unit tests stay fast and
CI without the C ``ForwardSimulationCompressed`` binary still runs.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"


def _e2e_enabled() -> bool:
    return os.environ.get("MIDAS_FF_PIPELINE_E2E", "0") in ("1", "true", "yes")


@pytest.fixture(scope="session")
def params_template() -> Path:
    """Path to the bundled FCC Au Parameters.txt template."""
    p = DATA_DIR / "Parameters.txt"
    if not p.exists():
        pytest.skip(f"missing test fixture: {p}")
    return p


@pytest.fixture(scope="session")
def au_synthetic_50(tmp_path_factory, params_template) -> Path:
    """Forward-simulate a 50-grain Au synthetic dataset on demand.

    Returns the .MIDAS.zip path. Skipped unless MIDAS_FF_PIPELINE_E2E=1
    (requires C ForwardSimulationCompressed on disk).
    """
    if not _e2e_enabled():
        pytest.skip("set MIDAS_FF_PIPELINE_E2E=1 to enable e2e fixture")

    from midas_ff_pipeline.testing import generate_synthetic_dataset
    work = tmp_path_factory.mktemp("au50")
    return generate_synthetic_dataset(
        out_dir=work, params_template=params_template,
        n_grains=50, seed=42, n_cpus=8,
    )


@pytest.fixture(scope="session")
def synthetic_run_dir(tmp_path_factory, au_synthetic_50) -> Path:
    """Result-directory shell for the e2e smoke test."""
    if not _e2e_enabled():
        pytest.skip("set MIDAS_FF_PIPELINE_E2E=1 to enable e2e fixture")
    work = tmp_path_factory.mktemp("au50_run")
    (work / "LayerNr_1").mkdir(parents=True)
    return work
