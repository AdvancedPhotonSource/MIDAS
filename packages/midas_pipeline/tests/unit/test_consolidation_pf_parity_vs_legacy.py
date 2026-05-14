"""Parity gate vs the legacy ``pf_MIDAS.py`` inline consolidation block.

Skipped unless the golden fixture exists at
``dev/golden_data/test_pf_5grain/`` — the 5-grain synthetic frozen output
that the user produces by running the legacy ``pf_MIDAS.py`` on the
matching parameter file. When the fixture is present we compare the
new orchestrator's ``microstrFull.csv`` against the legacy CSV at
fp64 1e-12 tolerance (the file is already round-tripped to ``%.6f``,
so this is effectively byte-identical modulo trailing whitespace).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.stages.consolidation_pf import consolidate_pf

# Resolve from the parent midas_pipeline/ package root, then up one dir
# to packages/, then to the repo root.
_PKG_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _PKG_ROOT.parents[1]
_GOLDEN = _PKG_ROOT / "dev" / "golden_data" / "test_pf_5grain"


@pytest.mark.skipif(
    not (_GOLDEN / "microstrFull.csv").exists(),
    reason=(
        "Legacy 5-grain golden fixture not present at "
        f"{_GOLDEN / 'microstrFull.csv'}. Freeze the fixture by running "
        "pf_MIDAS.py on the matching synthetic dataset and dropping the "
        "outputs into dev/golden_data/test_pf_5grain/."
    ),
)
def test_consolidation_pf_parity_against_legacy(tmp_path):
    """Run consolidate_pf against the frozen Results/ dir and diff CSV."""
    # The golden fixture should ship its own Results/ subdir + a copy of
    # the legacy microstrFull.csv to diff against.
    src_results = _GOLDEN / "Results"
    assert src_results.is_dir(), (
        "Golden fixture missing 'Results/' subdir; cannot run the parity test."
    )

    # Stage the fixture into the tmp workspace (we don't want to clobber
    # the golden dir on disk during the run).
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir()
    out_results = layer_dir / "Results"
    out_results.mkdir()
    for src in sorted(src_results.glob("*.csv")):
        (out_results / src.name).write_text(src.read_text())

    # Match the legacy SpaceGroup + nScans choice baked into the fixture.
    meta = (_GOLDEN / "fixture_meta.txt").read_text().splitlines()
    n_scans = int(next(l for l in meta if l.startswith("n_scans"))
                  .split()[1])
    space_group = int(next(l for l in meta if l.startswith("space_group"))
                      .split()[1])
    n_grains = int(next(l for l in meta if l.startswith("n_grains"))
                   .split()[1])

    result = consolidate_pf(
        layer_dir, n_grains=n_grains, n_scans=n_scans,
        space_group=space_group,
    )
    new_data = np.loadtxt(result.microstr_full_csv, delimiter=",")
    legacy_data = np.loadtxt(_GOLDEN / "microstrFull.csv", delimiter=",")

    assert new_data.shape == legacy_data.shape
    np.testing.assert_allclose(new_data, legacy_data, atol=1e-12, rtol=0)
