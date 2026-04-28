"""Regression test: midas-index output vs C IndexerOMP on a synthetic dataset.

Builds a small synthetic dataset (5 grains, identity-derived orientations),
runs both `IndexerOMP` (C) and `midas-index` (Python) on the same input,
and asserts the recovered orientations agree within tolerance.

Marked `slow` because it shells out to subprocess and depends on the C
binary being available. CI runs that don't have IndexerOMP skip this.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"

INDEXER_OMP_BIN = Path("/Users/hsharma/opt/MIDAS/FF_HEDM/bin/IndexerOMP")
GETHKLLIST_BIN = Path("/Users/hsharma/opt/MIDAS/FF_HEDM/bin/GetHKLList")


@pytest.mark.slow
@pytest.mark.skipif(
    not (INDEXER_OMP_BIN.exists() and GETHKLLIST_BIN.exists()),
    reason="C IndexerOMP / GetHKLList binaries not found",
)
def test_midas_index_matches_c_indexer_on_synthetic_5_grain_dataset(tmp_path):
    """Run both indexers on a 5-grain Cu synthetic dataset, compare records."""
    build_script = DATA_DIR / "build_reference.py"
    workdir = tmp_path / "ref"
    cmd = [
        sys.executable, str(build_script),
        "--n-grains", "5",
        "--seed", "42",
        "--n-procs", "1",
        "--workdir", str(workdir),
    ]
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Capture as bytes; C IndexerOMP can write garbage bytes to stdout from
    # uninitialized strings (the IDsFileName field appears truncated), so
    # text=True would crash on UnicodeDecodeError.
    res = subprocess.run(cmd, capture_output=True, env=env, timeout=300)
    if res.returncode != 0:
        out = res.stdout.decode("utf-8", errors="replace")
        err = res.stderr.decode("utf-8", errors="replace")
        raise AssertionError(
            f"build_reference.py failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        )

    golden = workdir / "golden" / "IndexBest.bin"
    ours = workdir / "midas" / "IndexBest.bin"
    assert golden.exists(), f"missing C output: {golden}"
    assert ours.exists(), f"missing midas output: {ours}"

    g = np.fromfile(golden, dtype=np.float64).reshape(-1, 15)
    m = np.fromfile(ours, dtype=np.float64).reshape(-1, 15)
    assert g.shape == m.shape == (5, 15)

    # Misorientation between recovered orientations (cubic symmetry, sg 225)
    from midas_stress.orientation import misorientation_om

    misos = []
    for i in range(g.shape[0]):
        # Skip empty slots (both zero)
        if (g[i] == 0).all() and (m[i] == 0).all():
            continue
        c_R = g[i, 1:10].reshape(3, 3)
        m_R = m[i, 1:10].reshape(3, 3)
        ang_rad, _ = misorientation_om(
            c_R.flatten().tolist(), m_R.flatten().tolist(), 225,
        )
        miso = math.degrees(float(ang_rad))
        misos.append(miso)

        # Match counts must agree exactly
        assert int(g[i, 14]) == int(m[i, 14]), (
            f"seed {i}: n_matches mismatch C={int(g[i, 14])} vs M={int(m[i, 14])}"
        )
        # Total theor must agree exactly
        assert int(g[i, 13]) == int(m[i, 13]), (
            f"seed {i}: n_t mismatch C={int(g[i, 13])} vs M={int(m[i, 13])}"
        )
        # Misorientation within tolerance (orientation grid is 0.5°, so allow up to 1°)
        assert miso < 1.0, f"seed {i}: miso {miso:.4f}° exceeds 1.0° tolerance"

    assert len(misos) >= 5, f"only {len(misos)} non-empty seeds compared"
    # Aggregate sanity
    assert max(misos) < 1.0
    assert sum(misos) / len(misos) < 0.5
