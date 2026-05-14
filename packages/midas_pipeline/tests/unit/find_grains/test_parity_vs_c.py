"""Parity tests vs frozen C output.

These are gated on the golden fixture directory ``dev/golden_data/
test_pf_5grain/`` existing — they auto-skip when the user hasn't frozen
the C output yet. Mark them with ``parity`` so they can be filtered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
GOLDEN_DIR = REPO_ROOT / "packages" / "midas_pipeline" / "dev" / "golden_data" / "test_pf_5grain"

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    not (GOLDEN_DIR / "UniqueOrientations.csv").exists(),
    reason="golden fixture not frozen yet — skipping parity test",
)
def test_unique_orientations_parity_vs_c(tmp_path):
    """Run find_grains_single on the golden inputs, compare to frozen output."""
    from midas_pipeline.find_grains import find_grains_single

    work = GOLDEN_DIR
    art = find_grains_single(
        work_dir=work,
        space_group=int((GOLDEN_DIR / "space_group.txt").read_text().strip()),
        sino_mode="tolerance",
    )
    got = np.genfromtxt(art.unique_orientations_csv, comments="#")
    want = np.genfromtxt(GOLDEN_DIR / "UniqueOrientations.csv", comments="#")
    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.skipif(
    not (GOLDEN_DIR / "sinos_raw_5_0_0.bin").exists(),
    reason="golden sinos fixture not frozen yet — skipping",
)
def test_sinos_parity_vs_c():
    """Bit-exact float64 parity vs frozen C sino output."""
    from midas_pipeline.find_grains import find_grains_single

    work = GOLDEN_DIR
    art = find_grains_single(
        work_dir=work,
        space_group=int((GOLDEN_DIR / "space_group.txt").read_text().strip()),
        sino_mode="tolerance",
    )
    raw_got = Path(art.sinogen.sino_paths["raw"]).read_bytes()
    # Pick whichever golden raw file is present (filename includes shape).
    raw_files = list(GOLDEN_DIR.glob("sinos_raw_*.bin"))
    assert raw_files, "no golden sinos_raw found"
    raw_want = raw_files[0].read_bytes()
    assert raw_got == raw_want
