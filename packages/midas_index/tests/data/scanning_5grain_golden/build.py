"""Freeze the 5-grain × 15-scan golden fixture for the P5c parity test.

How to use:
    1. From the MIDAS root, run the C pipeline that generates the synthetic:
           python tests/test_pf_hedm.py -nCPUs 8 --no-cleanup --doTomo 0
       (See CLAUDE.md for the C-binary build/setup if it's missing.)
    2. Run this script from any directory:
           python packages/midas_index/tests/data/scanning_5grain_golden/build.py
    3. The script copies the inputs (paramstest.txt, Spots.bin, hkls.csv,
       positions.csv, Data.bin, nData.bin, SpotsToIndex.csv,
       IDsHash.csv) and the C-generated output
       (Output/IndexBest_all.bin) here.

The frozen fixture pins the parity contract — re-running this script
will overwrite the on-disk fixture with whatever the current C binaries
produce. **Commit the result intentionally** if you want to update the
gate.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


# Source: the test_pf_hedm work_dir (set by MIDAS_HOME/FF_HEDM/Example/pfhedm_test).
# Path is: <MIDAS>/packages/midas_index/tests/data/scanning_5grain_golden/build.py
HERE = Path(__file__).parent.resolve()
MIDAS_HOME = HERE.parents[4]
SRC_WORK_DIR = MIDAS_HOME / "FF_HEDM" / "Example" / "pfhedm_test"

# Files copied verbatim into the fixture root.
#
# Notes:
# - ``IDsHash.csv`` only exists in FF mode; PF mode has per-scan
#   equivalents inside each scan subdir. The indexer's
#   ``load_observations`` doesn't read it.
# - ``Data.bin`` + ``nData.bin`` are the binned spot index produced by
#   ``SaveBinDataScanning``. ``nData.bin`` runs ~1 GB for this
#   fixture's bin geometry — too big to commit. They're gitignored
#   here and **copied locally** by this build script. The parity test
#   skips with a helpful message if they're absent.
ROOT_FILES = [
    "paramstest.txt",
    "Spots.bin",
    "hkls.csv",
    "positions.csv",
    "SpotsToIndex.csv",
    "Data.bin",
    "nData.bin",
]

# Files copied into the ``golden/`` subdir — these are the reference
# outputs the parity test compares against.
GOLDEN_FILES = [
    ("Output/IndexBest_all.bin", "IndexBest_all.bin"),
]


def main() -> int:
    if not SRC_WORK_DIR.exists():
        print(f"FATAL: {SRC_WORK_DIR} does not exist.")
        print(
            "Run `python tests/test_pf_hedm.py -nCPUs 8 --no-cleanup --doTomo 0`"
            "  from the MIDAS root first."
        )
        return 1

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "golden").mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for rel in ROOT_FILES:
        src = SRC_WORK_DIR / rel
        if not src.exists():
            missing.append(str(src))
            continue
        dst = HERE / rel
        shutil.copy2(src, dst)
        print(f"  {src.name:<24} → {dst.relative_to(MIDAS_HOME)}")
    for src_rel, dst_name in GOLDEN_FILES:
        src = SRC_WORK_DIR / src_rel
        if not src.exists():
            missing.append(str(src))
            continue
        dst = HERE / "golden" / dst_name
        shutil.copy2(src, dst)
        print(f"  {src.name:<24} → {dst.relative_to(MIDAS_HOME)}")

    if missing:
        print()
        print("Missing inputs (the C pipeline didn't produce them):")
        for m in missing:
            print(f"  - {m}")
        print(
            "\nRe-run test_pf_hedm.py and watch the indexer step — the "
            "missing files come from there."
        )
        return 2
    print("\nFixture frozen at:", HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
