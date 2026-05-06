"""Shared fixtures for midas_process_grains tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest


@pytest.fixture
def tiny_run_dir(tmp_path: Path) -> Path:
    """Build a minimal run directory with synthetic binary inputs.

    Used for IO smoke tests. Schemas match what the C pipeline emits.

    Parameters chosen for readability over realism:
      - 3 seeds (rows in OrientPosFit / Key / ProcessKey)
      - Each seed has up to 5 matched theoretical reflections
        (we still allocate the full MAX_N_HKLS=5000 slots zero-padded)
    """
    n_seeds = 3
    rd = tmp_path
    (rd / "Output").mkdir()
    (rd / "Results").mkdir()

    # OrientPosFit.bin: (3, 27) float64
    opf = np.zeros((n_seeds, 27), dtype=np.float64)
    # Sentinel SpId in cols 0, 10, 14, 21
    for i in range(n_seeds):
        sp = i + 1
        opf[i, 0] = opf[i, 10] = opf[i, 14] = opf[i, 21] = float(sp)
    # Identity orientation matrix
    for i in range(n_seeds):
        opf[i, 1] = 1.0  # O11
        opf[i, 5] = 1.0  # O22
        opf[i, 9] = 1.0  # O33
    # Distinct positions
    for i in range(n_seeds):
        opf[i, 11:14] = [10.0 * i, 20.0 * i, 30.0 * i]
    # Lattice params close to FCC Cu
    for i in range(n_seeds):
        opf[i, 15:21] = [3.6, 3.6, 3.6, 90.0, 90.0, 90.0]
    # Errors (pos, ome, IA)
    for i in range(n_seeds):
        opf[i, 22:25] = [0.5, 0.05, 0.01 * (i + 1)]
    # meanRadius, completeness
    for i in range(n_seeds):
        opf[i, 25] = 5.0 + i
        opf[i, 26] = 0.95
    opf.tofile(rd / "Results" / "OrientPosFit.bin")

    # Key.bin: (3, 2) int32
    key = np.zeros((n_seeds, 2), dtype=np.int32)
    key[:, 0] = 1  # all alive
    key[:, 1] = [4, 5, 3]  # NrIDsPerID
    key.tofile(rd / "Results" / "Key.bin")

    # ProcessKey.bin: (3, 5000) int32, with first few SpotIDs filled
    MAX_N_HKLS = 5000
    pk = np.zeros((n_seeds, MAX_N_HKLS), dtype=np.int32)
    pk[0, :4] = [101, 102, 103, 104]
    pk[1, :5] = [201, 202, 203, 204, 205]
    pk[2, :3] = [301, 302, 303]
    pk.tofile(rd / "Results" / "ProcessKey.bin")

    # IndexBest.bin: (3, 15) float64 -- avg_ia, 9 OM, 3 pos, n_t, n_match
    ib = np.zeros((n_seeds, 15), dtype=np.float64)
    for i in range(n_seeds):
        ib[i, 0] = 0.01
        ib[i, 1] = ib[i, 5] = ib[i, 9] = 1.0
        ib[i, 13] = 50.0  # n_t_spots
        ib[i, 14] = key[i, 1]  # n_matches
    ib.tofile(rd / "Output" / "IndexBest.bin")

    # IndexBestFull.bin: (3, 5000, 2) float64
    ibf = np.zeros((n_seeds, MAX_N_HKLS, 2), dtype=np.float64)
    ibf[0, :4, 0] = [101, 102, 103, 104]
    ibf[1, :5, 0] = [201, 202, 203, 204, 205]
    ibf[2, :3, 0] = [301, 302, 303]
    ibf.tofile(rd / "Output" / "IndexBestFull.bin")

    # FitBest.bin: (3, 5000, 22) float64
    fb = np.zeros((n_seeds, MAX_N_HKLS, 22), dtype=np.float64)
    # First column = SpotID, others zero — sufficient for IO smoke
    fb[0, :4, 0] = [101, 102, 103, 104]
    fb[1, :5, 0] = [201, 202, 203, 204, 205]
    fb[2, :3, 0] = [301, 302, 303]
    fb.tofile(rd / "Output" / "FitBest.bin")

    # paramstest.txt: minimal viable
    (rd / "paramstest.txt").write_text(
        "LatticeParameter 3.6 3.6 3.6 90.0 90.0 90.0;\n"
        "Wavelength 0.172979;\n"
        "Distance 800000.0;\n"
        "px 200.0;\n"
        "SpaceGroup 225;\n"
        "RingNumbers 1;\n"
        "RingNumbers 2;\n"
        "RingRadii 60000.0;\n"
        "RingRadii 70000.0;\n"
        "MinNrSpots 2;\n"
        "MisoriTol 0.25;\n"
        "OutputFolder " + str(rd / "Output") + "\n"
        "ResultFolder " + str(rd / "Results") + "\n"
    )

    # hkls.csv: full {111} + {200} orbits so every cubic op maps inside.
    text = "h k l D-spacing RingNr g1 g2 g3 Theta(deg) 2Theta(deg) Radius\n"
    for h in (-1, 1):
        for k in (-1, 1):
            for l in (-1, 1):
                text += f"{h} {k} {l} 2.0784 1 0 0 0 2.39 4.78 60000.0\n"
    for ax in (0, 1, 2):
        for sgn in (-2, 2):
            hkl = [0, 0, 0]
            hkl[ax] = sgn
            text += (
                f"{hkl[0]} {hkl[1]} {hkl[2]} 1.8 2 0 0 0 2.76 5.52 70000.0\n"
            )
    (rd / "hkls.csv").write_text(text)

    return rd


@pytest.fixture
def cubic_hkl_table(tmp_path: Path):
    """A 5-reflection FCC cubic HKL table for symmetry tests."""
    from midas_process_grains.io.hkls import load_hkl_table
    p = tmp_path / "hkls.csv"
    p.write_text(
        "h k l D-spacing RingNr g1 g2 g3 Theta(deg) 2Theta(deg) Radius\n"
        "1 1 1 2.0754 1 0.27819 0.27819 0.27819 2.39 4.78 60000.0\n"
        "2 0 0 1.7975 2 0.55620 0.0 0.0 2.76 5.52 70000.0\n"
        "2 2 0 1.2710 3 0.55620 0.55620 0.0 3.91 7.81 80000.0\n"
        "3 1 1 1.0838 4 0.83454 0.27819 0.27819 4.59 9.18 90000.0\n"
        "2 2 2 1.0377 5 0.55620 0.55620 0.55620 4.79 9.58 95000.0\n"
    )
    return load_hkl_table(p)
