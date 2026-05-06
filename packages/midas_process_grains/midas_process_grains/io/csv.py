"""CSV writers for ``Grains.csv``, ``SpotMatrix.csv``, ``GrainIDsKey.csv``.

These mirror what ``FF_HEDM/src/ProcessGrains.c`` emits so the new pipeline
is a drop-in for downstream MIDAS tooling (DREAM.3D bridges, paraview scripts,
midas_stress consumers, etc.).

Conventions
-----------

``Grains.csv``: column-major MIDAS format; header lines + 8 skip lines + N rows
of 23 columns. We mirror the legacy column order:

  GrainID, O11..O33 (9), X, Y, Z,
  ε11_lab, ε22_lab, ε33_lab, ε12_lab, ε13_lab, ε23_lab,
  GrainRadius, Confidence

The "23-column legacy" form has more fields (lattice parameters, lab-frame
strain, additional crystal-frame strain, etc.); we expose them as part of
``Grains_extended.csv`` to avoid breaking existing consumers.

``SpotMatrix.csv``: tab-separated, 12 columns:

  GrainID, SpotID, Omega, DetectorHor, DetectorVert, OmeRaw, Eta,
  RingNr, YLab, ZLab, Theta, StrainError

``GrainIDsKey.csv``: per-cluster mapping. Each line:

  bestGrainID bestPos otherID otherPos otherID otherPos ...

(see ``ProcessGrains.c:703-710``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Grains.csv
# ---------------------------------------------------------------------------

GRAINS_HEADER_LINES_LEGACY = [
    "%NumGrains 0\n",  # placeholder; first writer pass replaces this
    "%BeamCenter 0 0\n",
    "%BeamThickness 0\n",
    "%GlobalPosition 0\n",
    "%NumPhases 1\n",
    "%PhaseInfo\n",
    "%\tSpaceGroup:225\n",
    "%\tLattice Parameter:0 0 0 0 0 0\n",
    "%ID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tX\tY\tZ\t"
    "E11\tE22\tE33\tE12\tE13\tE23\tGrainRadius\tConfidence\n",
]


def write_grains_csv(
    path: Union[str, Path],
    grains: dict,
    *,
    sg_nr: int = 225,
    lattice: Sequence[float] = (0.0,) * 6,
    beam_center: Tuple[float, float] = (0.0, 0.0),
    beam_thickness: float = 0.0,
    global_position: float = 0.0,
) -> None:
    """Write a ``Grains.csv`` (legacy 21-column form).

    ``grains`` keys (each is an array of length n):
      ids:           int32   (n,)
      orient_mat:    float64 (n, 9)   row-major
      positions:     float64 (n, 3)   X, Y, Z (µm)
      strains_lab:   float64 (n, 6)   E11, E22, E33, E12, E13, E23
      grain_radius:  float64 (n,)
      confidence:    float64 (n,)
    """
    n = len(grains["ids"])
    p = Path(path)
    with open(p, "w") as fp:
        # Header
        fp.write(f"%NumGrains {n}\n")
        fp.write(f"%BeamCenter {beam_center[0]} {beam_center[1]}\n")
        fp.write(f"%BeamThickness {beam_thickness}\n")
        fp.write(f"%GlobalPosition {global_position}\n")
        fp.write("%NumPhases 1\n")
        fp.write("%PhaseInfo\n")
        fp.write(f"%\tSpaceGroup:{sg_nr}\n")
        latstr = "\t".join(f"{x:.6f}" for x in lattice)
        fp.write(f"%\tLattice Parameter:{latstr}\n")
        fp.write(
            "%ID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tX\tY\tZ\t"
            "E11\tE22\tE33\tE12\tE13\tE23\tGrainRadius\tConfidence\n"
        )
        # Body
        for i in range(n):
            row: List[str] = [str(int(grains["ids"][i]))]
            row.extend(f"{grains['orient_mat'][i, k]:.9f}" for k in range(9))
            row.extend(f"{grains['positions'][i, k]:.6f}" for k in range(3))
            row.extend(f"{grains['strains_lab'][i, k]:.6e}" for k in range(6))
            row.append(f"{grains['grain_radius'][i]:.6f}")
            row.append(f"{grains['confidence'][i]:.6f}")
            fp.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# SpotMatrix.csv
# ---------------------------------------------------------------------------

SPOT_MATRIX_HEADER = (
    "%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw\tEta\t"
    "RingNr\tYLab\tZLab\tTheta\tStrainError\n"
)


def write_spot_matrix_csv(
    path: Union[str, Path],
    rows: np.ndarray,
) -> None:
    """Write a ``SpotMatrix.csv`` from an (n_rows, 12) array.

    Column order matches ``ProcessGrains.c::SpotMatrix_l`` exactly:
    GrainID, SpotID, Omega, DetectorHor, DetectorVert, OmeRaw, Eta, RingNr,
    YLab, ZLab, Theta, StrainError.
    """
    if rows.ndim != 2 or rows.shape[1] != 12:
        raise ValueError(
            f"rows must have shape (n, 12); got {rows.shape}"
        )
    p = Path(path)
    with open(p, "w") as fp:
        fp.write(SPOT_MATRIX_HEADER)
        for r in range(rows.shape[0]):
            fp.write(
                "\t".join((
                    f"{int(rows[r, 0])}",                # GrainID
                    f"{int(rows[r, 1])}",                # SpotID
                    f"{rows[r, 2]:.6f}",                 # Omega
                    f"{rows[r, 3]:.6f}",                 # DetectorHor
                    f"{rows[r, 4]:.6f}",                 # DetectorVert
                    f"{rows[r, 5]:.6f}",                 # OmeRaw
                    f"{rows[r, 6]:.6f}",                 # Eta
                    f"{int(rows[r, 7])}",                # RingNr
                    f"{rows[r, 8]:.6f}",                 # YLab
                    f"{rows[r, 9]:.6f}",                 # ZLab
                    f"{rows[r, 10]:.6f}",                # Theta
                    f"{rows[r, 11]:.6e}",                # StrainError
                )) + "\n"
            )


# ---------------------------------------------------------------------------
# GrainIDsKey.csv
# ---------------------------------------------------------------------------


def write_grain_ids_key_csv(
    path: Union[str, Path],
    clusters: Iterable[Tuple[int, int, Sequence[Tuple[int, int]]]],
) -> None:
    """Write a ``GrainIDsKey.csv`` describing the cluster mapping.

    Each cluster yields a single line::

        bestGrainID bestPos otherID otherPos otherID otherPos ...

    where ``bestPos`` is the row index in ``OrientPosFit.bin`` of the cluster
    representative and ``other(ID, Pos)`` are the same for the cluster's
    non-representative members.
    """
    p = Path(path)
    with open(p, "w") as fp:
        for best_id, best_pos, others in clusters:
            tokens = [str(int(best_id)), str(int(best_pos))]
            for oid, opos in others:
                tokens.append(str(int(oid)))
                tokens.append(str(int(opos)))
            fp.write(" ".join(tokens) + "\n")
