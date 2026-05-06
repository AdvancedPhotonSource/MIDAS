"""Compute submodule.

Pure-tensor implementations of:
  - symmetry table builders (24-op cubic / hexagonal / etc.)
  - hkl-row permutation under each symmetry op
  - cluster-level orientation canonicalisation
  - misorientation graph + connected components (Phase 1)
  - spot-aware sub-clustering (Phase 2)
  - per-hkl SpotID conflict resolution (Phase 3)
  - lstsq strain solver (Phase 4)
  - Hooke's-law stress (Phase 5)
  - twin post-processor

Design rule: every public function takes a ``device`` / ``dtype`` argument or
honours the caller's tensors' device + dtype, mirroring the conventions of
``midas_index`` and ``midas_transforms``.
"""

from .symmetry import (
    SymmetryTable,
    build_symmetry_table,
    apply_sym_to_hkl_int,
)
from .canonicalize import (
    pick_best_sym_op,
    align_member_to_rep,
)

__all__ = [
    "SymmetryTable",
    "build_symmetry_table",
    "apply_sym_to_hkl_int",
    "pick_best_sym_op",
    "align_member_to_rep",
]
