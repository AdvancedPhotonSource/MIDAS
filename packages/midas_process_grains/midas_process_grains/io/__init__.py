"""IO submodule.

Binary readers (mmap'd) for the per-seed records produced by the upstream
indexing + refinement stages, plus CSV writers for the canonical MIDAS
grain-output files.
"""

from .binary import (
    BinaryInputs,
    read_index_best,
    read_index_best_full,
    read_fit_best,
    read_orient_pos_fit,
    read_key,
    read_process_key,
    read_all,
)
from .csv import (
    write_grains_csv,
    write_spot_matrix_csv,
    write_grain_ids_key_csv,
)
from .hkls import HklTable, load_hkl_table
from .ids_hash import IDsHash, load_ids_hash

__all__ = [
    "BinaryInputs",
    "read_index_best",
    "read_index_best_full",
    "read_fit_best",
    "read_orient_pos_fit",
    "read_key",
    "read_process_key",
    "read_all",
    "write_grains_csv",
    "write_spot_matrix_csv",
    "write_grain_ids_key_csv",
    "HklTable",
    "load_hkl_table",
    "IDsHash",
    "load_ids_hash",
]
