"""I/O layer for midas-index."""

from .binary import read_bins, read_spots
from .bins_builder import build_bin_index
from .csv import (
    read_grains_csv,
    read_hkls_csv,
    read_spots_to_index_csv,
    write_spots_to_index_csv,
)
from .consolidated import (
    ConsolidatedReadResult,
    header_size_bytes,
    read_index_best_all,
    split_records_by_voxel,
    write_index_best_all,
)
from .output import (
    close_output_files,
    open_output_files,
    write_block,
    write_seed_record,
)
from .params import read_params

__all__ = [
    "read_bins",
    "read_spots",
    "build_bin_index",
    "read_hkls_csv",
    "read_spots_to_index_csv",
    "write_spots_to_index_csv",
    "read_grains_csv",
    "read_params",
    "open_output_files",
    "close_output_files",
    "write_seed_record",
    "write_block",
    # Scan-aware (pf-HEDM) consolidated I/O — P5
    "ConsolidatedReadResult",
    "header_size_bytes",
    "read_index_best_all",
    "split_records_by_voxel",
    "write_index_best_all",
]
