"""bin_data — replaces the C ``SaveBinData`` and ``SaveBinDataScanning`` binaries.

FF mode (``bin_data``):
    Reads ``InputAll.csv``, ``InputAllExtraInfoFittingAll.csv``,
    ``paramstest.txt``; writes ``Spots.bin`` (9 cols), ``ExtraInfo.bin``,
    and (unless ``NoSaveAll==1``) ``Data.bin`` + ``nData.bin``.

PF / scanning mode (``bin_data_scanning``):
    Reads ``N`` per-scan ``InputAllExtraInfoFittingAll{n}.csv`` files,
    sorts globally, renumbers SpotID, and writes ``Spots.bin`` (10 cols
    — the FF nine + scanNr), ``ExtraInfo.bin``, ``IDsMergedScanning.csv``,
    ``voxel_scan_pos.bin``, ``positions.csv``, and (unless ``NoSaveAll==1``)
    ``Data.bin`` + ``nData.bin`` (size_t pairs of ``(rowno, scanno)``).

Unified entry point (``bin_data_unified``) dispatches on
``scan_positions`` — None → FF, otherwise → PF. The FF path is
**bit-identical** to today's behaviour (regression gate).
"""

from .core import bin_data, BinDataResult
from .voxel_binner import (
    bin_data_scanning,
    bin_data_unified,
    VoxelBinDataResult,
)

__all__ = [
    "bin_data",
    "BinDataResult",
    "bin_data_scanning",
    "bin_data_unified",
    "VoxelBinDataResult",
]
