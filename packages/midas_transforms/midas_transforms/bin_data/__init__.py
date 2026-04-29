"""bin_data — replaces the C ``SaveBinData`` binary.

Reads ``InputAll.csv``, ``InputAllExtraInfoFittingAll.csv``, ``paramstest.txt``;
writes ``Spots.bin``, ``ExtraInfo.bin``, and (unless ``NoSaveAll==1``)
``Data.bin`` + ``nData.bin``.
"""

from .core import bin_data, BinDataResult

__all__ = ["bin_data", "BinDataResult"]
