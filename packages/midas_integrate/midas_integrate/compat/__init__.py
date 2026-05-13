"""Compatibility adapters for inputs that don't come from a v1 paramstest.

The canonical input to ``midas_integrate`` is :func:`parse_params` reading
a v1-style paramstest text file.  ``compat.from_v2`` adds a direct path
from a ``midas_calibrate_v2`` calibration result so users don't have to
round-trip through the v1 text format.
"""
from . import from_v2

__all__ = ["from_v2"]
