"""Compute kernels for midas-index.

Most heavy lifting is delegated:
  - Forward simulation         -> midas-diffract.HEDMForwardModel (forward_adapter.py)
  - Orientation conversions    -> midas-stress.orientation         (rotation.py shim)

Owned here: seed enumeration, orientation/position grids, binned matching, scoring.
"""

from . import (
    binning,
    constants,
    forward_adapter,
    matching,
    orientation_grid,
    position_grid,
    reduce,
    rotation,
    seeds,
)

__all__ = [
    "binning",
    "constants",
    "forward_adapter",
    "matching",
    "orientation_grid",
    "position_grid",
    "reduce",
    "rotation",
    "seeds",
]
