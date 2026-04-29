"""calc_radius — replaces ``CalcRadiusAllZarr`` (442 LoC of C).

Filters merged peaks by ring membership (``|R - RingRad| < Width``), computes
Bragg angle, grain volume, grain radius, and per-ring powder intensity. Writes
the 24-column ``Radius_StartNr_*_EndNr_*.csv``.
"""

from .core import calc_radius, RadiusResult

__all__ = ["calc_radius", "RadiusResult"]
