"""fit_setup — replaces ``FitSetupZarr`` (1652 LoC of C).

Per-spot tilt + distortion correction, optional wedge correction, optional
geometry refine (via ``midas_calibrate.refine_geometry`` — no NLopt,
no Nelder-Mead), spot filtering, and ``paramstest.txt`` writer.
"""

from .core import fit_setup, FitSetupResult

__all__ = ["fit_setup", "FitSetupResult"]
