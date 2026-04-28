"""midas_peakfit: differentiable PyTorch peak-fitting for FF-HEDM Zarr archives.

Drop-in replacement for the C tool ``PeaksFittingOMPZarrRefactor``. Replaces
NLopt Nelder-Mead with batched Levenberg-Marquardt on CPU/CUDA. Output binary
files (``AllPeaks_PS.bin``, ``AllPeaks_PX.bin``) match the C tool's format
(see ``FF_HEDM/src/PeaksFittingConsolidatedIO.h``).

Entry points:
    peakfit_torch CLI         — see ``midas_peakfit.cli``
    midas_peakfit.run(...)    — programmatic; see ``midas_peakfit.pipeline_main``
"""

__version__ = "0.1.0"

from midas_peakfit.params import ZarrParams  # noqa: E402,F401

__all__ = ["__version__", "ZarrParams"]
