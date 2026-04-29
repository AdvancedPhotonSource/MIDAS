"""merge_overlapping_peaks — replaces ``MergeOverlappingPeaksAllZarr``.

Frame-by-frame mutual-nearest-neighbor merge of consolidated peakfit
output. v0.1.0 supports the centroid-distance mode (the production
default); the pixel-overlap mode is a follow-up.
"""

from .core import merge_overlapping_peaks, MergeResult

__all__ = ["merge_overlapping_peaks", "MergeResult"]
