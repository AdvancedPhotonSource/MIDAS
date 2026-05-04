"""ProcessImagesCombined port: median + LoG peak detection -> SpotsInfo.bin.

The differentiability boundary:

    autograd-alive: filtered, log_response, spot_prob
    detached:       labels, n_components, SpotsBitMask
"""

from .params import ProcessParams
from .io import load_tiff_stack, from_tensor, frame_paths
from .median import temporal_median, spatial_median
from .log_filter import build_log_kernel, apply_log
from .peaks import (
    label_components,
    zero_crossings,
    find_peaks,
    auto_temperature,
    PeakFindOutputs,
)
from .spots_io import SpotsBitMask
from .pipeline import ProcessImagesPipeline, FrameResult

__all__ = [
    "ProcessParams",
    "load_tiff_stack",
    "from_tensor",
    "frame_paths",
    "temporal_median",
    "spatial_median",
    "build_log_kernel",
    "apply_log",
    "label_components",
    "zero_crossings",
    "find_peaks",
    "auto_temperature",
    "PeakFindOutputs",
    "SpotsBitMask",
    "ProcessImagesPipeline",
    "FrameResult",
]
