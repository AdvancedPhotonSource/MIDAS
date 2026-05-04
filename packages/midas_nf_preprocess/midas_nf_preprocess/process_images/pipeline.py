"""End-to-end ProcessImagesCombined pipeline.

Orchestrates: TIFF load -> temporal median -> per-frame (median-subtract +
spatial median + multi-scale peak detection) -> SpotsInfo.bin.

The split between the differentiable path and the discrete spot mask:

    differentiable: filtered, log_response, spot_prob (autograd alive)
    detached:       labels, n_spots, SpotsBitMask    (graph cut)

Both ``process_layer(layer)`` and ``process_all(layers)`` are supported.
``process_layer`` is drop-in for the C executable's per-layer-invocation
pattern; ``process_all`` is the recommended Python API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import torch

from ..device import resolve_device, resolve_dtype, apply_cpu_threads
from .io import from_tensor, load_tiff_stack
from .log_filter import build_log_kernel
from .median import spatial_median, temporal_median
from .params import ProcessParams
from .peaks import PeakFindOutputs, find_peaks
from .spots_io import SpotsBitMask


@dataclass
class FrameResult:
    """Per-frame output bundle.

    All tensors share device and (for floats) dtype.
    """

    frame_index: int
    layer_nr: int  # 1-indexed, matching the C convention
    filtered: torch.Tensor          # [Z, Y], autograd
    peaks: PeakFindOutputs          # log_response/spot_prob/labels/n_components

    @property
    def labels(self) -> torch.Tensor:
        return self.peaks.labels

    @property
    def n_spots(self) -> int:
        return self.peaks.n_components

    @property
    def spot_prob(self) -> torch.Tensor:
        return self.peaks.spot_prob

    @property
    def log_response(self) -> torch.Tensor:
        return self.peaks.log_response


class ProcessImagesPipeline:
    """Orchestrator for the three-phase NF processing pipeline.

    Parameters
    ----------
    params : ProcessParams. Parsed parameter file.
    device : "cpu" | "cuda" | "mps" | torch.device | None. None auto-detects.
    dtype  : torch.dtype | str | None. None picks per-device default.
    n_cpus : optional int. Sets ``torch.set_num_threads`` on CPU only.

    Notes
    -----
    The C executable is invoked once per layer and accumulates into a shared
    ``SpotsInfo.bin`` via mmap. ``process_layer`` reproduces that pattern (also
    accepts an existing ``SpotsBitMask`` to write into). ``process_all``
    constructs the bitmask itself and processes every requested layer.
    """

    def __init__(
        self,
        params: ProcessParams,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        n_cpus: int = 0,
    ):
        self.params = params
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        apply_cpu_threads(n_cpus, self.device)

        # Pre-build LoG kernels once. Mirrors the C's two-scale pass:
        # primary (LoGMaskRadius, sigma) + fallback (4, 1.0).
        self._log_kernels = self._build_log_kernels()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_log_kernels(self) -> list[torch.Tensor]:
        if not self.params.do_log_filter:
            return []
        primary = build_log_kernel(
            self.params.log_mask_radius,
            self.params.sigma,
            integer=False,
            device=self.device,
            dtype=self.dtype,
        )
        # C L999-L1003: hardcoded fallback (radius=4, sigma=1.0)
        fallback = build_log_kernel(
            4, 1.0, integer=False, device=self.device, dtype=self.dtype
        )
        return [primary, fallback]

    # ------------------------------------------------------------------
    # Phase 1: load
    # ------------------------------------------------------------------

    def load_layer(self, layer_nr: int) -> torch.Tensor:
        """Load all frames for one layer into a tensor [N, Z, Y]."""
        return load_tiff_stack(self.params, layer_nr, self.device, self.dtype)

    def from_stack(self, stack: torch.Tensor) -> torch.Tensor:
        """Validate-and-pass-through a user-supplied stack tensor."""
        return from_tensor(
            stack,
            nr_pixels_y=self.params.nr_pixels_y,
            nr_pixels_z=self.params.nr_pixels_z,
        ).to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Phase 2: temporal median
    # ------------------------------------------------------------------

    def temporal_median(self, stack: torch.Tensor) -> torch.Tensor:
        return temporal_median(stack)

    # ------------------------------------------------------------------
    # Phase 3: per-frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_idx: int,
        frame: torch.Tensor,
        median: torch.Tensor,
        layer_nr: int,
    ) -> FrameResult:
        """Run the full per-frame pipeline on a single frame.

        Steps:
          (a) median subtraction + blanket subtraction + clamp at 0
          (b) spatial median of the configured radius
          (c) multi-scale LoG peak finding (or CC over Image2 if do_log_filter=0)
        """
        if frame.shape != median.shape:
            raise ValueError(
                f"frame shape {tuple(frame.shape)} != median shape {tuple(median.shape)}"
            )

        # (a) median subtraction. Match C: clamp negative results at 0.
        # Note: the C does integer subtraction (uint16 - uint16 - int -> int -> 0-clamped uint16).
        # We do float subtraction; clamp keeps the math equivalent for non-negative inputs.
        sub = frame - median - float(self.params.blanket_subtraction)
        img = torch.clamp(sub, min=0)

        # (b) spatial median
        if self.params.mean_filt_radius > 0:
            img = spatial_median(img, radius=self.params.mean_filt_radius)

        # (c) peak finding
        if self.params.do_log_filter and self._log_kernels:
            peaks = find_peaks(
                img,
                self._log_kernels,
                soft_temperature=self.params.soft_temperature,
            )
        else:
            # No-LoG path: label connected components of (img > 0) directly.
            from .peaks import auto_temperature, label_components

            with torch.no_grad():
                mask = img.detach() > 0
                labels, n = label_components(mask, return_n=True)
            t = self.params.soft_temperature
            T_img = auto_temperature(img) if (isinstance(t, str) and t == "auto") else float(t)
            peaks = PeakFindOutputs(
                log_response=torch.zeros_like(img),
                spot_prob=torch.sigmoid(img / T_img),
                labels=labels,
                n_components=n,
                temperature_img=float(T_img if not torch.is_tensor(T_img) else T_img.item()),
                temperature_log=1.0,
            )
        return FrameResult(
            frame_index=frame_idx,
            layer_nr=layer_nr,
            filtered=img,
            peaks=peaks,
        )

    # ------------------------------------------------------------------
    # Phase 3 + accumulate: per-layer
    # ------------------------------------------------------------------

    def process_layer(
        self,
        layer_nr: int,
        *,
        stack: Optional[torch.Tensor] = None,
        bitmask: Optional[SpotsBitMask] = None,
    ) -> SpotsBitMask:
        """Process all frames in one layer and return the populated SpotsBitMask.

        Parameters
        ----------
        layer_nr : 1-indexed layer number, matching the C ``argv[2]``.
        stack    : optional pre-loaded ``[N, Z, Y]`` tensor. If absent, loaded from disk.
        bitmask  : optional existing ``SpotsBitMask`` to write into. If absent, a
            fresh single-layer mask is allocated.
        """
        if stack is None:
            stack = self.load_layer(layer_nr)
        else:
            stack = self.from_stack(stack)
        median = self.temporal_median(stack)

        if bitmask is None:
            bitmask = SpotsBitMask(
                n_layers=self.params.n_distances,
                nr_files_per_layer=self.params.nr_files_per_distance,
                nr_pixels_y=self.params.nr_pixels_y,
                nr_pixels_z=self.params.nr_pixels_z,
            )

        # 0-indexed layer for the bitmask (matches C ``layer = nLayers - 1`` at L927).
        layer_idx = layer_nr - 1
        n_files = stack.shape[0]
        for j in range(n_files):
            result = self.process_frame(j, stack[j], median, layer_nr)
            bitmask.set_frame_from_labels(layer_idx, j, result.labels)
        return bitmask

    # ------------------------------------------------------------------
    # All layers
    # ------------------------------------------------------------------

    def process_all(
        self,
        layers: Optional[Iterable[int]] = None,
        *,
        bitmask: Optional[SpotsBitMask] = None,
    ) -> SpotsBitMask:
        """Process every requested layer into a single ``SpotsBitMask``.

        ``layers`` defaults to ``range(1, n_distances + 1)``.
        """
        if layers is None:
            layers = range(1, self.params.n_distances + 1)
        layers = list(layers)

        if bitmask is None:
            bitmask = SpotsBitMask(
                n_layers=self.params.n_distances,
                nr_files_per_layer=self.params.nr_files_per_distance,
                nr_pixels_y=self.params.nr_pixels_y,
                nr_pixels_z=self.params.nr_pixels_z,
            )
        for layer_nr in layers:
            self.process_layer(layer_nr, bitmask=bitmask)
        return bitmask
