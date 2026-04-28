"""Cross-frame region pool with async GPU consumer thread.

Per-frame fits are too small to saturate a modern GPU. This pool accumulates
seeded regions across many frames and flushes large batches asynchronously
on a dedicated GPU consumer thread, while producer threads continue feeding
CPU-side work.

Producer side (any thread):
    pool.add_frame(frame_idx, omega, seeded_list)   # non-blocking; just enqueues

Consumer side (dedicated thread, started by ``pool.start()``):
    while not done:
        wait for bucket to fill OR producer done
        flush all buckets above their memory cap (or ALL if done)

Bucket capacity is computed per-bucket from free GPU memory, so a bucket of
small regions has a much higher capacity than a bucket of large ones.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from midas_peakfit.adam_fallback import adam_polish
from midas_peakfit.fit import (
    FitOutput, _emit_seed_only_rows, _pixel_bucket, _build_unc_rows,
)
from midas_peakfit.lm import LMConfig, lm_solve
from midas_peakfit.model import integrated_intensity
from midas_peakfit.postfit import build_peak_rows
from midas_peakfit.seeds import SeededRegion


@dataclass
class _PoolEntry:
    frame_idx: int
    region_id: int
    omega: float
    sr: SeededRegion
    spot_id: int


class RegionPool:
    """Async region pool. Use ``start()`` / ``stop()`` (or context manager)
    to bracket the producer phase; the consumer thread runs in between.

    Thread-safety: ``add_frame`` is safe from any number of producer threads.
    GPU operations only happen on the dedicated consumer thread.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        Ycen: float,
        Zcen: float,
        do_peak_fit: int,
        local_maxima_only: int,
        lm_config: LMConfig,
        use_adam_fallback: bool = True,
        memory_safety_factor: float = 0.4,
        max_bucket_size: int = 8000,
        log_fn=print,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.Ycen = Ycen
        self.Zcen = Zcen
        self.do_peak_fit = do_peak_fit
        self.local_maxima_only = local_maxima_only
        self.lm_config = lm_config
        self.use_adam_fallback = use_adam_fallback
        self.memory_safety_factor = memory_safety_factor
        self.max_bucket_size = max_bucket_size
        self._log = log_fn

        self.buckets: Dict[Tuple[int, int], List[_PoolEntry]] = defaultdict(list)
        self.frame_outputs: Dict[int, List[FitOutput]] = defaultdict(list)
        self._spot_id_per_frame: Dict[int, int] = defaultdict(lambda: 1)
        self._cached_capacity: Dict[Tuple[int, int], int] = {}

        # Synchronization primitives for async consumer
        self._cond = threading.Condition()
        self._done = False
        self._consumer: threading.Thread | None = None
        self._error: Exception | None = None

    # ── Capacity estimation ───────────────────────────────────────────
    def _free_memory_bytes(self) -> int:
        if self.device.type == "cuda":
            try:
                free, _total = torch.cuda.mem_get_info(self.device)
                return int(free)
            except Exception:
                return 4 * 1024 ** 3
        return 8 * 1024 ** 3

    def _capacity_for_bucket(self, n_peaks: int, m_pixels: int) -> int:
        key = (n_peaks, m_pixels)
        if key in self._cached_capacity:
            return self._cached_capacity[key]

        n_params = 1 + 8 * n_peaks
        bytes_per_dtype = 8 if self.dtype == torch.float64 else 4
        per_region = (
            6 * m_pixels                   # residuals + inputs
            + 2 * m_pixels * n_params      # Jacobian + 1 working copy
            + 4 * n_params * n_params      # H, damped H, scratch
            + 8 * n_params
        ) * bytes_per_dtype
        per_region += 4096

        free = self._free_memory_bytes()
        cap = max(8, int(free * self.memory_safety_factor / per_region))
        # Lower hard cap (was 50_000) to force the async consumer to flush
        # mid-stream during the CPU producer phase. Smaller batches are
        # ~10% less GPU-efficient but 10× more overlap → net win.
        cap = min(cap, self.max_bucket_size)
        self._cached_capacity[key] = cap
        self._log(
            f"[pool] bucket(n_peaks={n_peaks}, M={m_pixels}): "
            f"per_region={per_region / 1024:.1f}KB, "
            f"free={free / 1024**3:.2f}GB → capacity={cap}"
        )
        return cap

    # ── Async lifecycle ───────────────────────────────────────────────
    def start(self) -> None:
        """Spawn the GPU consumer thread."""
        if self._consumer is not None:
            raise RuntimeError("Pool already started")
        self._done = False
        self._consumer = threading.Thread(
            target=self._consumer_loop, name="peakfit-gpu-consumer", daemon=True
        )
        self._consumer.start()

    def stop(self) -> None:
        """Signal end-of-stream and wait for the consumer to drain everything."""
        with self._cond:
            self._done = True
            self._cond.notify_all()
        if self._consumer is not None:
            self._consumer.join()
            self._consumer = None
        if self._error is not None:
            raise self._error

    def __enter__(self) -> "RegionPool":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Always stop, even on exception, so the worker exits.
        try:
            self.stop()
        except Exception:
            if exc_type is None:
                raise

    # ── Insert (producer side, thread-safe) ───────────────────────────
    def add_frame(
        self,
        frame_idx: int,
        omega: float,
        seeded_list: List[SeededRegion],
    ) -> None:
        if not seeded_list:
            return
        with self._cond:
            for sr in seeded_list:
                spot_id = self._spot_id_per_frame[frame_idx]
                self._spot_id_per_frame[frame_idx] += sr.n_peaks
                entry = _PoolEntry(
                    frame_idx=frame_idx, region_id=sr.region_id,
                    omega=omega, sr=sr, spot_id=spot_id,
                )
                key = (sr.n_peaks, _pixel_bucket(sr.n_pixels))
                self.buckets[key].append(entry)
            # Wake the consumer so it can re-check capacities.
            self._cond.notify_all()

    # ── Consumer thread ───────────────────────────────────────────────
    def _consumer_loop(self) -> None:
        try:
            if self.device.type == "cuda":
                # set_device requires an integer index. ``self.device.index``
                # is None when the user passed plain ``torch.device("cuda")`` —
                # in that case fall back to the current device (set via
                # CUDA_VISIBLE_DEVICES from the CLI).
                idx = self.device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                torch.cuda.set_device(idx)
            while True:
                with self._cond:
                    # Snapshot which buckets are flush-ready (over capacity).
                    ready_keys = []
                    for k, entries in self.buckets.items():
                        if not entries:
                            continue
                        cap = self._capacity_for_bucket(*k)
                        if len(entries) >= cap:
                            ready_keys.append(k)

                    if not ready_keys:
                        if self._done:
                            # Drain anything left, regardless of capacity.
                            ready_keys = [k for k, e in self.buckets.items() if e]
                            if not ready_keys:
                                return
                        else:
                            # Wait for more work or end-of-stream.
                            self._cond.wait()
                            continue

                    # Pull entries out under the lock, then release for the
                    # GPU work so producers can keep filling.
                    pulled: Dict[Tuple[int, int], List[_PoolEntry]] = {}
                    for k in ready_keys:
                        pulled[k] = self.buckets[k]
                        self.buckets[k] = []

                # GPU work happens outside the lock.
                for k, entries in pulled.items():
                    self._flush_bucket_entries(k, entries)
        except Exception as e:
            self._error = e
            with self._cond:
                # Make sure stop() can return cleanly.
                self._done = True
                self._cond.notify_all()

    # ── Flush a list of entries (called only by consumer thread) ──────
    def _flush_bucket_entries(
        self, key: Tuple[int, int], entries: List[_PoolEntry]
    ) -> None:
        if not entries:
            return
        n_peaks, m_pad = key
        B = len(entries)
        N = 1 + 8 * n_peaks

        if self.do_peak_fit == 0 or self.local_maxima_only == 1:
            for e in entries:
                rows = _emit_seed_only_rows(
                    e.sr, omega=e.omega, Ycen=self.Ycen, Zcen=self.Zcen,
                    spot_id_start=e.spot_id,
                )
                self.frame_outputs[e.frame_idx].append(
                    FitOutput(
                        region_id=e.region_id, rows=rows,
                        pixel_y=e.sr.pixels_y.astype(np.int16),
                        pixel_z=e.sr.pixels_z.astype(np.int16),
                    )
                )
            return

        # Build the per-region inputs as contiguous numpy arrays first, then
        # do ONE host→device copy per tensor. The previous per-region copy
        # loop was paying ~10µs of dispatch overhead per call × 7 tensors ×
        # B regions; for B=160k that's seconds of pure overhead.
        np_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        x_init_np = np.empty((B, N), dtype=np_dtype)
        lo_np = np.empty((B, N), dtype=np_dtype)
        hi_np = np.empty((B, N), dtype=np_dtype)
        z_np = np.zeros((B, m_pad), dtype=np_dtype)
        Rs_np = np.zeros((B, m_pad), dtype=np_dtype)
        Etas_np = np.zeros((B, m_pad), dtype=np_dtype)
        pmask_np = np.zeros((B, m_pad), dtype=np_dtype)
        for b, e in enumerate(entries):
            sr = e.sr
            x_init_np[b] = sr.x0
            lo_np[b] = sr.xl
            hi_np[b] = sr.xu
            n = sr.n_pixels
            z_np[b, :n] = sr.z_values
            Rs_np[b, :n] = sr.Rs
            Etas_np[b, :n] = sr.Etas
            pmask_np[b, :n] = 1.0
        x_init = torch.from_numpy(x_init_np).to(self.device, non_blocking=True)
        lo = torch.from_numpy(lo_np).to(self.device, non_blocking=True)
        hi = torch.from_numpy(hi_np).to(self.device, non_blocking=True)
        z = torch.from_numpy(z_np).to(self.device, non_blocking=True)
        Rs = torch.from_numpy(Rs_np).to(self.device, non_blocking=True)
        Etas = torch.from_numpy(Etas_np).to(self.device, non_blocking=True)
        pmask = torch.from_numpy(pmask_np).to(self.device, non_blocking=True)

        x_fit, c_fit, rc, sigma_x = lm_solve(
            x_init, lo, hi, z, Rs, Etas, pmask, n_peaks, config=self.lm_config
        )

        if self.use_adam_fallback:
            non_conv = (rc != 0)
            if non_conv.any():
                idx = non_conv.nonzero(as_tuple=False).squeeze(-1)
                x_a, c_a = adam_polish(
                    x_fit[idx], lo[idx], hi[idx], z[idx], Rs[idx],
                    Etas[idx], pmask[idx], n_peaks,
                )
                x_fit[idx] = x_a
                c_fit[idx] = c_a
                rc[idx] = -1

        ii_t, npix_t = integrated_intensity(x_fit, Rs, Etas, pmask, n_peaks)
        x_fit_np = x_fit.detach().cpu().numpy()
        c_fit_np = c_fit.detach().cpu().numpy()
        rc_np = rc.detach().cpu().numpy()
        ii_np = ii_t.detach().cpu().numpy()
        npix_np = npix_t.detach().cpu().numpy()
        sigma_x_np = (
            sigma_x.detach().cpu().numpy() if sigma_x.numel() > 0 else None
        )

        del x_init, lo, hi, z, Rs, Etas, pmask, x_fit, c_fit, rc, ii_t, npix_t, sigma_x
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Scatter back; mutating frame_outputs is single-writer (only the
        # consumer thread writes), so no extra lock needed.
        for b, e in enumerate(entries):
            sr = e.sr
            xb = x_fit_np[b]
            bg_f = float(xb[0])
            per_peak = xb[1:].reshape(sr.n_peaks, 8)
            rmse = float(np.sqrt(max(c_fit_np[b], 0.0)))
            rows = build_peak_rows(
                spot_id_start=e.spot_id, omega=e.omega,
                Ycen=self.Ycen, Zcen=self.Zcen,
                n_peaks_in_region=sr.n_peaks, n_pixels_in_region=sr.n_pixels,
                raw_sum_intensity=sr.raw_sum, mask_touched=sr.mask_touched,
                return_code=int(rc_np[b]), fit_rmse=rmse, bg=bg_f,
                Imax=per_peak[:, 0], R=per_peak[:, 1], Eta=per_peak[:, 2],
                sigmaGR=per_peak[:, 4], sigmaLR=per_peak[:, 5],
                sigmaGEta=per_peak[:, 6], sigmaLEta=per_peak[:, 7],
                Mu=per_peak[:, 3],
                integrated_intensity=ii_np[b][: sr.n_peaks],
                n_pixels_per_peak=npix_np[b][: sr.n_peaks],
                maxY=sr.maxY, maxZ=sr.maxZ, raw_imax=sr.maxima_values,
            )
            rows_unc = None
            if sigma_x_np is not None:
                rows_unc = _build_unc_rows(sigma_x_np[b], sr.n_peaks)
            self.frame_outputs[e.frame_idx].append(
                FitOutput(
                    region_id=e.region_id, rows=rows,
                    pixel_y=sr.pixels_y.astype(np.int16),
                    pixel_z=sr.pixels_z.astype(np.int16),
                    rows_unc=rows_unc,
                )
            )

        self._log(
            f"[pool] flushed bucket(n_peaks={n_peaks}, M={m_pad}): {B} regions"
        )

    # Backwards compat: the original sync API still works (calls
    # ``flush_all`` once at the end).
    def flush_all(self) -> None:  # pragma: no cover - kept for API compat
        with self._cond:
            for k, entries in list(self.buckets.items()):
                if not entries:
                    continue
                self.buckets[k] = []
                self._flush_bucket_entries(k, entries)


__all__ = ["RegionPool"]
