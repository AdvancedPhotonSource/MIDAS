"""TCP server implementing the same wire protocol as
``IntegratorFitPeaksGPUStream``.

Wire format (per frame, sent by the producer):

  4-byte header : <dataset_num: uint16 LE><dtype: uint16 LE>
  payload       : N * bytes_per_pixel(dtype)
                  (for hybrid dtype=6: 4-byte overflow_count followed by
                   uint16 base + (idx[C]:int32) + (val[C]:int64))

The server accepts multiple concurrent clients; each frame is enqueued and
processed by a single integrator worker thread that owns the sparse-CSR
geometry on a single device. A writer thread drains 1D-profile output to
``lineout.bin`` (and optionally ``lineout_simple_mean.bin`` and ``Int2D.bin``)
in the same byte layout the C version emits — pairs of (R, I) doubles per R
bin, one packed block per frame.
"""
from __future__ import annotations

import queue
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from midas_integrate.image import (
    DTYPE_CODE_HYBRID,
    bytes_per_pixel,
    decode_hybrid_payload,
)
from midas_integrate.kernels import CSRGeometry, integrate, profile_1d, r_axis
from midas_integrate.params import IntegrationParams

HEADER_SIZE = 4
DEFAULT_PORT = 60439
DEFAULT_HOST = "0.0.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Frame container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class IncomingFrame:
    dataset_num: int
    dtype_code: int
    payload: bytes


@dataclass
class ProcessedFrame:
    dataset_num: int
    profile_aw: np.ndarray              # 1D area-weighted, shape (n_r,)
    profile_sm: Optional[np.ndarray]    # 1D simple mean, shape (n_r,) or None
    int2d: Optional[np.ndarray]         # 2D, shape (n_r, n_eta) or None


def _read_exactly(sock: socket.socket, n: int) -> bytes:
    """Read ``n`` bytes from ``sock`` or raise ConnectionError if EOF."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        b = sock.recv(remaining)
        if not b:
            raise ConnectionError("peer closed")
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Per-client handler (accept-thread spawns one of these per connection)
# ─────────────────────────────────────────────────────────────────────────────
class _FrameTCPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        srv: "FrameServer" = self.server  # type: ignore[assignment]
        sock = self.request
        sock.settimeout(srv.client_timeout)
        try:
            while not srv.shutdown_event.is_set():
                header = _read_exactly(sock, HEADER_SIZE)
                dataset_num, dtype_code = struct.unpack("<HH", header)
                if dtype_code == DTYPE_CODE_HYBRID:
                    overflow_count = struct.unpack(
                        "<I", _read_exactly(sock, 4)
                    )[0]
                    body_len = (srv.n_pixels * 2
                                + overflow_count * (4 + 8))
                    body = _read_exactly(sock, body_len)
                    payload = struct.pack("<I", overflow_count) + body
                else:
                    bpp = bytes_per_pixel(dtype_code)
                    payload = _read_exactly(sock, srv.n_pixels * bpp)
                srv.frame_queue.put(IncomingFrame(dataset_num, dtype_code, payload))
        except (ConnectionError, OSError, socket.timeout):
            pass


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


# ─────────────────────────────────────────────────────────────────────────────
# Worker / writer threads
# ─────────────────────────────────────────────────────────────────────────────
def _worker_loop(
    frame_queue: "queue.Queue[Optional[IncomingFrame]]",
    output_queue: "queue.Queue[Optional[ProcessedFrame]]",
    *,
    geom: CSRGeometry,
    params: IntegrationParams,
    integration_mode: str,
    write_2d: bool,
    write_simple_mean: bool,
    n_pixels_y: int,
    n_pixels_z: int,
    dark: Optional[torch.Tensor],
    on_frame: Optional[Callable[[ProcessedFrame], None]],
):
    """Consume IncomingFrame objects from ``frame_queue``, run integration,
    push ProcessedFrame to ``output_queue``."""
    while True:
        item = frame_queue.get()
        if item is None:
            output_queue.put(None)
            return
        # Decode
        if item.dtype_code == DTYPE_CODE_HYBRID:
            arr = decode_hybrid_payload(
                item.payload,
                n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
            )
        else:
            from midas_integrate.image import NUMPY_DTYPE_FOR_CODE
            npdt = np.dtype(NUMPY_DTYPE_FOR_CODE[item.dtype_code])
            arr = np.frombuffer(item.payload, dtype=npdt).reshape(
                n_pixels_z, n_pixels_y
            )
        # Page-locked host buffer + non-blocking H2D copy when streaming to a
        # CUDA device. On CPU/MPS the pin/non-blocking flags are no-ops, so
        # the same code path serves every backend.
        host = torch.from_numpy(np.ascontiguousarray(arr))
        if geom.device.type == "cuda":
            try:
                host = host.pin_memory()
            except RuntimeError:
                # Fallback: pinning can fail on systems without CUDA-aware
                # allocators (e.g. inside containers without --ipc=host).
                pass
            img = host.to(device=geom.device, dtype=geom.dtype, non_blocking=True)
        else:
            img = host.to(device=geom.device, dtype=geom.dtype)
        if dark is not None:
            img = img - dark
        int2d = integrate(img, geom, mode=integration_mode,
                          normalize=bool(params.Normalize))
        prof_aw = profile_1d(int2d, geom, mode="area_weighted")
        prof_sm = profile_1d(int2d, geom, mode="simple_mean") if write_simple_mean else None
        out = ProcessedFrame(
            dataset_num=item.dataset_num,
            profile_aw=prof_aw.detach().cpu().numpy().astype(np.float64),
            profile_sm=(prof_sm.detach().cpu().numpy().astype(np.float64)
                        if prof_sm is not None else None),
            int2d=(int2d.detach().cpu().numpy().astype(np.float64)
                   if write_2d else None),
        )
        output_queue.put(out)
        if on_frame is not None:
            try:
                on_frame(out)
            except Exception:
                pass


def _writer_loop(
    output_queue: "queue.Queue[Optional[ProcessedFrame]]",
    *,
    out_dir: Path,
    n_r: int,
    r_axis_values: np.ndarray,
    write_2d: bool,
    write_simple_mean: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    f_lineout = open(out_dir / "lineout.bin", "wb")
    f_simple = open(out_dir / "lineout_simple_mean.bin", "wb") if write_simple_mean else None
    f_2d = open(out_dir / "Int2D.bin", "wb") if write_2d else None
    try:
        while True:
            item = output_queue.get()
            if item is None:
                return
            # Lineout: pairs of (R, I) doubles per R bin (matches C writer)
            pairs = np.empty(n_r * 2, dtype=np.float64)
            pairs[0::2] = r_axis_values
            pairs[1::2] = item.profile_aw
            f_lineout.write(pairs.tobytes())
            f_lineout.flush()
            if f_simple is not None and item.profile_sm is not None:
                pairs[1::2] = item.profile_sm
                f_simple.write(pairs.tobytes())
                f_simple.flush()
            if f_2d is not None and item.int2d is not None:
                f_2d.write(item.int2d.astype(np.float64, copy=False).tobytes())
                f_2d.flush()
    finally:
        f_lineout.close()
        if f_simple is not None:
            f_simple.close()
        if f_2d is not None:
            f_2d.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FrameServer:
    """Bind together the TCP server, decode worker, and writer thread."""
    geom: CSRGeometry
    params: IntegrationParams
    out_dir: Path
    integration_mode: str = "floor"
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    queue_size: int = 64
    client_timeout: float = 5.0
    write_2d: bool = False
    write_simple_mean: bool = True
    dark: Optional[torch.Tensor] = None
    on_frame: Optional[Callable[[ProcessedFrame], None]] = None

    # internal
    shutdown_event: threading.Event = field(default_factory=threading.Event)
    frame_queue: "queue.Queue[Optional[IncomingFrame]]" = field(init=False)
    output_queue: "queue.Queue[Optional[ProcessedFrame]]" = field(init=False)
    _server: Optional[_ThreadedTCPServer] = field(default=None, init=False)
    _server_thread: Optional[threading.Thread] = field(default=None, init=False)
    _worker_thread: Optional[threading.Thread] = field(default=None, init=False)
    _writer_thread: Optional[threading.Thread] = field(default=None, init=False)

    # exposed for handler
    @property
    def n_pixels(self) -> int:
        return self.geom.n_pixels_y * self.geom.n_pixels_z

    def __post_init__(self):
        self.frame_queue = queue.Queue(maxsize=self.queue_size)
        self.output_queue = queue.Queue(maxsize=self.queue_size)

    def start(self):
        self.out_dir = Path(self.out_dir)
        # Worker thread
        self._worker_thread = threading.Thread(
            target=_worker_loop,
            name="midas-integrator-worker",
            args=(self.frame_queue, self.output_queue),
            kwargs=dict(
                geom=self.geom, params=self.params,
                integration_mode=self.integration_mode,
                write_2d=self.write_2d,
                write_simple_mean=self.write_simple_mean,
                n_pixels_y=self.geom.n_pixels_y,
                n_pixels_z=self.geom.n_pixels_z,
                dark=self.dark,
                on_frame=self.on_frame,
            ),
            daemon=True,
        )
        self._worker_thread.start()

        # Writer thread
        r_vals = r_axis(n_r=self.geom.n_r,
                        RMin=self.params.RMin,
                        RBinSize=self.params.RBinSize)
        self._writer_thread = threading.Thread(
            target=_writer_loop,
            name="midas-integrator-writer",
            args=(self.output_queue,),
            kwargs=dict(
                out_dir=self.out_dir,
                n_r=self.geom.n_r,
                r_axis_values=r_vals,
                write_2d=self.write_2d,
                write_simple_mean=self.write_simple_mean,
            ),
            daemon=True,
        )
        self._writer_thread.start()

        # TCP server
        self._server = _ThreadedTCPServer((self.host, self.port),
                                          _FrameTCPHandler)
        # Annotate the server for the handler:
        self._server.shutdown_event = self.shutdown_event   # type: ignore
        self._server.frame_queue = self.frame_queue          # type: ignore
        self._server.n_pixels = self.n_pixels                # type: ignore
        self._server.client_timeout = self.client_timeout    # type: ignore
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs=dict(poll_interval=0.2),
            name="midas-integrator-tcp",
            daemon=True,
        )
        self._server_thread.start()

    def stop(self, *, drain: bool = True, timeout: float = 5.0):
        self.shutdown_event.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        # Sentinel into the worker queue to drain
        if drain:
            self.frame_queue.put(None)
            if self._worker_thread is not None:
                self._worker_thread.join(timeout=timeout)
            if self._writer_thread is not None:
                self._writer_thread.join(timeout=timeout)

    # context manager
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight client (used by tests + tooling)
# ─────────────────────────────────────────────────────────────────────────────
def send_frame(
    image: np.ndarray,
    *,
    host: str = "127.0.0.1", port: int = DEFAULT_PORT,
    dataset_num: int = 1,
    dtype_code: Optional[int] = None,
    sock: Optional[socket.socket] = None,
) -> None:
    """Send a single frame to a running FrameServer.

    If ``dtype_code`` is None, infer from ``image.dtype``.
    """
    from midas_integrate.image import NUMPY_DTYPE_FOR_CODE
    if dtype_code is None:
        for code, npdt in NUMPY_DTYPE_FOR_CODE.items():
            if image.dtype == np.dtype(npdt):
                dtype_code = code
                break
        if dtype_code is None:
            raise ValueError(f"no dtype code for numpy dtype {image.dtype}")
    payload = image.tobytes()
    header = struct.pack("<HH", dataset_num & 0xFFFF, dtype_code)

    if sock is None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((host, port))
        own = True
    else:
        s = sock
        own = False
    try:
        s.sendall(header)
        s.sendall(payload)
    finally:
        if own:
            s.close()
