"""Map.bin / nMap.bin I/O — full read and write of the MIDAS v3 mapping format.

The wire format is fixed by the C code in
``FF_HEDM/src/MapHeader.h`` and ``MapperCore.h``::

    struct MapHeader {       // 64 bytes total
        uint32_t magic;          // 0x3050414D = "MAP0"
        uint32_t version;        // 3
        uint8_t  param_hash[32]; // SHA-256 of geometry parameters
        uint8_t  q_mode;         // 0=R-mode, 1=Q-mode
        uint8_t  gradient_mode;  // 0/1
        uint8_t  reserved_pad[6];
        double   wavelength;     // Å (Q-mode only)
        uint8_t  reserved[8];
    };

    struct MapPixelData {    // 24 bytes total, naturally aligned
        float  y;                // pixel column with sub-pixel offset
        float  z;                // pixel row    with sub-pixel offset
        double frac;             // corrected weight = areaWeight / C
        float  deltaR;           // R_subpixel - R_bin_center
        float  areaWeight;       // raw geometric pixel-bin overlap area
    };

nMap.bin is an array of (count: int32, data_offset: int32) pairs, one per
(R, Eta) bin. ``data_offset`` is in *MapPixelData* units, not bytes.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match MapHeader.h
# ─────────────────────────────────────────────────────────────────────────────
MAP_HEADER_MAGIC = 0x3050414D      # "MAP0" little-endian
MAP_HEADER_VERSION = 3
MAP_HEADER_SIZE = 64               # bytes

PXLIST_DTYPE = np.dtype([
    ("y",          np.float32),
    ("z",          np.float32),
    ("frac",       np.float64),
    ("deltaR",     np.float32),
    ("areaWeight", np.float32),
], align=False)
assert PXLIST_DTYPE.itemsize == 24, (
    f"struct data layout drift: {PXLIST_DTYPE.itemsize} != 24"
)

NMAP_PAIR_DTYPE = np.dtype([
    ("count",   np.int32),
    ("offset",  np.int32),
])


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MapHeader:
    magic: int = MAP_HEADER_MAGIC
    version: int = MAP_HEADER_VERSION
    param_hash: bytes = b"\x00" * 32
    q_mode: int = 0
    gradient_mode: int = 0
    wavelength: float = 0.0

    def to_bytes(self) -> bytes:
        out = bytearray(MAP_HEADER_SIZE)
        struct.pack_into("<I", out, 0, self.magic)
        struct.pack_into("<I", out, 4, self.version)
        if len(self.param_hash) != 32:
            raise ValueError("param_hash must be 32 bytes")
        out[8:40] = self.param_hash
        out[40] = self.q_mode & 0xFF
        out[41] = self.gradient_mode & 0xFF
        # 42..47 reserved_pad (zero)
        struct.pack_into("<d", out, 48, self.wavelength)
        # 56..63 reserved (zero)
        return bytes(out)

    @classmethod
    def from_bytes(cls, raw: bytes) -> Optional["MapHeader"]:
        if len(raw) < MAP_HEADER_SIZE:
            return None
        magic = struct.unpack_from("<I", raw, 0)[0]
        if magic != MAP_HEADER_MAGIC:
            return None
        version = struct.unpack_from("<I", raw, 4)[0]
        return cls(
            magic=magic,
            version=version,
            param_hash=bytes(raw[8:40]),
            q_mode=raw[40],
            gradient_mode=raw[41],
            wavelength=struct.unpack_from("<d", raw, 48)[0],
        )


def compute_param_hash(
    *,
    Lsd: float, Ycen: float, Zcen: float, pxY: float, pxZ: float,
    tx: float, ty: float, tz: float,
    p0: float = 0.0, p1: float = 0.0, p2: float = 0.0,
    p3: float = 0.0, p4: float = 0.0, p5: float = 0.0,
    p6: float = 0.0, p7: float = 0.0, p8: float = 0.0,
    p9: float = 0.0, p10: float = 0.0, p11: float = 0.0,
    p12: float = 0.0, p13: float = 0.0, p14: float = 0.0,
    Parallax: float = 0.0,
    RhoD: float,
    RBinSize: float, EtaBinSize: float,
    RMin: float, RMax: float, EtaMin: float, EtaMax: float,
    NrPixelsY: int, NrPixelsZ: int,
    TransOpt: Sequence[int] = (),
    qMode: int = 0,
    Wavelength: float = 0.0,
) -> bytes:
    """Canonical SHA-256 parameter hash for Map.bin cache invalidation.

    Builds an alphabetised "key=value|..." string and hashes it with SHA-256.
    Used for parameter-drift detection — NOT for security.

    History.  The original MIDAS C implementation
    (``map_header_compute`` in MapHeader.h) hashed only six of the fifteen
    distortion coefficients (``p0, p1, p2, p3, p4, p6``) and only included
    ``Wavelength`` when ``qMode``.  After the v2 calibration refactor
    routinely refines ALL fifteen coefficients (and any of pxY/pxZ/Wavelength
    via the prior-aware S5 protocol), changing one of the previously-omitted
    parameters silently produced a stale Map.bin.

    This implementation hashes ALL fifteen distortion coefficients,
    ``Parallax``, and ``Wavelength`` unconditionally, so any v2 calibration
    parameter change correctly invalidates the cache.  Callers that only
    pass the legacy six p-coefficients still work (the rest default to 0.0)
    but every existing Map.bin will be rebuilt on first run after this fix.
    """
    parts = [
        f"BC={Ycen:.6f},{Zcen:.6f}",
        f"EtaBinSize={EtaBinSize:.6f}",
        f"EtaMax={EtaMax:.6f}",
        f"EtaMin={EtaMin:.6f}",
        f"Lsd={Lsd:.6f}",
        f"NrPixelsY={NrPixelsY}",
        f"NrPixelsZ={NrPixelsZ}",
        f"Parallax={Parallax:.6f}",
        f"RBinSize={RBinSize:.6f}",
        f"RMax={RMax:.6f}",
        f"RMin={RMin:.6f}",
        f"RhoD={RhoD:.6f}",
        f"TransOpt={len(TransOpt)}",
        f"Wavelength={Wavelength:.8f}",
    ]
    head = "|".join(parts)
    for t in TransOpt[:10]:
        head += f",{int(t)}"
    tail_parts = [
        f"p0={p0:.6f}",   f"p1={p1:.6f}",   f"p2={p2:.6f}",
        f"p3={p3:.6f}",   f"p4={p4:.6f}",   f"p5={p5:.6f}",
        f"p6={p6:.6f}",   f"p7={p7:.6f}",   f"p8={p8:.6f}",
        f"p9={p9:.6f}",   f"p10={p10:.6f}", f"p11={p11:.6f}",
        f"p12={p12:.6f}", f"p13={p13:.6f}", f"p14={p14:.6f}",
        f"pxY={pxY:.6f}",
        f"pxZ={pxZ:.6f}",
        f"tx={tx:.6f}",
        f"ty={ty:.6f}",
        f"tz={tz:.6f}",
    ]
    full = head + "|" + "|".join(tail_parts)
    if qMode:
        full += "|qMode=1"
    return hashlib.sha256(full.encode("ascii")).digest()


# ─────────────────────────────────────────────────────────────────────────────
# PixelMap container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PixelMap:
    """Read-only view of a Map.bin / nMap.bin pair, ready for CSR construction."""
    pxList: np.ndarray            # PXLIST_DTYPE, shape (n_entries,)
    counts: np.ndarray            # int32, shape (n_bins,)
    offsets: np.ndarray           # int32, shape (n_bins,)
    map_header: Optional[MapHeader]
    nmap_header: Optional[MapHeader]

    @property
    def n_bins(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_entries(self) -> int:
        return int(self.pxList.shape[0])


def _detect_header(buf: memoryview) -> tuple[Optional[MapHeader], int]:
    hdr = MapHeader.from_bytes(bytes(buf[:MAP_HEADER_SIZE]))
    return (hdr, MAP_HEADER_SIZE if hdr is not None else 0)


def load_map(map_path: str | Path, nmap_path: str | Path) -> PixelMap:
    """Memory-map Map.bin and nMap.bin, auto-detect header presence."""
    map_mm = np.memmap(map_path, mode="r", dtype=np.uint8)
    map_hdr, map_off = _detect_header(memoryview(map_mm))
    pxList_bytes = map_mm[map_off:]
    if pxList_bytes.nbytes % PXLIST_DTYPE.itemsize != 0:
        raise ValueError(
            f"Map.bin data size {pxList_bytes.nbytes} not divisible by "
            f"{PXLIST_DTYPE.itemsize} (struct data size). Likely a legacy "
            "16-byte map; regenerate with current DetectorMapper."
        )
    pxList = pxList_bytes.view(PXLIST_DTYPE)

    nmap_mm = np.memmap(nmap_path, mode="r", dtype=np.uint8)
    nmap_hdr, nmap_off = _detect_header(memoryview(nmap_mm))
    nmap_view = nmap_mm[nmap_off:].view(NMAP_PAIR_DTYPE)
    counts = np.array(nmap_view["count"], dtype=np.int32, copy=True)
    offsets = np.array(nmap_view["offset"], dtype=np.int32, copy=True)

    return PixelMap(
        pxList=pxList,
        counts=counts,
        offsets=offsets,
        map_header=map_hdr,
        nmap_header=nmap_hdr,
    )


def write_map(
    map_path: str | Path,
    nmap_path: str | Path,
    *,
    pxList: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    header: Optional[MapHeader] = None,
) -> None:
    """Write a Map.bin / nMap.bin pair in the v3 binary-on-disk format.

    If ``header`` is given, the same header bytes are prepended to *both*
    files (per the C convention; only Map.bin's hash is actually validated
    on read, but writing the header to nMap.bin too matches DetectorMapper).
    """
    if pxList.dtype != PXLIST_DTYPE:
        pxList = np.asarray(pxList, dtype=PXLIST_DTYPE)
    counts = np.asarray(counts, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int32)
    if counts.shape != offsets.shape or counts.ndim != 1:
        raise ValueError("counts and offsets must be 1D arrays of equal length")

    pairs = np.empty(counts.shape[0], dtype=NMAP_PAIR_DTYPE)
    pairs["count"] = counts
    pairs["offset"] = offsets

    map_path = Path(map_path)
    nmap_path = Path(nmap_path)

    with open(map_path, "wb") as f:
        if header is not None:
            f.write(header.to_bytes())
        f.write(pxList.tobytes(order="C"))

    with open(nmap_path, "wb") as f:
        if header is not None:
            f.write(header.to_bytes())
        f.write(pairs.tobytes(order="C"))


# ─────────────────────────────────────────────────────────────────────────────
# Helper used by tests and the synthetic map demo
# ─────────────────────────────────────────────────────────────────────────────
def write_synthetic_map(
    map_path: str | Path,
    nmap_path: str | Path,
    *,
    pxList_records: Iterable[tuple[float, float, float, float, float]],
    bin_pixel_lists: Sequence[Sequence[int]],
    write_header: bool = False,
) -> None:
    """Pack a list of (y, z, frac, deltaR, areaWeight) records into v3 layout.

    ``bin_pixel_lists[b]`` is a list of indices into ``pxList_records`` that
    belong to bin ``b``. Records are emitted in caller-provided order so that
    each bin's entries form a contiguous slab — that's what ``offset`` points
    into.
    """
    pxList_records = list(pxList_records)
    counts = np.zeros(len(bin_pixel_lists), dtype=np.int32)
    offsets = np.zeros(len(bin_pixel_lists), dtype=np.int32)
    cursor = 0
    out_records = []
    for b, blist in enumerate(bin_pixel_lists):
        offsets[b] = cursor
        counts[b] = len(blist)
        for src in blist:
            out_records.append(pxList_records[src])
            cursor += 1

    arr = np.array(out_records, dtype=PXLIST_DTYPE) if out_records \
          else np.empty(0, dtype=PXLIST_DTYPE)

    hdr = MapHeader() if write_header else None
    write_map(map_path, nmap_path,
              pxList=arr, counts=counts, offsets=offsets, header=hdr)
