"""Parameter parsers for midas-transforms.

Two input formats are supported:

1. **paramstest.txt** (used by ``SaveBinData`` and downstream by ``IndexerOMP``).
   Plain-text key-value pairs; ``RingNumbers`` / ``RingRadii`` repeat. Same
   format as ``midas-index``.

2. **Zarr archive** (used by ``MergeOverlappingPeaksAllZarr``,
   ``CalcRadiusAllZarr``, ``FitSetupZarr``).  Parameters live under
   ``analysis/process/analysis_parameters/<key>``; image data under
   ``exchange/data``. The full key inventory is in ``REQUIRED_FITSETUP_KEYS``
   and ``OPTIONAL_FITSETUP_KEYS`` below.

Per the implementation plan (§8 risk #6), missing required keys raise
``KeyError`` with the full list of required keys, never default silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# paramstest.txt
# ---------------------------------------------------------------------------


@dataclass
class ParamsTest:
    """A typed view of paramstest.txt that ``SaveBinData``/``IndexerOMP`` read."""

    Wavelength: float = 0.0
    Lsd: float = 0.0
    px: float = 200.0
    StepSizeOrient: float = 0.2
    MarginOme: float = 0.5
    MarginEta: float = 500.0
    MarginRad: float = 500.0
    MarginRadial: float = 500.0
    EtaBinSize: float = 0.1
    OmeBinSize: float = 0.1
    NoSaveAll: int = 0

    RingNumbers: List[int] = field(default_factory=list)
    RingRadii: List[float] = field(default_factory=list)
    OmegaRanges: List[Tuple[float, float]] = field(default_factory=list)
    BoxSizes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    LatticeConstant: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 90.0, 90.0, 90.0,
    )

    SpaceGroup: int = 225
    UseFriedelPairs: int = 1

    SpotsFileName: str = "InputAll.csv"
    IDsFileName: str = "SpotsToIndex.csv"
    OutputFolder: str = ""
    GrainsFileName: str = ""

    raw: Dict[str, Any] = field(default_factory=dict)

    def get_ring_radius(self, ring_number: int) -> float:
        """Return the radius for ``ring_number`` (or 0.0 if not configured)."""
        for r, rad in zip(self.RingNumbers, self.RingRadii):
            if r == ring_number:
                return rad
        return 0.0

    @property
    def highest_ring_no(self) -> int:
        return max(self.RingNumbers) if self.RingNumbers else 0


_FLOAT_KEYS = {
    "px", "Wavelength", "Lsd", "Distance",
    "StepSizeOrient", "StepsizeOrient", "StepSizePos", "StepsizePos",
    "MarginOme", "MarginEta", "MarginRad", "MarginRadius", "MarginRadial",
    "EtaBinSize", "OmeBinSize",
}
_INT_KEYS = {"NoSaveAll", "SpaceGroup", "UseFriedelPairs"}
_STR_KEYS = {"SpotsFileName", "IDsFileName", "OutputFolder", "GrainsFile"}


def read_paramstest(path: Union[str, Path]) -> ParamsTest:
    """Parse a paramstest.txt file into a ``ParamsTest`` dataclass."""
    p = ParamsTest()
    with open(path, "r") as fp:
        for raw in fp:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # The C FitSetupZarr writes each line with a trailing ';'
            # (FitSetupParamsAllZarr.c:1591-1631). Strip it before tokenising.
            line = line.rstrip(";")
            tokens = [t.rstrip(";") for t in line.split()]
            key, args = tokens[0], tokens[1:]
            p.raw[key] = args

            if key == "RingNumbers":
                p.RingNumbers.append(int(args[0]))
            elif key == "RingRadii":
                p.RingRadii.append(float(args[0]))
            elif key == "OmegaRange":
                p.OmegaRanges.append((float(args[0]), float(args[1])))
            elif key == "BoxSize":
                p.BoxSizes.append(
                    (float(args[0]), float(args[1]), float(args[2]), float(args[3]))
                )
            elif key in ("LatticeParameter", "LatticeConstant"):
                if len(args) >= 6:
                    p.LatticeConstant = (
                        float(args[0]), float(args[1]), float(args[2]),
                        float(args[3]), float(args[4]), float(args[5]),
                    )
                else:
                    a = float(args[0])
                    p.LatticeConstant = (a, a, a, 90.0, 90.0, 90.0)
            elif key == "GrainsFile":
                p.GrainsFileName = args[0]
            elif key in _FLOAT_KEYS:
                attr = "Lsd" if key in ("Lsd", "Distance") else (
                    "StepSizeOrient" if key in ("StepSizeOrient", "StepsizeOrient") else
                    ("MarginRad" if key == "MarginRadius" else key)
                )
                setattr(p, attr, float(args[0]))
            elif key in _INT_KEYS:
                setattr(p, key, int(args[0]))
            elif key in _STR_KEYS:
                attr = "GrainsFileName" if key == "GrainsFile" else key
                setattr(p, attr, args[0])
            # unknown keys are recorded in p.raw and otherwise ignored
    return p


def write_paramstest(p: ParamsTest, path: Union[str, Path]) -> None:
    """Write a ``ParamsTest`` back out as paramstest.txt (used by FitSetup)."""
    with open(path, "w") as fp:
        if p.Wavelength:
            fp.write(f"Wavelength {p.Wavelength}\n")
        if p.Lsd:
            fp.write(f"Lsd {p.Lsd}\n")
        if p.px:
            fp.write(f"px {p.px}\n")
        fp.write(f"StepSizeOrient {p.StepSizeOrient}\n")
        fp.write(f"MarginOme {p.MarginOme}\n")
        fp.write(f"MarginEta {p.MarginEta}\n")
        fp.write(f"MarginRad {p.MarginRad}\n")
        fp.write(f"EtaBinSize {p.EtaBinSize}\n")
        fp.write(f"OmeBinSize {p.OmeBinSize}\n")
        fp.write(f"NoSaveAll {p.NoSaveAll}\n")
        fp.write(f"SpaceGroup {p.SpaceGroup}\n")
        fp.write(f"UseFriedelPairs {p.UseFriedelPairs}\n")
        for r in p.RingNumbers:
            fp.write(f"RingNumbers {r}\n")
        for r in p.RingRadii:
            fp.write(f"RingRadii {r}\n")
        for omr in p.OmegaRanges:
            fp.write(f"OmegaRange {omr[0]} {omr[1]}\n")
        for bx in p.BoxSizes:
            fp.write(f"BoxSize {bx[0]} {bx[1]} {bx[2]} {bx[3]}\n")
        if p.LatticeConstant != (0.0, 0.0, 0.0, 90.0, 90.0, 90.0):
            fp.write("LatticeParameter " + " ".join(str(v) for v in p.LatticeConstant) + "\n")
        if p.OutputFolder:
            fp.write(f"OutputFolder {p.OutputFolder}\n")


# ---------------------------------------------------------------------------
# Zarr archive
# ---------------------------------------------------------------------------

# Per dev/implementation_plan.md §8 risk #6: full key inventory locked here.

REQUIRED_FITSETUP_KEYS = (
    "Lsd", "Wavelength", "PixelSize",
    "YCen", "ZCen",
    "RingThresh",
    "LatticeParameter",
    "tx", "ty", "tz",
)

OPTIONAL_FITSETUP_KEYS_FLOAT = (
    "Width", "WidthTthPx", "Hbeam", "Rsample", "BeamThickness",
    "RhoD", "MaxRingRad",
    "MarginRadius", "MarginRadial", "MarginEta", "MarginOme",
    "EtaBinSize", "OmeBinSize",
    "StepSizeOrient", "StepSizePos",
    "MargABG", "MargABC",
    "tolTilts", "tolBC", "tolLsd",
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
    "p8", "p9", "p10", "p11", "p12", "p13", "p14",
    "Wedge",
    "MinEta", "MinMatchesToAcceptFrac",
    "MaxOmeSpotIDsToIndex", "MinOmeSpotIDsToIndex",
    "WeightFitRMSE", "WeightMask",
    "tInt", "tGap",
)
OPTIONAL_FITSETUP_KEYS_INT = (
    "DoFit", "UseFriedelPairs", "OverallRingToIndex",
    "MaxNFrames", "SkipFrame", "LayerNr",
    "NPanelsY", "NPanelsZ", "PanelSizeY", "PanelSizeZ",
    "SpaceGroup",
)
OPTIONAL_FITSETUP_KEYS_STR = (
    "PanelShiftsFile", "ResidualCorrectionMap", "ResultFolder",
)
OPTIONAL_FITSETUP_KEYS_ARRAY = (
    "BoxSizes", "OmegaRanges", "PanelGapsY", "PanelGapsZ",
)


@dataclass
class ZarrParams:
    """Geometry + bookkeeping parameters parsed from a Zarr archive.

    Defaults match the C code's initial values
    (``FitSetupParamsAllZarr.c:644-664``).
    """

    # geometry (required)
    Lsd: float = 0.0
    Wavelength: float = 0.0
    PixelSize: float = 200.0
    YCen: float = 0.0
    ZCen: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    LatticeConstant: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 90.0, 90.0, 90.0,
    )

    # detector / sample
    Hbeam: float = 0.0
    Rsample: float = 0.0
    BeamThickness: float = 0.0
    RhoD: float = 0.0
    MaxRingRad: float = 0.0
    NrPixelsY: int = 0
    NrPixelsZ: int = 0
    NrPixels: int = 0
    EndNr: int = 0

    # ring filter
    RingThresh: List[Tuple[int, float]] = field(default_factory=list)
    Width: float = -1.0
    WidthOrig: float = -1.0

    # margins / bin sizes
    MarginRadius: float = 500.0
    MarginRadial: float = 500.0
    MarginEta: float = 500.0
    MarginOme: float = 0.5
    EtaBinSize: float = 0.1
    OmeBinSize: float = 0.1
    StepSizeOrient: float = 0.2
    StepSizePos: float = 5.0
    MargABG: float = 2.0
    MargABC: float = 2.0

    # tolerances (NLopt-era; carried through for paramstest.txt parity)
    tolTilts: float = 1.0
    tolBC: float = 1.0
    tolLsd: float = 5000.0

    # distortion polynomial
    p0: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    p3: float = 0.0
    p4: float = 0.0
    p5: float = 0.0
    p6: float = 0.0
    p7: float = 0.0
    p8: float = 0.0
    p9: float = 0.0
    p10: float = 0.0
    p11: float = 0.0
    p12: float = 0.0
    p13: float = 0.0
    p14: float = 0.0

    # other
    Wedge: float = 0.0
    MinEta: float = 0.0
    MinMatchesToAcceptFrac: float = 0.0
    MaxOmeSpotIDsToIndex: float = 0.0
    MinOmeSpotIDsToIndex: float = 0.0
    WeightFitRMSE: float = 0.0
    WeightMask: float = 1.0
    tInt: float = 1.0
    tGap: float = 0.0

    DoFit: int = 0
    UseFriedelPairs: int = 1
    OverallRingToIndex: int = 0
    MaxNFrames: int = 100000
    SkipFrame: int = 0
    LayerNr: int = 0
    SpaceGroup: int = 225

    # panels
    NPanelsY: int = 0
    NPanelsZ: int = 0
    PanelSizeY: int = 0
    PanelSizeZ: int = 0
    PanelGapsY: List[int] = field(default_factory=list)
    PanelGapsZ: List[int] = field(default_factory=list)
    PanelShiftsFile: str = ""

    # files
    ResidualCorrectionMap: str = ""
    ResultFolder: str = ""

    # arrays
    BoxSizes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    OmegaRanges: List[Tuple[float, float]] = field(default_factory=list)

    # scan
    OmegaStep: float = 0.0

    # merge-specific (also in some Zarr archives)
    # C default ``MarginOmegaOverlap = sqrt(4) = 2.0``
    # (MergeOverlappingPeaksAllZarr.c:524).
    OverlapLength: float = 2.0
    UsePixelOverlap: int = 0
    UseMaximaPositions: int = 0

    # raw passthrough for any keys we didn't enumerate
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.RhoD and not self.MaxRingRad:
            self.MaxRingRad = self.RhoD
        if self.MaxRingRad and not self.RhoD:
            self.RhoD = self.MaxRingRad
        if self.Width == -1.0 and self.WidthOrig != -1.0:
            self.Width = self.WidthOrig

    def to_paramstest(self) -> ParamsTest:
        """Convert to a ``ParamsTest`` view (for downstream binning / indexing)."""
        pt = ParamsTest()
        pt.Wavelength = self.Wavelength
        pt.Lsd = self.Lsd
        pt.px = self.PixelSize
        pt.StepSizeOrient = self.StepSizeOrient
        pt.MarginOme = self.MarginOme
        pt.MarginEta = self.MarginEta
        pt.MarginRad = self.MarginRadius
        pt.EtaBinSize = self.EtaBinSize
        pt.OmeBinSize = self.OmeBinSize
        pt.SpaceGroup = self.SpaceGroup
        pt.UseFriedelPairs = self.UseFriedelPairs
        pt.LatticeConstant = self.LatticeConstant
        pt.OmegaRanges = list(self.OmegaRanges)
        pt.BoxSizes = list(self.BoxSizes)
        pt.RingNumbers = [int(rn) for (rn, _) in self.RingThresh]
        pt.OutputFolder = self.ResultFolder
        return pt


def read_zarr_params(zarr_path: Union[str, Path]) -> ZarrParams:
    """Read a MIDAS Zarr archive (typically a .zip) into a ``ZarrParams``.

    Validates required keys and raises ``KeyError`` (per implementation plan
    §8 risk #6) with the full list of required keys when any are missing.
    """
    import zarr

    store = zarr.ZipStore(str(zarr_path), mode="r")
    try:
        root = zarr.group(store=store)
    finally:
        # we keep store open for the lifetime of the read (we don't close
        # until after reading); zarr.group returns a view that needs the store
        pass

    p = ZarrParams()
    seen_required: set = set()

    # Walk all known keys.
    ap_path = "analysis/process/analysis_parameters"

    def _read(key: str, dtype: type, allow_missing: bool = True):
        full = f"{ap_path}/{key}"
        try:
            arr = root[full]
        except KeyError:
            if not allow_missing:
                raise
            return None
        if dtype is str:
            data = arr[...]
            if data.dtype.kind in ("S", "U"):
                val = data.flat[0]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                return val
            return None
        return np.asarray(arr[...]).flatten()

    # Required scalar floats / ints
    for key in ("Lsd", "Wavelength", "PixelSize", "YCen", "ZCen", "tx", "ty", "tz"):
        v = _read(key, float, allow_missing=True)
        if v is not None and len(v) > 0:
            setattr(p, key, float(v[0]))
            seen_required.add(key)

    # LatticeParameter (6-vector)
    v = _read("LatticeParameter", float, allow_missing=True)
    if v is not None and len(v) >= 6:
        p.LatticeConstant = tuple(float(x) for x in v[:6])  # type: ignore[assignment]
        seen_required.add("LatticeParameter")

    # RingThresh: shape (N, 2) — (ring_number, threshold)
    try:
        rt = root[f"{ap_path}/RingThresh"][...]
        rt = np.asarray(rt).reshape(-1, 2)
        p.RingThresh = [(int(r[0]), float(r[1])) for r in rt]
        seen_required.add("RingThresh")
    except KeyError:
        pass

    # Optional scalar floats
    for key in OPTIONAL_FITSETUP_KEYS_FLOAT:
        v = _read(key, float)
        if v is not None and len(v) > 0:
            setattr(p, key, float(v[0]))

    # Optional scalar ints
    for key in OPTIONAL_FITSETUP_KEYS_INT:
        v = _read(key, int)
        if v is not None and len(v) > 0:
            setattr(p, key, int(v[0]))

    # Optional strings
    for key in OPTIONAL_FITSETUP_KEYS_STR:
        s = _read(key, str)
        if s is not None:
            setattr(p, key, s)

    # Optional 2-D arrays
    try:
        bs = np.asarray(root[f"{ap_path}/BoxSizes"][...]).reshape(-1, 4)
        p.BoxSizes = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in bs]
    except KeyError:
        pass
    try:
        oms = np.asarray(root[f"{ap_path}/OmegaRanges"][...]).reshape(-1, 2)
        p.OmegaRanges = [(float(r[0]), float(r[1])) for r in oms]
    except KeyError:
        pass
    try:
        gy = np.asarray(root[f"{ap_path}/PanelGapsY"][...]).flatten()
        p.PanelGapsY = [int(x) for x in gy]
    except KeyError:
        pass
    try:
        gz = np.asarray(root[f"{ap_path}/PanelGapsZ"][...]).flatten()
        p.PanelGapsZ = [int(x) for x in gz]
    except KeyError:
        pass

    # Image stack shape: exchange/data is (N_frames, NrPixelsZ, NrPixelsY)
    try:
        shape = root["exchange/data"].shape
        p.EndNr = int(shape[0])
        p.NrPixelsZ = int(shape[1])
        p.NrPixelsY = int(shape[2])
        p.NrPixels = max(p.NrPixelsY, p.NrPixelsZ)
    except KeyError:
        pass

    # OmegaStep
    try:
        v = root["measurement/process/scan_parameters/step"][...]
        p.OmegaStep = float(np.asarray(v).flat[0])
    except KeyError:
        pass

    # Validate required keys.
    missing = [k for k in REQUIRED_FITSETUP_KEYS if k not in seen_required]
    if missing:
        raise KeyError(
            f"Required Zarr keys missing from {zarr_path}: {missing}. "
            f"Full required list: {list(REQUIRED_FITSETUP_KEYS)}"
        )

    p.__post_init__()  # propagate RhoD<->MaxRingRad, Width<->WidthOrig
    return p
