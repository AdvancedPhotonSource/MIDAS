"""IndexerParams dataclass.

Mirrors `struct TParams` from `FF_HEDM/src/IndexerOMP.c:119`. All keys parsed
by `ReadParams` (lines 1281-1547) map to fields here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IndexerParams:
    """All paramstest.txt fields consumed by the indexer.

    Aliases (from C): Distance/Lsd, LatticeParameter/LatticeConstant,
    MarginRadius/MarginRad, Completeness/MinMatchesToAcceptFrac,
    StepsizeOrient/StepSizeOrient, MinEta/ExcludePoleAngle. The reader
    folds aliases onto the canonical field below.

    `BigDetSize` is parsed for backward compat but ignored (deprecated).
    """

    # --- Geometry ---
    px: float = 0.0
    Distance: float = 0.0          # also "Lsd"
    Wavelength: float = 0.0
    Rsample: float = 0.0
    Hbeam: float = 0.0

    # --- Crystal ---
    SpaceGroup: int = 0
    LatticeConstant: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 90.0, 90.0, 90.0,
    )                                # also "LatticeParameter"

    # --- Grid stepping ---
    StepsizePos: float = 0.0
    StepsizeOrient: float = 0.0      # also "StepSizeOrient"

    # --- Margins ---
    MarginOme: float = 0.0
    MarginRad: float = 0.0           # also "MarginRadius"
    MarginRadial: float = 0.0
    MarginEta: float = 0.0
    EtaBinSize: float = 0.0
    OmeBinSize: float = 0.0
    ExcludePoleAngle: float = 0.0    # also "MinEta" — same field in C
    MinMatchesToAcceptFrac: float = 0.0  # also "Completeness"

    # --- Rings / detectors ---
    RingNumbers: list[int] = field(default_factory=list)
    RingsToReject: list[int] = field(default_factory=list)
    # RingRadii is sparse-by-ring-index: RingRadii[ring_nr] -> radius (um).
    # Built from parallel parse-time lists RingNumbers + RingRadiiUser.
    RingRadii: dict[int, float] = field(default_factory=dict)

    # --- Omega / box ranges (parallel lists) ---
    OmegaRanges: list[tuple[float, float]] = field(default_factory=list)
    BoxSizes: list[tuple[float, float, float, float]] = field(default_factory=list)

    # --- Mode A vs B ---
    UseFriedelPairs: int = 0  # 0 | 1 | 2
    isGrainsInput: bool = False
    GrainsFileName: str = ""
    SpotsFileName: str = "Spots.bin"
    IDsFileName: str = "IDsHash.csv"
    OutputFolder: str = "."

    def get_ring_radius(self, ring_nr: int) -> float:
        """Sparse lookup mirroring `Params.RingRadii[ring_nr]` in C."""
        return self.RingRadii.get(ring_nr, 0.0)

    def highest_ring_nr(self) -> int:
        """Mirrors `HighestRingNo` computation in IndexerOMP.c:2238."""
        return max(self.RingRadii.keys(), default=0)
