"""Compile-time constants from `FF_HEDM/src/IndexerOMP.c` / `IndexerGPU.cu`.

In Python these are advisory caps + bookkeeping (column counts).
Buffer sizes are determined dynamically at runtime, not from these.
"""

# --- Bookkeeping caps (matches C; advisory only) ---
MAX_N_SPOTS = 100_000_000
MAX_N_RINGS = 500
MAX_N_HKLS = 5000
MAX_N_STEPS = 2000
MAX_N_OMEGARANGES = 2000
MAX_N_OR_CPU = 36_000     # IndexerOMP.c
MAX_N_OR_GPU = 7_200      # IndexerGPU.cu

# --- Column counts of legacy flat layouts ---
N_COL_THEORSPOTS = 14   # TheorSpots row layout
N_COL_OBSSPOTS = 9      # Spots.bin row layout
N_COL_GRAINSPOTS = 17
N_COL_GRAINMATCHES = 16

# --- Numerical ---
EPS_F32 = 1e-9
EPS_F64 = 1e-12

# --- Conversions ---
import math

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
