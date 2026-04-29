# peakfit_torch — PyTorch peak-fitting backend for FF-HEDM

`peakfit_torch` is a drop-in replacement for the C tool
`PeaksFittingOMPZarrRefactor`. It reads the same Zarr archive, applies the
same pre-processing, computes connected components and regional maxima the
same way, and writes the same two binary output files
(`AllPeaks_PS.bin`, `AllPeaks_PX.bin`).

The only difference is the optimizer:

- **C tool:** NLopt Nelder-Mead (derivative-free simplex), per-region, OpenMP-parallel.
- **peakfit_torch:** batched **Levenberg-Marquardt** (gradient-based,
  Gauss-Newton), CPU or CUDA, fp32 or fp64.

Output is **scientifically equivalent** (not bit-exact). LM and Nelder-Mead
land in slightly different minima within the same basin — fitted positions
match within 0.05 px, intensities within ~1%, sigmas within ~5%. Downstream
indexing produces equivalent grain orientations.

The C tool is kept as the validation oracle.

## Install

```bash
pip install -e packages/midas_peakfit[dev]
```

PyTorch with CUDA support must be installed separately. See
<https://pytorch.org/get-started/locally/> for the right wheel.

## Usage

### Direct CLI (drop-in for `PeaksFittingOMPZarrRefactor`)

```bash
peakfit_torch DataFile.MIDAS.zip 0 1 8                   # auto device, fp64
peakfit_torch DataFile.MIDAS.zip 0 1 8 OutputFolder 1    # override ResultFolder + fitPeaks
peakfit_torch DataFile.MIDAS.zip 0 1 8 \
    --device cuda --dtype float32 --batch-size 8192      # GPU + fp32
peakfit_torch DataFile.MIDAS.zip 0 1 8 \
    --validate-against /path/to/c_AllPeaks_PS.bin        # parity report
```

Positional arguments mirror the C tool exactly:
`DataFile blockNr nBlocks numProcs [ResultFolder] [fitPeaks]`.

### Through `ff_MIDAS.py` / `pf_MIDAS.py`

The workflow scripts honor an environment variable:

```bash
# Default (no env): use C tool (PeaksFittingOMPZarrRefactor)
python ff_MIDAS.py ...

# Use the PyTorch backend
MIDAS_PEAKFIT_BACKEND=torch python ff_MIDAS.py ...
```

### Through tests

Both `tests/test_ff_hedm.py` and `tests/test_pf_hedm.py` accept a `--backend`
flag that sets the env var:

```bash
python tests/test_ff_hedm.py -nCPUs 8 --backend torch
python tests/test_pf_hedm.py -nCPUs 8 --backend torch
```

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--device {cpu,cuda}` | `cuda` if available, else `cpu` | Compute device |
| `--dtype {float32,float64}` | `float64` | Numeric precision |
| `--batch-size N` | 4096 | Cross-frame region batch threshold (advisory) |
| `--validate-against PATH` | — | Compare to a C-produced `AllPeaks_PS.bin` |
| `--deterministic` | off | Force deterministic algorithms (fp64 only) |

## Output

Two binary files written to `{ResultFolder}/Temp/`:

- **`AllPeaks_PS.bin`** — peak summary, 29 columns × nPeaks per frame.
  Layout matches `WriteConsolidatedPeakFiles` in
  `FF_HEDM/src/PeaksFittingConsolidatedIO.h`.
- **`AllPeaks_PX.bin`** — pixel coordinates per peak.

These are byte-compatible with the C tool: existing downstream MIDAS tools
(indexer, refiner, MergeOverlappingPeaks, `UnpackConsolidatedPeaks.py`)
read them without modification.

## Internals

```
        Zarr archive
              │
              ▼
   ┌────────────────────────────┐
   │ params.py  + zarr_io.py    │   parse 40+ params, dtype dispatch
   └────────────────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │ panels.py  + geometry.py   │   tilt + distortion → goodCoords[N,N]
   └────────────────────────────┘
              │
              ▼
       (per-frame loop)
              │
              ▼
   ┌────────────────────────────┐
   │ preprocess.py              │   dark/flood/threshold/transform/transpose
   └────────────────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │ connected.py + seeds.py    │   8-conn CC; regional maxima; moment seeds
   └────────────────────────────┘
              │
              ▼ (regions, x0, lo, hi, z, R, η)
   ┌────────────────────────────┐
   │ fit.py (bucket dispatcher) │   group by (n_peaks, M_padded)
   └────────────────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │ model.py  → lm.py          │   Pseudo-Voigt forward + batched LM
   │   reparam.py (sigmoid)     │   bound enforcement
   │   adam_fallback.py         │   recovery for divergent fits
   └────────────────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │ postfit.py + output.py     │   29-col rows + AllPeaks_PS/PX.bin
   └────────────────────────────┘
```

The differentiable Pseudo-Voigt is the **factored** form (NOT
radial-symmetric):

```
G_j(r,η) = exp(-0.5 × [(r-R_j)²/σGR_j² + (η-η_j)²/σGEta_j²])
L_j(r,η) = 1 / [(1 + (r-R_j)²/σLR_j²) × (1 + (η-η_j)²/σLEta_j²)]
I(r,η)   = bg + Σ_j Imax_j × [μ_j × L_j + (1-μ_j) × G_j]
```

This matches the C objective at `PeaksFittingOMPZarrRefactor.c:760-767`
exactly.

## Parity gate

| Field | Tolerance |
|---|---|
| `nPeaks` per frame, pixel sets, `maxY/maxZ`, `maskTouched` | exact |
| `YCen, ZCen, Radius, diffY, diffZ` | ≤ 0.05 px |
| `Eta` | ≤ 0.02° |
| `IMax, IntegratedIntensity, RawSumIntensity` | ≤ 1% relative |
| `BG, SigmaR, SigmaEta, σGR, σLR, σGEta, σLEta, MU, FitRMSE` | ≤ 5–10% relative |

Use `--validate-against` to produce a parity report:

```bash
peakfit_torch dataset.MIDAS.zip 0 1 8 \
    --validate-against /path/to/c_output/AllPeaks_PS.bin
```

## GPU testing on alleppey

```bash
# On alleppey
cd ~/opt/MIDAS/packages/midas_peakfit
pip install -e .[dev]
pytest tests/ -v -m "gpu"          # CUDA-only tests
python tests/test_ff_hedm.py -nCPUs 8 --backend torch   # full FF parity
```

GPU-side runs use float64 by default. fp32 (~2× faster) is selectable but
loses up to 0.5 px on position recovery in extreme cases.

## Release

```bash
cd packages/midas_peakfit
./release.sh 0.1.1                  # local prepare
./release.sh 0.1.1 --dry-run        # build without committing
./release.sh 0.1.1 --publish        # full: tag, push, GitHub release, PyPI
```

See `packages/midas_peakfit/RELEASING.md` for full release docs.

## Limitations

- **σL drift (the Pseudo-Voigt G/L degeneracy):** when μ → 0 or μ → 1, the
  unused sigma is unidentifiable and may converge to its bound. This is a
  property of the model, not a bug. Position parameters are unaffected.
- **Connected-component labeling order may differ from C DFS:** SciPy's
  `ndimage.label` and C's iterative DFS partition pixels identically but
  number labels differently. Output `SpotID` may not match C's, but the
  fitted peak set is the same.
- **Determinism in fp32:** GPU fp32 ops are not deterministic by default;
  use `--dtype float64 --deterministic` for reproducible runs.
