# MIDAS Detector Calibration

Two engines are available:

1. **C engine (default).** `AutoCalibrateZarr.py` → `CalibrantIntegratorOMP` → `CalibrationCore` (NLopt SBPLX/Nelder-Mead). The proven path.
2. **Python engine (new, v0.1.0).** Native PyTorch implementation that replaces NLopt with batched Levenberg-Marquardt and runs on CPU or GPU. Built from three packages:
   - **`midas-hkls`** — pure-Python crystallography & HKL list generator. sginfo-equivalent (Hall-symbol parser, all 230 space groups). Replaces `GetHKLList`.
   - **`midas-peakfit`** (extended) — exposes a callback-based generic `lm_solve_generic` and a Schur-complement `lm_solve_arrowhead`.
   - **`midas-calibrate`** — the calibration package. Uses `midas-integrate`'s CSR pipeline for E-step integration, `midas-peakfit`'s LM for M-step refinement.

## Choosing an engine

```bash
# Drop-in C path (unchanged behavior)
python AutoCalibrateZarr.py --data ceo2.tif --params calib.txt

# Python path (new)
python AutoCalibrateZarr.py --data ceo2.tif --params calib.txt --engine python
```

Or invoke the Python pipeline directly:

```bash
midas-autocalibrate calib.txt --image ceo2.tif --output calib_refined.txt
```

## When to use the Python engine

- Need GPU acceleration (set `Device cuda` in the params file or pass via the `--device` flag in `midas-autocalibrate`).
- Want fp32 throughput on H100 (TF32 matmul path; set `Dtype fp32`).
- Want differentiable parameter exposure for downstream research (every geometry parameter is autograd-traced).
- Want a smaller dependency footprint — no NLopt, no sginfo C library.
- Want byte-compatible refined-parameter output that `MakeDiffrSpots` and the integrator consume unchanged.

The C engine remains the parity reference.

## Parameter file

The Python engine consumes the same `.txt` parameter file format as the C engine, plus a few new keys:

| Key | Meaning | Default |
|---|---|---|
| `Engine` | `"alternating"` (v0.1) or `"joint"` (delegates to alternating in v0.1) | `joint` |
| `Device` | `auto` / `cpu` / `cuda` / `mps` | `auto` |
| `Dtype` | `fp32` / `fp64` | `fp64` |
| `Loss` | `L2` / `L1` / `huber` | `L2` |
| `HuberDelta` | Huber transition (when `Loss=huber`) | `1.0` |
| `RBinSize` | Sub-pixel R bin size (px) | `0.25` |
| `EtaBinSize` | Eta bin size (deg) | `5.0` |
| `Width` | Per-ring R window (μm) | `800.0` |

All distortion parameters (`p0`–`p14`), tilts (`tx ty tz`), beam center (`BC`), `Lsd`, `Wavelength`, `Parallax`, `LatticeConstant`, `SpaceGroup`, etc. are read identically to the C path.

## Architecture

```
midas-hkls (no MIDAS deps)
   ↓
midas-detector-core (deferred — currently part of midas-integrate)
   ↓
midas-integrate            midas-peakfit (LM solver)
       ↘                ↙
        midas-calibrate
            ↓
        AutoCalibrateZarr.py --engine python
```

The single source of truth for detector geometry lives in `midas-integrate/midas_integrate/geometry.py`. The torch path in `midas_calibrate/geometry_torch.py` is byte-for-byte equivalent (verified by parity tests).

## Tests

```bash
cd ~/opt/MIDAS/packages/midas_hkls && pytest -q          # 251 tests
cd ~/opt/MIDAS/packages/midas_peakfit && pytest -q       # 29+ tests
cd ~/opt/MIDAS/packages/midas_calibrate && pytest -q     # 2 tests (M-step + e2e synthetic)
```

The `midas-hkls` test suite includes byte-level parity against the C `GetHKLList` for CeO₂, LaB₆, Si, α-Fe, α-Ti, calcite, Pnma, P21/c — same ring count, d-spacing, 2θ, multiplicity.

The `midas-calibrate` synthetic test forward-simulates a CeO₂ calibrant image at known geometry, perturbs the seed, and verifies the alternating engine recovers the truth (Lsd within 200μm, BC within 1px, tilts within 0.05°, mean strain ≤ 50μϵ).

## v0.1 Limitations (planned for v0.2)

- **Joint differentiable engine** (§13 of the implementation plan). Currently `Engine=joint` delegates to the alternating engine. Forward model + arrowhead-LM wiring is the v0.2 deliverable.
- **Per-panel calibration.** API hooks present (`PerPanelLsd`, `PerPanelDistortion`, `FixedPanelID`); not yet exercised end-to-end.
- **Pseudo-Voigt radial peak fit in the E-step.** Currently uses weighted centroid. Switching to `lm_solve_generic` per-(ring, η) is a drop-in upgrade.
- **`midas-detector-core` extraction.** Geometry/CSR/integration code lives in `midas-integrate` for now; the architectural split into a shared core package is a follow-up refactor with no behavior change.
- **Doublet detection.** Wide rings under significant tilt may produce two peaks within one ring window; v0.1 picks one centroid.
- **Adaptive η bins.** Uniform η binning only.

See `~/opt/MIDAS/calibrate_torch_implementation_plan.md` for the full design and roadmap.

## Conventions (preserved across both engines)

| Quantity | Units / convention |
|---|---|
| `Lsd` | μm, positive |
| `BC` | pixels |
| Tilts (`tx`, `ty`, `tz`) | degrees, **TRs = Rx(tx) · Ry(ty) · Rz(tz)** order |
| Eta | degrees, atan2(−Y′, Z′), [−180, 180) |
| `R_ideal` | `Lsd · tan(2θ) / px`, in pixels |
| Strain residual | `1 − R_obs / R_ideal` (signed) |
| Wavelength | Å |
| Lattice constants | Å, angles in degrees |
