# Implementation Plan — `midas-diffract` v0.1.0

**Status:** scaffolding for the Differentiable HEDM Forward Model paper companion package.
**Target PyPI release:** within the Day 8–12 co-author review window of the paper timeline.

---

## 1. Scope (v0.1.0 — MVP)

Ship exactly what the paper demonstrates. Nothing more.

**IN SCOPE:**
- `HEDMForwardModel` class — FF geometry, differentiable, pixel-exact against C.
- `NFHEDMForwardModel` class — NF geometry, differentiable, pixel-exact against C.
- `PFForwardModel` class — pf-HEDM geometry, differentiable forward only.
- Shared `HEDMGeometry` dataclass.
- Shared reciprocal-space utilities (rotation, diffraction condition, detector projection).
- Loss function library (re-exported from existing `hedm_losses.py`).
- Minimal single-grain optimization utility (`optimize_single_grain`) demonstrated in the paper.
- Zenodo DOI on release.

**OUT OF SCOPE for v0.1.0:** every item from the paper's OUT OF SCOPE list. No EM, no sub-voxel, no CPFEM, no streaming, no multi-grain optimization, no Bayesian wrappers. Each of those is a v0.2+ candidate after the paper lands.

## 2. Source migration (existing code → package)

Existing code in `fwd_sim/` that becomes the package:

| Source | Destination | Lines | Keep/Refactor |
|--------|-------------|-------|---------------|
| `fwd_sim/hedm_forward.py` | `midas_diffract/forward/ff.py` | 1299 | Keep; split FF/NF/pf if needed |
| `fwd_sim/nfhedm.py` | `midas_diffract/forward/nf.py` | 300 | Keep |
| `fwd_sim/hedm_losses.py` | `midas_diffract/losses.py` | 494 | Keep |
| `fwd_sim/single_grain_optimization_ff.py` | `midas_diffract/optimize.py` | 304 | Refactor into clean API |
| `fwd_sim/verify_pf_fwdsim.py` | `midas_diffract/forward/pf.py` | 360 | Extract pf forward as class |
| `fwd_sim/tests/test_c_comparison.py` | `tests/test_c_comparison.py` | 562 | Keep as integration test |
| `fwd_sim/tests/test_hedm_forward.py` | `tests/test_forward.py` | 1093 | Keep |
| `fwd_sim/tests/test_hedm_losses.py` | `tests/test_losses.py` | 392 | Keep |

Excluded from package (belong to other papers):
- `fwd_sim/em_spot_ownership.py`, `fwd_sim/diagnose_em_matching.py`, `fwd_sim/tests/test_em_spot_ownership.py` — EM paper.
- `fwd_sim/compare_recon_methods.py` — diagnostic only.

## 3. Dependencies

Pure Python + PyTorch. No C build. D1 difficulty.

- `numpy>=1.22`
- `scipy>=1.8` (only if used in optimize.py)
- `torch>=2.0`
- optional: `matplotlib` for example notebooks

## 4. Package structure

```
midas-diffract/
  pyproject.toml
  README.md
  LICENSE (BSD-3)
  midas_diffract/
    __init__.py          # version, top-level re-exports
    geometry.py          # HEDMGeometry dataclass
    forward/
      __init__.py
      ff.py              # HEDMForwardModel (FF)
      nf.py              # NFHEDMForwardModel
      pf.py              # PFForwardModel
      _common.py         # shared reciprocal-space utilities
    losses.py            # re-export from hedm_losses
    optimize.py          # optimize_single_grain convenience API
  tests/
    test_forward.py
    test_c_comparison.py  # skipped if C binaries not present
    test_losses.py
  examples/
    01_ff_pixel_exact.ipynb
    02_nf_pixel_exact.ipynb
    03_pf_forward.ipynb
    04_single_grain_optimization.ipynb
  RELEASING.md            # copy from midas_stress pattern
  release.sh              # copy from midas_stress pattern
```

## 5. API design (v0.1.0 — kept minimal)

```python
import midas_diffract as md
import torch

# FF forward simulation
geom = md.HEDMGeometry.from_parameters_file("Parameters_Cu.txt")
model = md.HEDMForwardModel(geom, hkls=hkls, thetas=thetas)

# Differentiable forward: gradient flows through all of these
orientation = torch.tensor(orient_matrix, requires_grad=True)
strain      = torch.zeros(3, 3, requires_grad=True)
position    = torch.zeros(3,    requires_grad=True)

spots = model(orientation=orientation, strain=strain, position=position)
# spots: (N, 6) — omega, y_pixel, z_pixel, eta, ring, intensity

# Joint optimization (convenience wrapper)
result = md.optimize_single_grain(
    model,
    observed_spots=observed,
    init=dict(orientation=orient_init, strain=torch.zeros(3,3), position=torch.zeros(3)),
    loss=md.losses.spot_position_loss,
    n_iter=500,
    lr=1e-3,
)

# NF forward
nf_model = md.NFHEDMForwardModel(geom, ...)
image    = nf_model(orientations_per_voxel=...)
# image: (H, W) differentiable detector image

# pf forward (forward only — no optimization demo in this paper)
pf_model = md.PFForwardModel(geom, ...)
```

## 6. Timeline

Work happens in parallel with paper drafting:

| Day | Work |
|-----|------|
| 1 (today) | Scaffolding (this plan, folder skeleton, pyproject.toml) |
| 2 | Migrate `hedm_forward.py` + `hedm_losses.py`; port tests |
| 3 | Migrate `nfhedm.py`, extract pf forward; split into `forward/{ff,nf,pf}.py` |
| 4 | `optimize.py` — clean single-grain API; example notebook 04 |
| 5 | Example notebooks 01–03 (FF, NF, pf pixel-exact demos) |
| 6 | README + RELEASING.md; cibuild wheels (CPU only — no C deps, trivial) |
| 7–10 | Hold during co-author paper review window |
| 11 | PyPI release v0.1.0 concurrent with preprint + submission |

## 7. Release checklist (reuses midas_stress pattern)

1. `release.sh 0.1.0 --dry-run` (build + test)
2. Tag annotated: `midas-diffract-v0.1.0`
3. GitHub release with wheels
4. PyPI via trusted publishing (OIDC) on tag push
5. Zenodo DOI auto-minted via GitHub↔Zenodo integration
6. Update paper software-availability section with Zenodo DOI before submitting

## 8. What this package is NOT

- Not a full MIDAS replacement.
- Not a reconstruction tool. Forward simulation + gradient-based single-grain refinement only.
- Not multi-grain optimized in v0.1.0.
- Not a substitute for the C simulators (they remain the reference; this package is *validated against* them).

Anything broader is v0.2.0+ after the paper lands.
