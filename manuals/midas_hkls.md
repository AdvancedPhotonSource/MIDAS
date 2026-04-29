# midas-hkls — crystallography & differentiable structure factors

`midas-hkls` is the crystallographic backbone for MIDAS's intensity-aware
HEDM workflows. It provides:

1. **HKL generation** for all 230 space groups (sginfo-equivalent).
2. **CIF I/O** via gemmi (preferred) or pycifrw.
3. **Differentiable structure factors `F_hkl`** in PyTorch — gradients flow
   through atomic positions, occupancies, B-factors, lattice parameters, and
   wavelength (when anomalous corrections are enabled).
4. **Anomalous (resonant) scattering** corrections f', f'' (Cromer-Liberman).
5. **Powder intensity** with Lorentz-polarization weighting.

The package is the foundation for intensity-aware peak fitting in pf-HEDM
(`fwd_sim/`) and complements MIDAS's existing position-based pipelines.

## Installation

`midas-hkls` is a separately versioned package shipped under
`packages/midas_hkls/`. From the MIDAS root:

```bash
pip install -e packages/midas_hkls[all]      # gemmi + torch
```

or, more granular:

```bash
pip install -e packages/midas_hkls            # base: numpy only
pip install -e packages/midas_hkls[cif]       # + gemmi
pip install -e packages/midas_hkls[torch]     # + torch
```

## Module map

| Module | Purpose | Optional deps |
|---|---|---|
| `space_group.py`, `hall.py`, `symops.py`, `tables.py` | 230-SG catalog + symmetry ops | — |
| `lattice.py` | Direct/reciprocal metric, d-spacings (numpy) | — |
| `lattice_torch.py` | Same, torch | torch |
| `hkl_gen.py` | `generate_hkls()` | — |
| `crystal.py` | `Atom`, `Crystal`, unit-cell expansion | — |
| `crystal_torch.py` | `CrystalTensor` packaging for torch | torch |
| `form_factors.py` | Cromer-Mann (IT92) form factors | — |
| `structure_factor.py` | Differentiable F_hkl | torch |
| `intensity.py` | Powder intensity, Lorentz-polarization, Reflection enrichment | torch |
| `anomalous.py` | f', f'' lookup with log-energy interpolation | torch (optional) |
| `io/cif.py` | CIF reader/writer | gemmi (preferred) or pycifrw |

## End-to-end pf-HEDM workflow

```python
import torch
from midas_hkls import (
    read_cif, generate_hkls, intensity_from_crystal,
    attach_intensities,
)

# 1. Read the structure
xt = read_cif("CeO2.cif")

# 2. Pack into torch tensors with selected gradients
xt_t = xt.to_torch(requires_grad={
    "B_iso":   True,
    "lattice": True,
    "fract":   False,        # Ce/O are on special positions; lock them
    "occ":     False,
})

# 3. Generate the reflection list
refs = generate_hkls(xt.space_group, xt.lattice,
                     wavelength_A=0.173, two_theta_max_deg=20.0)

# 4. Compute F_hkl + powder intensity (anomalous OFF for high-energy HEDM)
F, I = intensity_from_crystal(xt_t, refs, wavelength_A=0.173,
                              polarization=0.5, anomalous=False)

# 5. Compare to observed; backprop drives the optimizer
loss = ((torch.log(I + 1e-3) - torch.log(I_obs + 1e-3)) ** 2).mean()
loss.backward()

# 6. Read off gradients and apply your favorite optimizer
print("∂loss/∂B(Ce, O):", xt_t.B_iso_asu.grad.tolist())
print("∂loss/∂a:        ", xt_t.lattice_params.grad[0].item())

# 7. After fitting, attach the final F/I back onto the Reflection list
F, I = intensity_from_crystal(xt_t, refs, wavelength_A=0.173)
enriched = attach_intensities(refs, F.detach(), I.detach())
```

## Anomalous scattering

For energies near absorption edges (LaueMatching with synchrotron beamlines,
Fe Kα ≈ 6.4 keV, Cu Kα ≈ 8 keV), enable `anomalous=True`:

```python
from midas_hkls import structure_factors

F = structure_factors(xt_t, hkls,
                      wavelength_A=1.5418, anomalous=True)
```

The Cromer-Liberman tables ship with the package (689 KB, 92 elements ×
401 log-spaced energies from 100 eV to 200 keV). At grid points the values
match `gemmi.cromer_liberman` exactly; off-grid uses linear interpolation in
`log(E)`. Gradients with respect to wavelength are well-defined.

For HEDM at 50–100 keV, anomalous corrections are typically <1 e⁻ and can
be omitted; for resonant studies near edges they're essential.

## Conventions

| Quantity | Unit | Notes |
|---|---|---|
| Lattice constants | Å | |
| Lattice angles | degrees | |
| Wavelength | Å | `E_eV = 12398.4 / λ_Å` |
| B-factor (isotropic) | Å² | B = 8π²U |
| ADP U_ij (anisotropic) | Å² | CIF fractional convention |
| HKL | integers | h, k, l |
| 2θ output | degrees (numpy path), radians (torch path) | |
| Phases | radians | F_hkl as complex |

## Validation summary

| Test | Reference | Tolerance |
|---|---|---|
| HKL ring count, d, 2θ, multiplicity | `GetHKLList` (sginfo) | exact |
| 230-SG Hall-symbol resolution | sginfo | exact |
| `f(s)` form factor at s=0 | atomic number | < 5% |
| `f(s)` at general s | gemmi IT92 | < 1e-3 abs |
| `|F|` for CeO₂, Si, LaB₆, α-Fe, calcite | gemmi (deduped) | < 1e-4 rel |
| `f'`, `f''` at grid energies | gemmi.cromer_liberman | < 1e-3 abs |
| `f'`, `f''` off-grid | gemmi.cromer_liberman | < 0.05 abs |
| Autograd gradcheck on `|F|²` w.r.t. lattice | finite-diff (float64) | atol 1e-4 |
| Autograd gradcheck on `|F|²` w.r.t. atom positions | finite-diff (float64) | atol 1e-4 |
| Synthetic B-factor refinement (CeO₂) | recovers truth | < 0.1 Å² |
| Synthetic lattice refinement (CeO₂) | recovers truth | < 0.005 Å |

Run the full suite with:

```bash
cd packages/midas_hkls
pytest -v
```

## Limitations & roadmap

`v0.2.0` covers the IAM (independent-atom model) with isotropic and
anisotropic ADPs. Out of scope (planned for later releases):

- Wyckoff-position constraints during refinement (currently only the general
  position is enforced; refining a special-position atom requires the user
  to constrain the gradient manually).
- Multipole / aspherical form factors.
- Ion form factors (only neutral atoms in the IT92 export).
- Magnetic structure factors.

## See also

- [`packages/midas_hkls/README.md`](../packages/midas_hkls/README.md) — package-level docs.
- [`fwd_sim/`](../fwd_sim/) — differentiable HEDM forward model that consumes `midas-hkls`.
- [`docs_generated/`](../docs_generated/) — auto-generated module-level docs.
