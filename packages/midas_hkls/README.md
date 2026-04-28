# midas-hkls

Pure-Python crystallography & HKL list generator. **sginfo-equivalent**: all 230
space groups via Hall-symbol parsing, no C dependencies at runtime.

## What it provides

- `SpaceGroup` — load by number, Hermann-Mauguin symbol, or Hall symbol; expose
  symmetry operations, systematic absences, equivalent reflections, multiplicities,
  Laue class, centering.
- `Lattice` — direct/reciprocal metric tensors, d-spacings, Bragg 2θ, with
  per-crystal-system convenience constructors.
- `generate_hkls()` — enumerate Laue-unique allowed reflections within a
  d-spacing or 2θ cutoff, sorted by d-descending, with multiplicities.
- CLI: `midas-hkls gen|info|list` (drop-in for `GetHKLList`).

## Quick start

```python
from midas_hkls import SpaceGroup, Lattice, generate_hkls

sg  = SpaceGroup.from_number(225)              # CeO₂ / Cu / Au / NaCl  (Fm-3m)
lat = Lattice.for_system("cubic", a=5.411)     # Å
refs = generate_hkls(sg, lat, wavelength_A=0.173, two_theta_max_deg=15.0)

for r in refs:
    print(r.ring_nr, (r.h, r.k, r.l), r.d_spacing, r.two_theta_deg, r.multiplicity)
```

## CLI

```
midas-hkls gen --sg 225 --lat 5.411 5.411 5.411 90 90 90 --wavelength 0.173 \
               --two-theta-max 15.0 -o ceo2.csv
midas-hkls info --sg "Fm-3m" --ops
midas-hkls list
```

## Parity with sginfo C library

`midas-hkls` is parity-tested byte-for-byte against MIDAS's `GetHKLList` (sginfo):

- Ring count, ring d-spacing, ring 2θ, ring multiplicity match exactly across
  CeO₂, LaB₆, Si, α-Fe, α-Ti, calcite, Pnma, P21/c.
- All 230 space groups parse without error and have correct Friedel-corrected
  Laue-class group orders.

Run `pytest` to exercise the parity matrix.

## Conventions

- Lattice constants in Å; angles in degrees.
- Wavelengths in Å.
- Symmetry operations stored as integer Seitz matrices over translation base
  STBF=12 (so 1/2 → 6, 1/3 → 4, 1/4 → 3, etc.) — exact-arithmetic absence
  detection, no float fuzz.
- Equivalent HKLs include Friedel pairs (centric structure factor under X-ray
  Laue symmetry).

## Roadmap (post v0.1.0)

- CIF reader (`io/cif.py`).
- Atomic form factors (Cromer-Mann) and structure factors.
- Anomalous scattering (Henke / Cromer tables).
- Wyckoff positions tables (currently only general position is exposed).
- Origin-choice and alternate-setting transformations beyond what sginfo's
  `extension` field provides.

## Origin

The 530-entry Hall-symbol table is extracted verbatim from sginfo
(© 1994-96 Ralf W. Grosse-Kunstleve, public domain) so that midas-hkls and
MIDAS's existing C tools resolve the same standard setting for every space
group.
