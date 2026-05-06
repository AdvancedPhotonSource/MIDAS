# midas-suite

A meta-package that installs the MIDAS Python pipeline in a single command.

```bash
pip install midas-suite
```

This pulls in the currently published MIDAS sub-packages — the FF/NF
HEDM analysis chain, calibration, peak fitting, radial integration,
forward model, indexing, transforms, grain processing, and stress/strain
analysis.

`midas-suite` itself contains no scientific code. It's a thin
[meta-package](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package)
whose only job is to declare the sub-packages as dependencies so users
don't have to install them one at a time.

## What you get

`pip install midas-suite` installs **16 sub-packages** (as of v0.1.0):

**Top-level orchestrators (the entry points most users want):**

| Sub-package | Role |
|---|---|
| `midas-ff-pipeline` | End-to-end pure-Python FF-HEDM workflow orchestrator (1-N detectors). Drives the full FF pipeline from raw data to grains. |
| `midas-nf-pipeline` | Pure-Python NF-HEDM pipeline orchestrator (single + multi-resolution, multi-layer). Drop-in for `nf_MIDAS.py` / `nf_MIDAS_Multiple_Resolutions.py`. |
| `midas-parsl-configs` | Bundled + user-extensible Parsl configs for running MIDAS pipelines on laptops, workstations, clusters. |

**FF-HEDM building blocks:**

| Sub-package | Role |
|---|---|
| `midas-peakfit` | Differentiable PyTorch peak fitting for FF-HEDM Zarr |
| `midas-transforms` | FF-HEDM peak transforms (merge / radius / fit-setup / save-bin) |
| `midas-index` | Pure-Python/PyTorch FF-HEDM indexer (drop-in for `IndexerOMP`) |
| `midas-fit-grain` | Single/multi-grain refiner |
| `midas-process-grains` | FF-HEDM grain-determination + strain pipeline |

**NF-HEDM building blocks:**

| Sub-package | Role |
|---|---|
| `midas-nf-preprocess` | NF-HEDM preprocessing (hex grid, tomo filter, spot prediction) |
| `midas-nf-fitorientation` | NF-HEDM orientation/calibration fitter |

**Shared foundations:**

| Sub-package | Role |
|---|---|
| `midas-stress` | Crystallographic stress/strain analysis (Voigt-Mandel, Cij inversion, slip/Schmid/Taylor) |
| `midas-params` | Parameter-file registry, validator, wizard for FF/NF/PF/RI |
| `midas-hkls` | Pure-Python crystallography & HKL list generator (sginfo-equivalent) |
| `midas-diffract` | End-to-end differentiable HEDM forward model (FF + NF + pf-HEDM) |
| `midas-integrate` | Pure-Python radial integration (`DetectorMapper` + CSR + streaming server) |
| `midas-calibrate` | Native Python/Torch detector geometry calibration (LM-based) |

You then `import midas_stress`, `import midas_diffract`, etc. directly —
each sub-package retains its own API. `midas-suite` does not re-export
them.

To check what was installed:

```python
import midas_suite
print(midas_suite.installed())
```

## Modality bundles

If you don't want everything, the optional extras let you pick a workflow:

```bash
pip install "midas-suite[ff]"      # FF-HEDM stack
pip install "midas-suite[nf]"      # NF-HEDM stack
pip install "midas-suite[calib]"   # calibration + integration only
pip install "midas-suite[ff,plots]"
```

| Extra | What it pulls |
|---|---|
| `ff` | `midas-ff-pipeline` (which transitively pulls hkls, peakfit, transforms, index, fit-grain, process-grains, diffract, parsl-configs) + stress, params, calibrate, integrate |
| `nf` | `midas-nf-pipeline` (which transitively pulls hkls, stress, nf-preprocess, nf-fitorientation) + params |
| `calib` | hkls, integrate, peakfit, calibrate |
| `plots` | matplotlib (for sub-package plotting helpers) |

## What `pip install midas-suite` does NOT include

Be aware:

- **The MIDAS C executables** (`IndexerOMP`, `ProcessGrains`, `MakeDiffrSpots`, …)
  still need to be built from source via `cmake --build .` from the MIDAS
  monorepo. The pure-Python pipeline (calibration → integration → indexing
  → grain processing) is now end-to-end in PyTorch and does not require
  them.
- **The PyQt FF viewer GUI** needs `PyQt5` or `PySide6` installed
  separately. Not declared here because it's optional and platform-sensitive.
- **Optional crystallography backends** for `midas-hkls`: install
  `gemmi` or `pycifrw` separately for CIF I/O via `pip install
  midas-hkls[cif]`.
- **GPU acceleration** is a runtime backend selected by PyTorch device
  string. CUDA/MPS just work if your `torch` install supports them; no
  separate `*-gpu` package needed.
- **Coming in v0.2.0** — `midas-grain-odf` (per-grain ODF inversion) and
  `midas-pf-odf` (joint pf-HEDM peak-shape inversion) are not yet on
  PyPI. They fold in when `midas-suite` v0.2.0 ships, bringing the
  bundled count to 18 sub-packages.

## Cross-platform

All MIDAS Python sub-packages are pure Python or PyTorch and ship as
`py3-none-any` wheels. Tested install paths: Linux, macOS, Windows.
See [`packages/RELEASE_READINESS.md`](../RELEASE_READINESS.md) for the
detailed cross-platform readiness matrix.

## Versioning

`midas-suite` versions are independent of the sub-package versions.
The rule:

| Change | Bump |
|---|---|
| Floors tightened (no new sub-package added) | patch (`0.1.0` → `0.1.1`) |
| New sub-package added to the dep list | minor (`0.1.0` → `0.2.0`) |
| Backwards-incompatible reorganisation of bundles | major (`0.x.y` → `1.0.0`) |

Floors are pinned with `>=`, never `==`, so a sub-package patch release
doesn't break `midas-suite`.

## Releasing a new version

See [`RELEASING.md`](RELEASING.md) for the full release flow. TL;DR:

```bash
cd packages/midas_suite
./release.sh 0.2.0 --publish
```

## License

BSD-3-Clause, same as the sub-packages.
