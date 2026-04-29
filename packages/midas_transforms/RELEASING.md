# Releasing midas-transforms

Same workflow as `midas-index` — see `release.sh` for the full script.

## Quick reference

```bash
cd packages/midas_transforms
./release.sh <new_version> [--publish | --dry-run]
```

| Command | What it does |
|---------|--------------|
| `./release.sh 0.1.0` | Prepare locally (version bump + tests + build + commit + tag). Push/publish manually. |
| `./release.sh 0.1.0 --publish` | Prepare + push + GitHub release. CI auto-publishes to PyPI. |
| `./release.sh 0.1.0 --dry-run` | Prepare without committing or tagging. |

## Pre-release checklist

Before bumping out of `0.1.0.dev0`:

- [ ] `pytest` passes (unit + smoke + e2e on a small fixture).
- [ ] Byte-level regression vs C `SaveBinData`, `CalcRadiusAllZarr`,
      `MergeOverlappingPeaksAllZarr`, `FitSetupZarr` checked in
      `tests/test_regression_vs_c.py`.
- [ ] CUDA regression on copland/alleppey passes at float32 (rel-tol 1e-5).
- [ ] `ff_MIDAS.py --useTorchTransforms 1` runs end-to-end.
- [ ] `Pipeline.from_zarr(...).run()` smoke test runs cleanly.
- [ ] `apply_tilt_distortion` differentiability test produces non-zero grads.
- [ ] `midas-calibrate>=0.2.1` available on PyPI (pinned dep).
