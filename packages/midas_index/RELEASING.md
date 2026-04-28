# Releasing midas-index

This document describes how to cut a new release of the
`midas-index` package to PyPI and GitHub.

## Quick reference

```bash
cd packages/midas_index
./release.sh <new_version> [--publish | --dry-run]
```

## Three modes

| Command | What it does |
|---------|--------------|
| `./release.sh 0.1.0` | **Prepare locally only** (default). Version bump + tests + build + commit + tag. You push/publish manually. |
| `./release.sh 0.1.0 --publish` | **Fully automated**: prepare + push + GitHub release. CI auto-publishes to PyPI via trusted publishing (OIDC). |
| `./release.sh 0.1.0 --dry-run` | **Prepare but don't commit or tag**. Easy to undo with `git checkout -- pyproject.toml midas_index/__init__.py`. |

## Step-by-step (prepare-only mode)

```bash
cd packages/midas_index
./release.sh 0.1.0
```

The script will:
1. Verify you're on `master` with a clean working tree.
2. Verify the tag `midas-index-v0.1.0` does not already exist.
3. Bump the version in `pyproject.toml` and `midas_index/__init__.py`.
4. Run the test suite (aborts release on failure, rolls back version).
5. Clean `dist/`, `build/`, and `*.egg-info/`, then build sdist + wheel.
6. Commit the version bump.
7. Create an annotated git tag `midas-index-v0.1.0`.
8. Print the remaining manual commands to push + publish.

## Pre-release checklist

Before bumping to a non-dev version (e.g. `0.1.0` from `0.1.0.dev0`):

- [ ] Reference dataset committed under `tests/data/` (or Zenodo gate).
- [ ] Regression tests pass: CPU/float64 byte-identical to `IndexerOMP` golden.
- [ ] CUDA regression tests pass on a GPU box (not local CI).
- [ ] `ff_MIDAS.py` `--useTorchIndexer` flag landed and tested.
- [ ] `midas-diffract>=0.1.1` (with `forward_from_R`) on PyPI.
- [ ] `midas-stress>=0.5.0` (torch-native orientation) on PyPI.
- [ ] Quick-start in `README.md` runs cleanly on a fresh install.
