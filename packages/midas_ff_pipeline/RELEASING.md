# Releasing midas-ff-pipeline

This document describes how to cut a new release of the
`midas-ff-pipeline` package to PyPI and GitHub.

## Quick reference

```bash
cd packages/midas_ff_pipeline
./release.sh <new_version> [--publish | --dry-run]
```

## Three modes

| Command | What it does |
|---------|--------------|
| `./release.sh 0.1.1` | **Prepare locally only** (default, safest). Version bump + tests + build + commit + tag. You push/publish manually. |
| `./release.sh 0.1.1 --publish` | **Fully automated**: prepare + push + GitHub release. The CI workflow (`python-packages.yml`) then runs tests and auto-publishes to PyPI via trusted publishing (OIDC). |
| `./release.sh 0.1.1 --dry-run` | **Prepare but don't commit or tag**. For testing the build. Easy to undo with `git checkout -- pyproject.toml midas_ff_pipeline/__init__.py`. |

## Step-by-step (prepare-only mode)

This is the default. The script stops after building artifacts so you
can review before publishing.

```bash
cd packages/midas_ff_pipeline
./release.sh 0.1.1
```

The script will:
1. Verify you're on `master` with a clean working tree.
2. Verify the tag `midas-ff-pipeline-v0.1.1` does not already exist.
3. Bump the version in `pyproject.toml` and `midas_ff_pipeline/__init__.py`.
4. Run the test suite (aborts release on failure, rolls back version).
5. Clean `dist/`, `build/`, and `*.egg-info/`, then build the sdist
   and wheel via `python -m build`.
6. Commit the version bump.
7. Create an annotated git tag `midas-ff-pipeline-v0.1.1`.
8. Print the remaining manual commands to push + publish.

After the script completes, you run:

```bash
git push origin master --follow-tags

gh release create midas-ff-pipeline-v0.1.1 dist/* \
    --title "midas-ff-pipeline v0.1.1" \
    --generate-notes

twine upload dist/*
```

## One-shot mode (`--publish`)

For routine releases where you trust the script:

```bash
cd packages/midas_ff_pipeline
./release.sh 0.1.1 --publish
```

This does the local + GitHub part end-to-end:
prepare -> commit -> tag -> push -> GitHub release.

The GitHub Actions workflow (`.github/workflows/python-packages.yml`)
takes over from there:
1. Runs tests on Linux and macOS across Python 3.9/3.11/3.12.
2. Builds the sdist + wheel.
3. Uploads to PyPI via trusted publishing (OIDC) -- no API token
   needed.

## Dry-run mode (`--dry-run`)

For testing the build without modifying git history:

```bash
cd packages/midas_ff_pipeline
./release.sh 0.1.1 --dry-run
```

The script bumps version, runs tests, and builds, but does NOT commit
or tag. To undo the local version bump:

```bash
git checkout -- pyproject.toml midas_ff_pipeline/__init__.py
```

## Safety features

The script refuses to release in unsafe situations:

- **Not on master**: release must be cut from `master`.
- **Uncommitted changes**: release must start from a clean tree.
- **Tag collision**: aborts if the tag already exists locally
  (and in `--publish` mode, if it exists on `origin`).
- **Test failure rollback**: version bump auto-reverted on failure.
- **Build failure rollback**: version bump auto-reverted on failure.
- **Dependency auto-install**: `build` and `twine` installed if missing.
- **Prerequisite checks**: `--publish` mode verifies `gh` and `twine`.

## Test environment

The test suite has two tiers:

- **Pure-Python tests** (`test_forward.py`, `test_losses.py`,
  `test_strain_tensor.py`): run unconditionally; require only
  `torch` and `numpy`.
- **C cross-validation tests** (`test_c_comparison.py`,
  `test_tilts.py`): auto-skip when the MIDAS C executables
  (`ForwardSimulationCompressed`, `simulateNF`, `GetHKLList`) are not
  found. Set `MIDAS_HOME=/path/to/MIDAS` to point at a non-default
  install. CI runs without the C binaries; release-time validation
  on a developer machine should run with them present.

## Prerequisites (one-time setup)

### For all modes
- Python environment with `pip` working.
- Working `git` config and push access to `origin`.

### For `--publish` mode additionally
- `gh` (GitHub CLI) installed and authenticated.
- GitHub Actions workflow `python-packages.yml` configured.
- PyPI Trusted Publisher configured at
  https://pypi.org/manage/account/publishing/ with:
  - Owner: `marinerhemant`
  - Repository: `MIDAS`
  - Workflow: `python-packages.yml`
  - Environment: `pypi`
- GitHub environment named `pypi` exists in repo settings.

## Version numbering

Follow [Semantic Versioning](https://semver.org):

| Change | Bump |
|--------|------|
| Bug fix, doc tweak, no API change | `0.1.0` -> `0.1.1` (patch) |
| New feature, backwards compatible | `0.1.1` -> `0.2.0` (minor) |
| Breaking API change | `0.2.0` -> `1.0.0` (major) |

## Troubleshooting

### "tag already exists"
Either pick a higher version, or delete the local tag:
```bash
git tag -d midas-ff-pipeline-v0.1.1
```

### "file already exists on PyPI"
PyPI rejects re-uploads of the same version, ever. Bump to the next
patch version and retry.

### Tests failed, release aborted
The version bump was rolled back. Fix the tests, commit, and retry.

### Build failed, release aborted
The version bump was rolled back. Check `python -m build` output for
the error (usually a missing dependency in `pyproject.toml`).
