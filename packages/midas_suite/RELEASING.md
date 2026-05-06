# Releasing midas-suite

This document describes how to cut a new release of the
`midas-suite` meta-package to PyPI and GitHub.

`midas-suite` is a thin **meta-package**: its only job is to declare
the MIDAS sub-packages as dependencies. There is no scientific code
to test — the smoke test only verifies the package imports and that
``midas_suite.installed()`` returns the expected sub-package list.

## When to bump

| Change | Bump |
|---|---|
| Tighten a dependency floor (no new sub-package) | patch (`0.1.0` → `0.1.1`) |
| Add a sub-package to `dependencies` (e.g. `midas-grain-odf` lands on PyPI) | minor (`0.1.0` → `0.2.0`) |
| Reorganise modality bundles (`[ff]`/`[nf]`/etc.) in a non-additive way | major (`0.x.y` → `1.0.0`) |

Only bump floors when a meta-package change actually requires it. If a
sub-package ships a patch release that doesn't break our usage, we do
**not** need to bump `midas-suite`.

## Quick reference

```bash
cd packages/midas_suite
./release.sh <new_version> [--publish | --dry-run]
```

## Three modes

| Command | What it does |
|---|---|
| `./release.sh 0.2.0` | **Prepare locally only** (default, safest). Version bump + smoke test + build + commit + tag. You push/publish manually. |
| `./release.sh 0.2.0 --publish` | **Fully automated**: prepare + push + GitHub release. The CI workflow (`python-packages.yml`) then runs smoke tests and auto-publishes to PyPI via trusted publishing (OIDC). |
| `./release.sh 0.2.0 --dry-run` | **Prepare but don't commit or tag**. For testing the build. Easy to undo with `git checkout -- pyproject.toml midas_suite/__init__.py`. |

## Step-by-step (prepare-only mode)

```bash
cd packages/midas_suite
./release.sh 0.2.0
```

The script will:
1. Verify you're on `master` with a clean working tree.
2. Verify the tag `midas-suite-v0.2.0` does not already exist.
3. Bump the version in `pyproject.toml` and `midas_suite/__init__.py`.
4. Run a **smoke test** that imports `midas_suite` and reports which
   sub-packages resolved (it does not fail if some sub-packages are
   missing locally — but it must import).
5. Clean `dist/`, `build/`, and `*.egg-info/`, then build the sdist
   and wheel via `python -m build`.
6. Commit the version bump.
7. Create an annotated git tag `midas-suite-v0.2.0`.

## Adding a new sub-package to the dep list

When a sub-package finally lands on PyPI (e.g. `midas-grain-odf`),
the meta-package needs three coordinated edits:

1. **`pyproject.toml`**: add the new dep to `dependencies` and any
   relevant `[project.optional-dependencies]` bundles, with a
   `>=<current-pypi-version>` floor.
2. **`midas_suite/__init__.py`**: append the new module name to
   `SUBPACKAGES` so `installed()` reports it.
3. **`README.md`**: add a row to the "What you get" table; remove
   the entry from the "Coming soon" list if applicable; refresh
   the sub-package count in the intro.

Then:

```bash
./release.sh 0.2.0 --publish    # minor bump for new sub-package
```

## Pre-release validation

Before tagging a release, sanity-check that every dependency floor in
`pyproject.toml` matches a real PyPI version:

```bash
cd packages/midas_suite
python -c "
import re, urllib.request, json
text = open('pyproject.toml').read()
deps = re.findall(r'\"(midas-[\w-]+)>=([\d.a-z]+)\"', text)
for name, floor in sorted(set(deps)):
    try:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/json', timeout=5) as r:
            latest = json.load(r)['info']['version']
        print(f'  {name:30s} floor={floor:10s} pypi-latest={latest}')
    except Exception as e:
        print(f'  {name:30s} floor={floor:10s} NOT ON PYPI ({e})')
"
```

Any "NOT ON PYPI" line is a release-blocker — pip will fail to resolve.

## Safety features

The script refuses to release in unsafe situations:

- **Not on master**: release must be cut from `master`.
- **Uncommitted changes** in `packages/midas_suite/`: must start clean.
- **Tag collision**: aborts if `midas-suite-v<X.Y.Z>` already exists
  locally (and on `origin` in `--publish` mode). PyPI rejects same-version
  re-uploads forever.
- **Smoke-test failure rollback**: if the import test fails, the version
  bump is automatically reverted.
- **Build failure rollback**: if `python -m build` fails, the version
  bump is automatically reverted.

## Prerequisites (one-time setup)

### For all modes
- Python environment with `pip` working; `build` and `twine` (auto-installed
  if missing).
- Working `git` config and push access to `origin`.

### For `--publish` mode additionally
- `gh` (GitHub CLI) installed and authenticated (`gh auth login`).
- GitHub Actions workflow `python-packages.yml` configured (already
  present in this repo).
- PyPI Trusted Publisher configured at
  https://pypi.org/manage/account/publishing/ with:
  - Owner: `marinerhemant`
  - Repository: `MIDAS`
  - Workflow: `python-packages.yml`
  - Environment: `pypi`
  - **Project name: `midas-suite`** (this is a separate PyPI project from
    the sub-packages — register the trusted publisher independently).

The CI workflow handles the PyPI upload automatically when a release
is created, using OIDC trusted publishing. No API token needed.

## Troubleshooting

### "tag already exists"
Either pick a higher version, or delete the local tag:
```bash
git tag -d midas-suite-v0.2.0
```

### "file already exists on PyPI"
PyPI rejects re-uploads of the same version, ever (even after deletion).
Bump to the next patch version and retry.

### "could not resolve midas-X" during install of midas-suite
A sub-package floor is set higher than what's on PyPI, or the sub-package
isn't published yet. Run the pre-release validation snippet above to
identify which dep is unresolvable.

### Manual PyPI upload (fallback)
If CI is broken and you need to ship urgently:
```bash
cd packages/midas_suite
twine upload dist/*
```
Requires a PyPI API token in `~/.pypirc`.
