# midas-pipeline

End-to-end MIDAS HEDM orchestrator. **FF is the single-scan degeneracy of PF.** One package, one CLI, two scan modes.

## Status

Alpha (P1 scaffold). Stages currently shell out to existing C binaries; in-place Python kernels arrive in P2–P8. Not for production use yet.

## Install

```bash
pip install -e packages/midas_pipeline
```

## CLI

```bash
midas-pipeline run --scan-mode {ff,pf} --params Parameters.txt --result rundir/
midas-pipeline status rundir/
midas-pipeline resume rundir/ --from <stage>
midas-pipeline reprocess rundir/
midas-pipeline inspect rundir/LayerNr_1/
midas-pipeline simulate --out simdir/ --params Parameters.txt
midas-pipeline seed --params ... --output UniqueOrientations.csv
```

When `--scan-mode` is omitted, the CLI sniffs the parameter file: `nScans > 1` or presence of `BeamSize` / scanning keys → `pf`, otherwise `ff`.

## Back-compat

The `midas-ff-pipeline` console-script remains available; it is a thin shim that injects `--scan-mode ff` and delegates here.

## Architecture

See [`../../.claude/plans/for-pf-we-don-t-lovely-locket.md`](../../.claude/plans/for-pf-we-don-t-lovely-locket.md) for the long-form plan. Quick summary:

- **One orchestrator** with a mode-dependent `STAGE_ORDER`.
- **Shared kernel packages** (`midas-index`, `midas-fit-grain`, `midas-transforms`, etc.) extended in place; FF behavior preserved by parity gates.
- **PF-only modules** live inside `midas_pipeline` (`find_grains/`, `sinogen`, `recon/`, `fuse`, `potts`, `em_refine`, `seeding/`).
- **Differentiability + multi-device** mandatory on every new compute path (CPU / CUDA / MPS via torch).

## Constraints

- No CUDA C; GPU support is torch-only.
- No deletions of legacy code in this effort.
- `midas-process-grains` is FF-only; PF consolidation is fresh pure-Python.
- `utils/calcMiso.py` is not imported by this package; all orientation math comes from `midas-stress`.
- `TOMO/midas_tomo_python.py` is imported in place, not relocated.
