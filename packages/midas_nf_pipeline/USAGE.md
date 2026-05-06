# Running the NF-HEDM pipeline

`midas-nf-pipeline` replaces the legacy `nf_MIDAS.py` and
`nf_MIDAS_Multiple_Resolutions.py` with a single Python-only entry
point. Notebooks under `notebooks/` are the corresponding Python-API
tutorials.

## TL;DR — the recommended invocation

Single-resolution, single-layer:

```bash
midas-nf-pipeline run /path/to/Parameters.txt --device cuda
```

Multi-resolution (driven by the `GridRefactor` key in the parameter file)
or multi-layer is just a flag away:

```bash
midas-nf-pipeline run /path/to/Parameters.txt \
    --device cuda \
    --start-layer 1 --end-layer 5 \
    --result-folder /scratch/out/
```

That's enough — every other performance knob has an `auto` default that
picks the right value at run time.

## Hardware-tuned auto-detect

| flag | rule |
|---|---|
| `--device auto` (default) | `cuda` if `torch.cuda.is_available()`, else `cpu` |
| `--dtype auto` (default) | `float32` on cuda/mps, `float64` on cpu — matches the `midas-nf-preprocess` default and the Triton-kernel internal precision |
| `--refine nm-batched` (default) | midas-nf-fitorientation auto-promotes to `nm-triton` (single fused CUDA kernel; ~3× over batched) when device=cuda **and** the Triton package is installed; falls back to `nm-batched` cleanly otherwise |

Each auto-resolution is logged at startup so you can see what got picked.

## Subcommands at a glance

```bash
midas-nf-pipeline run            <paramFN> [...]   # full pipeline (single + multi-res, single + multi-layer)
midas-nf-pipeline parse-mic      <paramFN>          # ParseMic byte-parity port
midas-nf-pipeline mic2grains     <paramFN> <mic> <out> [...]  # Mic2GrainsList port
midas-nf-pipeline consolidate    <mic> [--paramFN ...] [--output ...]  # rebuild consolidated HDF5
midas-nf-pipeline refine-params  <paramFN> [--multi-point] [--row-nr N] # parameter / calibration refinement
```

## What happens when `run` executes

For each layer in `[--start-layer, --end-layer]`:

```
denoise (optional)
  → loop 0 unseeded:
      preprocessing (HKL, seed orientations, hex grid, tomo filter, grid mask, diffr spots)
      → image processing (ProcessImagesCombined per detector distance)
      → fitting (FitOrientation — nm-triton on cuda, nm-batched otherwise)
      → parse-mic
      → consolidate (loop_0_unseeded resolution)
  → for each refinement loop k=1..NumLoops:
      seeded pass on refined grid
      → unseeded pass
      → merge seeded ∪ unseeded
      → consolidate (loop_k_seeded / unseeded resolutions)
  → mic2grains on the final mic
```

Single-resolution mode (`NumLoops=0`, no `GridRefactor` key) skips the
refinement loops; multi-resolution drives them off the `GridRefactor`
parameter triplet `(starting_grid, scaling_factor, num_loops)`.

Each stage's completion is recorded in
`<result>/<MicFileText>_pipeline.h5` so `--resume` and `--restart-from`
can pick up where you left off.

## Common workflows

### 1. Single-resolution, single-layer (legacy `nf_MIDAS.py` equivalent)

```bash
midas-nf-pipeline run Parameters.txt --device cuda
```

### 2. Multi-resolution, single-layer (legacy `nf_MIDAS_Multiple_Resolutions.py`)

Add `GridRefactor <starting_grid> <scaling> <num_loops>` to your param
file. The `run` subcommand auto-detects multi-res mode from that key.

```bash
midas-nf-pipeline run Parameters.txt --device cuda
```

### 3. Multi-layer batch

```bash
midas-nf-pipeline run Parameters.txt \
    --device cuda \
    --start-layer 1 --end-layer 10 \
    --result-folder /scratch/out/
```

Per-layer outputs go to `<result-folder>/LayerNr_<n>/`. The pipeline
auto-discovers per-layer subdirs and writes per-layer
`{MicFileText}_pipeline.h5` ledgers.

### 4. Seed from FF results

```bash
midas-nf-pipeline run Parameters.txt \
    --device cuda \
    --ff-seed-orientations
```

Loop 0 starts from the `GrainsFile` key in the parameter file (an FF
`Grains.csv`) instead of the default cache-based fundamental-zone
seeds.

### 5. Skip image processing (already produced `SpotsInfo.bin`)

```bash
midas-nf-pipeline run Parameters.txt \
    --device cuda \
    --no-image-processing
```

### 6. Parameter refinement (calibration)

```bash
# Single-point refinement at a specific row
midas-nf-pipeline refine-params Parameters.txt --row-nr 42 --device cuda

# Multi-point refinement
midas-nf-pipeline refine-params Parameters.txt --multi-point --device cuda
```

### 7. Resume / restart

```bash
# Resume from the last completed stage in the pipeline H5
midas-nf-pipeline run Parameters.txt --resume /path/to/run.MicFileText_pipeline.h5

# Restart from a named stage (e.g. after editing params for loop 1)
midas-nf-pipeline run Parameters.txt --restart-from loop_1_seeded
```

Stage labels are `loop_<k>_initial`, `loop_<k>_seeded`,
`loop_<k>_unseeded`, `loop_<k>_merge`. The label ordering is what
`get_completed_stages` walks to find the resume point.

### 8. Rebuild a consolidated HDF5

If you have a `.mic` file from an older run and want the consolidated
HDF5 with parameter provenance:

```bash
midas-nf-pipeline consolidate /path/to/result.mic \
    --paramFN /path/to/Parameters.txt \
    --output /path/to/result_consolidated.h5
```

### 9. Convert a `.mic` to grain list

```bash
midas-nf-pipeline mic2grains Parameters.txt result.mic grains.txt 1 8
```

(`do_neighbor_search=1`, `n_cpus=8`.)

## Phase-2 refine strategy

The `--refine` flag picks how the FitOrientation phase 2 (orientation
fit on the winner from phase 1) runs. Options:

| value | when to use |
|---|---|
| `nm-batched` | safe default. Vectorised PyTorch Nelder-Mead, batches every voxel in one forward call per NM iteration. Production-ready on CPU. |
| `nm-triton` | cuda only. Single fused Triton kernel — Bragg + projection + tilts + obs lookup in one launch. Auto-selected when device=cuda + Triton installed. ~3× faster than `nm-batched` on GPU. |
| `nm-serial` | per-voxel `scipy.optimize.minimize` on the hard objective. Used as a parity oracle for the batched path; way slower in production. |
| `lbfgs+nm` | L-BFGS warmup on the soft Gaussian-splat surrogate, then NM polish on the hard objective. Useful when starting orientations are far from the basin. |
| `lbfgs` | pure L-BFGS on the soft surrogate. Smoothest but loses the hard FracOverlap objective. |

For real production runs on CUDA: leave `--refine nm-batched` and the
auto path will pick `nm-triton` on entry.

## Inspecting a finished run

```bash
ls /path/to/results/LayerNr_1/
# → <stem>_pipeline.h5         # provenance ledger (used by --resume)
# → <stem>_consolidated.h5     # consolidated HDF5 (voxels + grains + maps)
# → <MicFileText>.<loop>.mic   # per-loop microstructure outputs
```

The consolidated HDF5 holds:

```
/parameters/                 # all paramfile keys + run args
/voxels/{position, euler, confidence, grain_id, ...}
/grains/{grain_id, orientation, position, ...}
/maps/{kam, grod, ...}        # populated when ParseMic produces these
/multi_resolution/loop_<k>_<pass>/   # per-resolution snapshots
/raw_data_ref/                # path to original DataDirectory
```

## Common pitfalls

| symptom | fix |
|---|---|
| `OOM during fitting` | drop precision: `--dtype fp64 → fp32`. The Triton kernels are fp32-only anyway, so this is a free switch on cuda. |
| `Triton kernel not found, falling back to nm-batched` | `pip install triton` (CUDA-only Linux build) — significantly speeds up phase 2 on cuda. |
| `MicFileBinary missing` warning | the pipeline deletes the stale binary before fitting; benign. |
| Stale namespace-package shim after a pip refresh | `pip uninstall midas-nf-pipeline midas-nf-fitorientation midas-nf-preprocess; rm -rf $SITE_PACKAGES/midas_nf_*; pip install -e ...` |
| `MicFileText` collisions across layers | the pipeline auto-suffixes `.0`, `.1`, ... per loop. If you see `nf_output.mic` clobbering, you're running with the legacy script — switch to `midas-nf-pipeline run`. |

## Running from Python (notebook style)

The `notebooks/` directory has six end-to-end demos:

| notebook | covers |
|---|---|
| `00_quickstart_au.ipynb` | bundled Au synthetic, single-resolution end-to-end |
| `01_single_resolution.ipynb` | single-resolution drilldown with paramfile inspection |
| `02_multi_resolution.ipynb` | multi-resolution loop walk (`GridRefactor`) |
| `03_refine_parameters.ipynb` | calibration refinement (single + multi-point) |
| `04_resume_restart.ipynb` | resume + restart-from semantics |
| `05_multi_layer_batch.ipynb` | multi-layer driver |

The notebooks default to `device='cpu' + dtype='fp64'` so they run on a
laptop without a GPU. On a CUDA box, change to `device='cuda'` and pass
`dtype='fp32'` (or just leave both at `'auto'` if invoking the CLI
flow) for the same auto-detect behaviour as the CLI.

## Verifying your install

```bash
midas-nf-pipeline --version
python -m pytest packages/midas_nf_pipeline/tests   # 31 tests, ~7 s
midas-nf-pipeline run --help                          # see every flag
midas-nf-pipeline consolidate \
    /path/to/some.mic --output /tmp/check.h5         # offline subcommand smoke test
```

## What's NOT yet wired (vs the FF pipeline)

The FF pipeline picked up multi-GPU sharding (`--shard-gpus 0,1`) and
cluster-dispatch (`--machine umich --n-nodes 4`) earlier in the
parity work. Both are deliberately deferred for NF:

- **Multi-GPU sharding for fitting** — `fit_orientation_run` already
  supports `block_nr / n_blocks`, but the `MicFileBinary` writes need a
  pwrite-safety audit before two processes can share the file (similar
  to the `IndexBest.bin` race we fixed in midas-index). When that
  audit lands, this becomes a one-flag addition.
- **Parsl cluster dispatch** — punted per the FF parity decision. NF
  workflows are typically single-node on the operator workstation.

`--skip-validation` / `--strict-validation` are wired and call into
`midas_params.hook.preflight_validate` when that package is installed
(soft dependency).

## Differences from `nf_MIDAS.py`

| legacy flag | new flag / status |
|---|---|
| `-paramFN` | positional `paramFN` |
| `-nCPUs` | `--n-cpus` |
| `-machineName` | not yet wired (deferred) |
| `-nNodes` | not yet wired (deferred) |
| `-ffSeedOrientations` | `--ff-seed-orientations` |
| `-doImageProcessing` | `--no-image-processing` (inverted-default) |
| `-refineParameters 1 -multiGridPoints` | `refine-params` subcommand + `--multi-point` |
| `-gpuFit` | superseded by `--device cuda` (auto by default) |
| `-resume` / `-restartFrom` | `--resume` / `--restart-from` |
| `-skipValidation` / `-strictValidation` | `--skip-validation` / `--strict-validation` |
| `-startLayerNr` / `-endLayerNr` | `--start-layer` / `--end-layer` |
| `-resultFolder` | `--result-folder` |
| `-minConfidence` | `--min-confidence` |
| (legacy multi-res) | `GridRefactor` key in paramfile auto-detected by `run` |
