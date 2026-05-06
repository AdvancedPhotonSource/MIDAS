# Running the FF-HEDM pipeline

This is the user-facing entry point. The CLI is `midas-ff-pipeline run`;
notebooks under `notebooks/` are the corresponding Python-API tutorials.

## TL;DR — the recommended invocation

On any single-detector dataset, on a CUDA box (1+ GPUs):

```bash
midas-ff-pipeline run \
    --params /path/to/Parameters.txt \
    --result /path/to/results/ \
    --layers 1-1 \
    --device cuda
```

That's it. Three arguments and `--device cuda` are enough — every other
performance knob auto-resolves at run time.

The pipeline logs the auto-picked values at startup, e.g.:

```
auto: dtype=float32        ← float32 on cuda/mps, float64 on cpu
auto: shard-gpus=0,1       ← all visible CUDA devices used for the indexer
auto: group-size=4         ← scaled to smallest GPU memory in the shard set
```

If you want to override anything, pass it explicitly — explicit values
always win over `auto`.

## Hardware-tuned auto-detect

The pipeline introspects your hardware and picks:

| knob | rule |
|---|---|
| `--dtype auto` | `float32` on `cuda`/`mps`, `float64` on `cpu` (matches MIDAS production defaults; `cpu` only path needs fp64 for parity tests) |
| `--shard-gpus auto` | uses every visible CUDA device to fan the indexer; single-GPU boxes get no sharding overhead |
| `--group-size auto` | sized to smallest visible-GPU memory across the shard set: ≥70 GB → 8, ≥32 GB → 4, ≥16 GB → 2, else 1 |

Set `CUDA_VISIBLE_DEVICES=0` (or similar) to hide GPUs from the pipeline
when you want to leave one for another job.

For real datasets with many seeds (≥5 k), also export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent
memory-fragmentation OOMs.

## What happens when the pipeline runs

13 stages execute in order, each with its own provenance entry in
`<result>/LayerNr_N/midas_state.h5`:

```
zip_convert  → hkl  → peakfit  → merge_overlaps  → calc_radius
             → transforms  → cross_det_merge  → global_powder
             → binning  → indexing  → refinement
             → process_grains  → consolidation
```

Stages that aren't applicable to your run are no-ops: e.g.
`zip_convert` skips when the zarr already exists, `cross_det_merge` and
`global_powder` are no-ops for single-detector runs, `consolidation`
is gated by `--generate-h5`.

## Common workflows

### 1. From a `.MIDAS.zip` you already produced

```bash
midas-ff-pipeline run \
    --params Parameters.txt --result results/ \
    --zarr /path/to/my_dataset.MIDAS.zip \
    --device cuda
```

### 2. Raw GE5/HDF5/TIFF ingestion (no zarr yet)

The same command — `zip_convert` runs first and produces the zarr from
the `RawFolder` field in `Parameters.txt`. Override the raw dir at the
CLI if needed:

```bash
midas-ff-pipeline run \
    --params Parameters.txt --result results/ \
    --raw-dir /scratch/this_run/raw/ \
    --device cuda
```

### 3. Multi-layer batch mode

```bash
midas-ff-pipeline run \
    --params Parameters.txt --result results/ \
    --layers 1-20 --batch \
    --device cuda
```

`--batch` auto-discovers files matching `{FileStem}_{NNNNNN}{Ext}` in
the raw folder, skips `dark_*`, and runs one layer per discovered file.

### 4. NF→FF seeded indexing

```bash
midas-ff-pipeline run \
    --params Parameters.txt --result results/ \
    --layers 1-5 \
    --nf-result-dir /path/to/nf_results/ \
    --device cuda
```

For each layer N, the pipeline picks `GrainsLayer{N}.csv` from
`--nf-result-dir` and uses it as a seed grain list. Equivalent
`--grains-file` for a single-grains-file-fits-all-layers workflow.

### 5. Multi-detector pinwheel

```bash
midas-ff-pipeline run \
    --params Parameters.txt --result results/ \
    --detectors detectors.json \
    --device cuda
```

`detectors.json` carries per-panel geometry (Lsd, BC, tilts, distortion,
zarr_path). Each panel runs through `peakfit / merge / radius /
transforms` independently, then `cross_det_merge` concatenates them
with a `Spots_det.bin` side-car.

### 6. Resume / reprocess

```bash
# Resume an interrupted run from where it stopped
midas-ff-pipeline run --params P.txt --result results/ --resume auto

# Re-run from a specific stage onward
midas-ff-pipeline run --params P.txt --result results/ \
    --resume from --from indexing

# Re-run merge + consolidated HDF5 on completed result dirs
midas-ff-pipeline reprocess /path/to/results/

# Inspect per-stage timings + grain count
midas-ff-pipeline inspect /path/to/results/LayerNr_1
```

### 7. Generate consolidated HDF5

```bash
midas-ff-pipeline run \
    --params P.txt --result results/ \
    --device cuda --generate-h5
```

Adds a `consolidation` stage at the end that emits
`{stem}_consolidated.h5` with full grain↔spot↔peak provenance — see
`stages/consolidation.py` for the schema.

### 8. Cluster dispatch (Parsl)

The default is `--machine local`. Other options point at
`midas-parsl-configs` entries:

```bash
midas-ff-pipeline run --params P.txt --result results/ \
    --machine umich --n-nodes 4
```

To bring up a new cluster the first time:

```bash
midas-parsl-configs generate /path/to/your_submit.sh --name mycluster
midas-ff-pipeline run --params P.txt --result results/ --machine mycluster
```

The generator parses `#SBATCH` / `#PBS` directives + body
`module load` / `conda activate` lines and writes a runnable
`~/.midas/parsl_configs/myclusterConfig.py`.

## Performance — what auto-detect buys you

Measured on park22 304L SS in-situ tensile data (160k spots,
8353 indexer seeds, 4397 valid → 884 grains) on a 2× A6000 (48 GB
each) box:

| invocation | total wall clock | speedup vs original |
|---|---|---|
| original `--device cuda` (fp64 + gs=1, single GPU) | 1546 s | 1.0× |
| just `--device cuda` (auto picks fp32 + gs=4 + shard 0,1) | ~300 s | **~5×** |

On a single-GPU 48 GB box, the auto config still wins ~3× over the
original baseline (gets fp32+gs=4 but no sharding).

## Common pitfalls

| symptom | fix |
|---|---|
| `OOM at compare_spots._compute_avg_ia` during indexing | dataset too dense for the auto-picked `group_size`; pass `--group-size 2` (or 1) explicitly |
| Indexer says `0/N seeds with non-zero data` | most likely an upstream paramstest issue (no spots survived binning) — check `merge_overlaps` and `calc_radius` log files |
| `KeyError: 'midas-index ... __version__'` after running | stale namespace-package shim from an older install. Run: `pip uninstall midas-index midas-fit-grain midas-calibrate; rm -rf $SITE_PACKAGES/midas_*; pip install -e ...` |
| Refinement crash with 331 GB Jacobian | `--solver lm --mode all_at_once` is broken at >100 grain scale; use the default `--solver lbfgs` |

## Inspecting a finished run

```bash
midas-ff-pipeline status /path/to/results/        # per-layer / per-stage status table
midas-ff-pipeline inspect /path/to/results/LayerNr_1
```

Per-layer outputs land in `LayerNr_<N>/`:

```
Output/
    IndexBest.bin         per-seed best orientation+position
    IndexBestFull.bin     full per-seed candidate matches
    FitBest.bin           per-grain refinement traces
Results/
    OrientPosFit.bin      refined orientations + positions
    Key.bin               grain ID lookup
    ProcessKey.bin        process_grains internal key
Grains.csv                FINAL grain list (consumed by ProcessGrains)
SpotMatrix.csv            per-grain spot assignments
midas_state.h5            provenance ledger (used by --resume)
```

## Running from Python (notebook style)

The `notebooks/` directory has six end-to-end demos covering the
above workflows. Pick the one that matches your use case:

| notebook | covers |
|---|---|
| `01_smoke_walkthrough.ipynb` | single-detector synthetic Au, full pipeline |
| `02_stage_diagnostics.ipynb` | per-stage drilldown + plotting |
| `03_multi_detector_demo.ipynb` | 4-panel hydra pinwheel |
| `04_batch_multilayer.ipynb` | `--batch` discovery + multi-layer loop |
| `05_nf_seeded.ipynb` | `--nf-result-dir` seeded refinement |
| `06_resume_and_reprocess.ipynb` | resume + reprocess subcommand |

The notebooks default to `device='cpu'` so they run on a laptop without
a GPU. On a CUDA box, change to `device='cuda'` and you get the same
auto-detect behaviour as the CLI (the resolvers live in `cli.py` —
the Python API takes concrete values, so for notebooks you pick
explicitly).

## Verifying your install

```bash
midas-ff-pipeline --version
python -m pytest packages/midas_ff_pipeline/tests   # 51 tests, ~2 s
midas-ff-pipeline run --help                          # see every flag
```
