# midas-integrate benchmarks

End-user performance benchmarks. These reproduce the same three timing
stages reported in the paper — detector-mapper build, sparse-CSR build,
and per-frame `integrate + 1D profile` — on **your** hardware.

## Quick start

```bash
# Default: PILATUS3 2M, CPU, float32, mode=floor (~30 seconds)
python tests/benchmarks/run_benchmarks.py

# Full sweep: all 8 paper detectors on CPU
python tests/benchmarks/run_benchmarks.py --all

# GPU sweep (requires CUDA-capable PyTorch)
python tests/benchmarks/run_benchmarks.py --all --device cuda

# Side-by-side comparison vs pyFAI on the same machine
pip install pyFAI
python tests/benchmarks/run_benchmarks.py --all --pyfai
```

## What it measures

For each `(detector, device, dtype, mode)` combination:

| Stage | What | Reported |
|---|---|---|
| `build_map` | One-shot detector-pixel → (R, η)-bin map (numba kernel) | wall ms |
| `build_csr` | Pack the map into a `torch.sparse_csr_tensor` for the chosen device | wall ms |
| `integrate + profile_1d` | Per-frame: `torch.matmul(sparse, image)` + 1D azimuthal mean | median, p95, min, max ms over `--n-iter` iterations after `--n-warmup` warmup |

`fps` in the summary is `1000 / median_ms` for the per-frame stage. The
one-shot stages (`build_map`, `build_csr`) are paid once per detector
geometry, not per frame.

## Detector configurations

The same eight detectors used in the paper, embedded in
[`run_benchmarks.py`](run_benchmarks.py#L30):

| Name | Mpx | Pixel size (µm) |
|---|---|---|
| `eiger2_500k` | 0.53 | 75 |
| `pilatus3_1m` | 1.02 | 172 |
| `pilatus3_2m` | 2.48 | 172 |
| `eiger2_4m` | 4.47 | 75 |
| `pilatus3_6m` | 6.22 | 172 |
| `varex_2923` | 8.29 | 150 |
| `eiger2_9m` | 10.17 | 75 |
| `eiger2_16m` | 18.09 | 75 |

All use `Lsd = 500 mm`, `RBinSize = 1 px`, `EtaBinSize = 5°`,
`Eta ∈ [-180°, 180°]`. Beam centre at the geometric centre, no tilt,
no distortion — the same configuration the paper benchmarks were run
under.

## Output

A JSON file is written to the current directory (or `--output PATH`).
The JSON contains:

- `host_info` — hostname, platform, Python/PyTorch versions, CPU count, CUDA devices
- `config` — the CLI options used
- `midas_integrate` — list of one result dict per `(detector, device, dtype, mode)`
- `pyfai` — same shape, only present when `--pyfai` was passed and pyFAI is installed

Result dict fields: `detector`, `device`, `dtype`, `mode`, `n_pixels_y`, `n_pixels_z`,
`mpx`, `n_r`, `n_eta`, `n_entries`, `build_map_ms`, `build_csr_ms`,
`median_ms`, `p95_ms`, `min_ms`, `max_ms`, `fps`, `n_warmup`, `n_iter`.

A side-by-side comparison printout is written to stdout at the end.

## CLI reference

```
--detectors NAME [NAME ...]   one or more detector names (default: pilatus3_2m)
--all                         all 8 detectors (overrides --detectors)
--device DEVICE               cpu | cuda | cuda:0 | mps      (default: cpu)
--dtype DTYPE                 float32 | float64               (default: float32)
--mode MODE                   floor | bilinear | gradient     (default: floor)
--image-dtype DTYPE           uint16 | uint32 | float32       (default: uint16)
--rbin FLOAT                  radial bin size, pixels         (default: 1.0)
--eta-bin FLOAT               azimuthal bin size, degrees     (default: 5.0)
--n-warmup N                  warmup iterations               (default: 10)
--n-iter N                    timed iterations                (default: 100)
--pyfai                       also benchmark pyFAI            (silent skip if not installed)
--output PATH                 JSON output path                (default: ./benchmark_results_<host>.json)
-q, --quiet                   suppress progress; print only the final summary
```

## Why the data is synthetic

Per-frame compute time is determined entirely by the precomputed sparse
matrix (its non-zero count and structure), not by pixel intensity values.
A uniform-noise image exercises the same memory and compute paths as a
real diffraction frame, so the reported `fps` is faithful to production
behaviour. The synthetic data also keeps the script self-contained — no
real CeO₂ frames or detector calibrations are needed.

## Comparing to the paper

The paper reports numbers from one specific machine
(*alleppey*: AMD EPYC 9474F, 16 threads, NVIDIA H100 PCIe). To compare
your numbers against the paper directly, run:

```bash
python tests/benchmarks/run_benchmarks.py --all --pyfai \
    --output bench_$(hostname).json
```

Then look at the `fps` values in the printed summary. CPU throughput
ratios (versus the paper's H100) will reflect your CPU's memory
bandwidth + thread count vs. the EPYC 9474F. GPU throughput (with
`--device cuda`) on a comparable NVIDIA card should land within a
factor of ~2 of the paper's H100 numbers.

## What's not benchmarked here

- Streaming server end-to-end throughput (TCP wire format + integration).
  See [`midas_integrate.server`](../../midas_integrate/server.py) for
  the production code; the per-frame number reported by this benchmark
  is the steady-state integration cost it bottlenecks on.
- azint / MatFRAIA. azint requires the maxiv conda channel, not pip,
  and has no GPU backend, so a meaningful comparison requires a more
  involved setup. The paper's azint numbers come from a separate run
  on alleppey under conda.
- Map-bin parity vs. the legacy C/OpenMP implementation. That's a
  correctness check, run separately as
  `tests/test_distortion_parity.py` etc.
