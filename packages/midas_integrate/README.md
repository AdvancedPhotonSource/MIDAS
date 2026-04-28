# midas-integrate

Pure-Python radial integration for area X-ray detectors. A drop-in,
pip-installable replacement for the MIDAS C/CUDA radial integration
pipeline — no compilers, no native libraries, no CMake.

```bash
pip install midas-integrate
```

## What's in the box

| C/CUDA source                          | Python module                          |
| -------------------------------------- | -------------------------------------- |
| `MapperCore.c`, `DetectorGeometry.c`, `DetectorMapper.c` | `midas_integrate.detector_mapper`, `midas_integrate.geometry` |
| `IntegratorFitPeaksGPUStream.cu` (GPU streaming) | `midas_integrate.kernels`, `midas_integrate.server`, `midas_integrate.pipeline` |
| `IntegratorZarrOMP.c` (CPU OMP, bilinear) | `midas_integrate.kernels` (`mode='bilinear'`) |
| `PeakFit.c`                            | `midas_integrate.peakfit`              |
| `PeakFitIO.c`                          | `midas_integrate.peak_io`              |
| `Map.bin` / `nMap.bin`                 | `midas_integrate.bin_io`               |

## CPU/GPU selection

Everything that touches arrays accepts a `device` argument that is forwarded
to PyTorch. CPU and CUDA are first-class:

```python
from midas_integrate import build_csr, integrate, profile_1d, load_map

pixmap = load_map('Map.bin', 'nMap.bin')
geom_cpu  = build_csr(pixmap, n_r=990, n_eta=72, n_pixels_y=1475, n_pixels_z=1679, device='cpu')
geom_cuda = build_csr(pixmap, ..., device='cuda')

import torch
img = torch.from_numpy(image_2d)
profile = profile_1d(integrate(img, geom_cuda, mode='bilinear'), geom_cuda)
```

## Three integration modes (full parity with C codes)

| `mode`      | Equivalent C kernel                                   | Use when |
|-------------|--------------------------------------------------------|----------|
| `'floor'`   | `integrate_noMapMask` in `IntegratorFitPeaksGPUStream.cu` | streaming, max throughput |
| `'bilinear'`| pixel loop in `IntegratorZarrOMP.c` lines 1733–1744    | offline analysis, max accuracy |
| `'gradient'`| `GradientCorrection=1` branch in the GPU stream         | strong tilt + small R |

## CLI

Three entry points mirror the C binaries:

```bash
# 1. Build Map.bin / nMap.bin from a parameter file (one-shot, slow):
midas-detector-mapper params.txt -j 8

# 2. Integrate one frame (one-shot, fast):
midas-integrate params.txt --image frame.tif --device cuda

# 3. Streaming socket server (matches the C wire protocol on port 60439):
midas-integrate-server params.txt --device cuda --num-streams 4
```

## Numerical parity

- DetectorMapper output (`Map.bin`/`nMap.bin`): byte-equivalent to the C
  version (entry counts, sums of `frac` and `areaWeight` per bin agree to
  ULP; entry order within a bin may differ).
- Per-frame integration: float32 ULP-level (median 1.7e-8 relative error
  vs the C/CUDA `IntegratorFitPeaksGPUStream` binary on PILATUS3 2M with
  CeO₂ data; max 2.1e-7 relative).
- Peak fitting: same model (pseudo-Voigt + global background, SNIP
  background subtraction), different optimizer (scipy LM vs NLopt
  Nelder-Mead). Fit parameters typically agree to ~1e-5 relative on
  noisy real data.

## Performance (PILATUS3 2M, 1475×1679, NVIDIA H100)

|                                              | Throughput       |
|----------------------------------------------|------------------|
| C MIDAS GPU stream (per the paper)           | ~1,600 fps       |
| **midas-integrate (PyTorch CSR, FP32, CUDA)**| **~3,250 fps**   |
| midas-integrate (PyTorch CSR, FP64, CPU)     | ~675 fps         |
| C MIDAS CPU (per the paper)                  | ~262 fps         |
| pyFAI CSR-cython (per the paper)             | ~7 fps           |

## Requirements

- Python ≥ 3.9
- numpy, scipy, torch, tifffile, h5py, joblib

CUDA support is automatic if your installed `torch` has CUDA. macOS Metal (MPS)
works for the integration kernel; sparse-CSR support on MPS is partial as of
torch 2.7.

## License

BSD-3-Clause. See LICENSE.
