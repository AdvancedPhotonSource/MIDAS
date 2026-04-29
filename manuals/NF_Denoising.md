# NF-HEDM Step 0 — Image Denoising

This page documents the optional **step 0** of the NF-HEDM workflow: self-supervised denoising of raw TIFF stacks before peak extraction. The denoising itself lives in the standalone pip package [`MIDAS-NF-preProc`](https://github.com/d-beniwal/MIDAS_NF_preProc) (BSD-3-Clause, by Devendra Beniwal). This page covers how MIDAS plugs into it.

## When to use it

Use step 0 when raw NF frames have visible noise that is producing spurious peaks in `ProcessImagesCombined` — extra spots, unstable orientation fits, or low confidence values across many voxels. Skip it when peak extraction already looks clean. The default (`Denoise 0`) preserves the current behaviour exactly.

## Install

```bash
pip install MIDAS-NF-preProc
```

Requires Python ≥ 3.10. NLM has no extra requirements. N2V additionally needs PyTorch with CUDA — install via the standard PyTorch channel for your CUDA version, then `pip install MIDAS-NF-preProc`.

## Two methods

| Method | Cost                      | When to use |
|--------|---------------------------|-------------|
| `nlm`  | CPU, seconds per stack    | Quick first pass; no GPU available; modest noise. |
| `n2v`  | **GPU required**, minutes–hours per stack | High-noise data; willing to train a per-image (or joint) U-Net for best quality. |

When `DenoiseMethod n2v` is set, the workflow performs `torch.cuda.is_available()` at the start of step 0 and **aborts immediately** if no CUDA GPU is detected — N2V training on CPU is not viable.

## Enabling step 0

Add to your parameter file:

```text
Denoise 1
DenoiseMethod nlm        # or 'n2v' on a GPU host
```

That's it. The workflow will:
1. Read raw TIFFs from `DataDirectory`.
2. Write denoised TIFFs to `{DataDirectory}/denoised/` (override with `DenoisedDirectory <path>`).
3. Re-point `DataDirectory` (in-memory and by appending a new `DataDirectory` line to the parameter file) so every downstream stage — `ProcessImagesCombined`, `MMapImageInfo`, `FitOrientationOMP` — reads the denoised stack.
4. Continue with the existing pipeline unchanged.

Raw data is never modified.

## Tuning

For NLM, defaults usually work. For N2V, drop a YAML next to your parameter file and point `DenoiseConfigFile` at it. The full list of tunable hyperparameters (epochs, depth, learning rate, masking percentage, tile size, etc.) is in the `MIDAS-NF-preProc` README. To export a starting template:

```bash
MIDAS-NF-preProc show-config n2v --output n2v.yaml
```

then edit `n2v.yaml` and reference it via `DenoiseConfigFile n2v.yaml` in the NF parameter file.

### N2V training modes

| Mode                                    | Parameter-file flags                            |
|-----------------------------------------|-------------------------------------------------|
| Per-image training (default)            | `DenoiseMethod n2v`                             |
| Joint training (one model for the stack)| `DenoiseMethod n2v` + `DenoiseTrainJointly 1`   |
| Predict from saved checkpoint           | `DenoiseMethod n2v` + `DenoiseCheckpoint <ckpt>`|
| Fine-tune checkpoint per image          | `DenoiseMethod n2v` + `DenoiseCheckpoint <ckpt>` + `DenoiseFinetune 1` |

### Threshold-mask mode

Setting `DenoiseMaskThreshold <value>` switches preProc to mask mode: it uses the denoised image as a binary signal detector and applies that mask to the **original** intensities. Useful when you want noise removal without altering the dynamic range of detected spots.

## Disk and time overhead

- Disk: a denoised copy of every TIFF in `DataDirectory` (same dtype, same dimensions). Plan for 1× the raw size.
- Time: NLM is essentially instantaneous (seconds for a typical NF stack). N2V per-image training takes minutes per frame on a single GPU; joint training is a single training run plus fast prediction across the stack.
- N2V checkpoints land in `{resultFolder}/_denoise_work/` and can be reused on later runs via `DenoiseCheckpoint`.

## Resume and restart

The denoise step is wired into `nf_MIDAS.py`'s pipeline-state machinery:

- `-restartFrom denoise` re-runs step 0 from scratch.
- `-restartFrom preprocessing` (or any later stage) skips step 0 but still re-points `DataDirectory` to `{DataDirectory}/denoised/` if a prior denoised stack exists, so you don't have to manually edit the parameter file between runs.
- The multi-resolution driver (`nf_MIDAS_Multiple_Resolutions.py`) runs step 0 **once** at the very start. Re-running with the denoised directory already populated short-circuits to a re-point, so resolution iterations don't pay the denoise cost twice.

## Interaction with `ProcessImagesCombined`'s built-in filtering

`MIDAS-NF-preProc`'s default temporal **median subtraction** (per-pixel median across the stack — removes static background) is complementary to `ProcessImagesCombined`'s **spatial median filter** (`MedFiltRadius`, applied per frame). Both can run safely in series. Disable the package's median pass with `DenoiseNoMedian 1` if you have already pre-subtracted background separately.
