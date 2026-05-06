"""Pipeline stages — thin Python wrappers around existing package APIs.

Each stage is a one-shot function: it parses the parameter file (or
takes the already-parsed dict), calls into the relevant package's
public Python API, and writes its outputs to ``params['resultFolder']``.

No subprocess shells, no C binaries — every step is pure Python.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .params import collect_multiline, parse_parameters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers shared by stages
# ---------------------------------------------------------------------------

def _resolve_lsd(p: Dict[str, Any], param_file: str | Path) -> float:
    """Read the LAST ``Lsd`` value from the param file.

    NF param files typically have one ``Lsd`` line per detector
    distance; the C parser ``cfg->Lsd = ...`` overwrites on each call,
    so the LAST one wins. We mirror that.
    """
    lsds = collect_multiline(param_file, "Lsd")
    if not lsds:
        raise ValueError(f"No Lsd line found in {param_file}")
    return float(lsds[-1].split()[0])


def _resolve_max_ring_rad(p: Dict[str, Any]) -> float:
    """Pick ``RhoD`` (preferred) or ``MaxRingRad`` (alias)."""
    if "RhoD" in p:
        return float(p["RhoD"])
    if "MaxRingRad" in p:
        return float(p["MaxRingRad"])
    raise ValueError("Param file missing RhoD/MaxRingRad")


# ---------------------------------------------------------------------------
#  Stage 0: denoise (opt-in, requires MIDAS-NF-preProc)
# ---------------------------------------------------------------------------

def run_denoise(p: Dict[str, Any], param_file: str | Path) -> None:
    """Optional step 0: denoise raw TIFFs and re-point ``DataDirectory``."""
    if int(p.get("Denoise", 0)) != 1:
        logger.info("Denoise disabled (Denoise=0); skipping.")
        return

    method = str(p.get("DenoiseMethod", "nlm")).lower()
    if method not in ("nlm", "n2v"):
        raise ValueError(f"DenoiseMethod must be 'nlm' or 'n2v', got {method!r}")
    if method == "n2v":
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("DenoiseMethod=n2v requires torch") from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DenoiseMethod=n2v requires a CUDA GPU; none detected. "
                "Use DenoiseMethod=nlm for CPU-only."
            )
    try:
        from MIDAS_NF_preProc import denoise_directory
    except ImportError as e:
        raise RuntimeError(
            "Denoise=1 requires MIDAS-NF-preProc; "
            "install with: pip install MIDAS-NF-preProc"
        ) from e

    raw_dir = p.get("DataDirectory")
    if not raw_dir:
        raise RuntimeError("Denoise=1 requires DataDirectory in param file")
    denoised_dir = p.get("DenoisedDirectory") or os.path.join(raw_dir, "denoised")
    os.makedirs(denoised_dir, exist_ok=True)
    work_dir = os.path.join(p["resultFolder"], "_denoise_work")
    mask_threshold = (float(p["DenoiseMaskThreshold"])
                      if "DenoiseMaskThreshold" in p else None)

    logger.info(f"Denoise (method={method}): {raw_dir} -> {denoised_dir}")
    denoise_directory(
        input_dir=raw_dir, output_dir=denoised_dir,
        method=method,
        config_path=p.get("DenoiseConfigFile") or None,
        checkpoint_path=p.get("DenoiseCheckpoint") or None,
        pattern=str(p.get("DenoisePattern", "*.tif")),
        work_dir=work_dir,
        train_jointly=bool(int(p.get("DenoiseTrainJointly", 0))),
        finetune=bool(int(p.get("DenoiseFinetune", 0))),
        mask_threshold=mask_threshold,
        median=(int(p.get("DenoiseNoMedian", 0)) == 0),
    )

    # Re-point DataDirectory in-memory + on-disk so downstream stages
    # (ProcessImagesCombined, etc.) read the denoised TIFFs.
    p["DataDirectory"] = denoised_dir
    from .params import append_param_line
    append_param_line(param_file, "DataDirectory", denoised_dir)
    logger.info(f"Denoise complete; DataDirectory → {denoised_dir}")


# ---------------------------------------------------------------------------
#  Stage 1: preprocessing — hkls + seeds + hex grid + tomo filter + diffr spots
# ---------------------------------------------------------------------------

def run_get_hkls(p: Dict[str, Any], param_file: str | Path) -> str:
    """Generate ``hkls.csv`` via :func:`midas_hkls.write_nf_hkls_csv`."""
    from midas_hkls import Lattice, SpaceGroup, write_nf_hkls_csv

    sg_nr = int(p.get("SpaceGroup", p.get("SGNr", 225)))
    lp = p["LatticeParameter"]
    sg = SpaceGroup.from_number(sg_nr)
    lat = Lattice(lp[0], lp[1], lp[2], lp[3], lp[4], lp[5])
    wl = float(p["Wavelength"])
    lsd = _resolve_lsd(p, param_file)
    max_r = _resolve_max_ring_rad(p)

    out = Path(p["resultFolder"]) / "hkls.csv"
    n = write_nf_hkls_csv(
        out, sg, lat,
        wavelength_A=wl, lsd_um=lsd, max_ring_rad_um=max_r,
    )
    logger.info(f"Generated hkls.csv with {n} reflections at {out}")
    return str(out)


def run_seed_orientations_from_ff(p: Dict[str, Any]) -> str:
    """Far-field → near-field seed conversion (port of
    GenSeedOrientationsFF2NFHEDM via midas_nf_preprocess.seed_orientations).
    """
    from midas_nf_preprocess.seed_orientations.from_grains import (
        read_grains_orientations,
    )
    from midas_nf_preprocess.seed_orientations.io import write_seeds_csv

    grains_path = p["GrainsFile"]
    out_path = p["SeedOrientations"]
    seeds = read_grains_orientations(grains_path)
    write_seeds_csv(out_path, seeds)
    logger.info(f"Wrote {len(seeds)} seed orientations: {grains_path} → {out_path}")
    return out_path


def run_seed_orientations_from_cache(p: Dict[str, Any], install_dir: Optional[str] = None) -> str:
    """Auto-extract seed orientations from the MIDAS lookup cache."""
    from midas_nf_preprocess.seed_orientations.from_cache import (
        load_seeds_for_space_group, DEFAULT_SEED_DIR,
    )
    from midas_nf_preprocess.seed_orientations.io import write_seeds_csv

    sg = int(p.get("SpaceGroup", p.get("SGNr", 225)))
    seed_dir = DEFAULT_SEED_DIR
    if install_dir:
        candidate = Path(install_dir) / "NF_HEDM" / "seedOrientations"
        if candidate.is_dir():
            seed_dir = candidate
    seeds = load_seeds_for_space_group(sg, seed_dir=seed_dir)
    out_path = p["SeedOrientations"]
    write_seeds_csv(out_path, seeds)
    logger.info(f"Auto-extracted {len(seeds)} seeds for SG {sg}: {out_path}")
    return out_path


def run_hex_grid(p: Dict[str, Any], param_file: str | Path) -> str:
    """Generate ``grid.txt`` via :class:`midas_nf_preprocess.hex_grid.HexGrid`."""
    from midas_nf_preprocess.hex_grid.params import HexGridParams
    from midas_nf_preprocess.hex_grid.grid import HexGrid

    grid_params = HexGridParams.from_paramfile(str(param_file))
    grid = HexGrid.from_params(grid_params)
    out_path = Path(p["resultFolder"]) / grid_params.grid_filename
    grid.write(str(out_path))
    logger.info(f"Generated hex grid with {grid.n_points} voxels at {out_path}")
    return str(out_path)


def run_tomo_filter(p: Dict[str, Any]) -> None:
    """Mask grid points by a tomography image (port of filterGridfromTomo)."""
    if not p.get("TomoImage") or len(str(p["TomoImage"])) < 1:
        return
    from midas_nf_preprocess.tomo_filter.filter import (
        filter_grid_by_tomo, load_square_tomo,
    )

    tomo_path = p["TomoImage"]
    tomo_pixel_size = float(p.get("TomoPixelSize", 1.0))
    tomo = load_square_tomo(tomo_path)
    grid_path = Path(p["resultFolder"]) / "grid.txt"
    new_grid_path = filter_grid_by_tomo(
        str(grid_path), tomo=tomo, tomo_pixel_size=tomo_pixel_size,
    )
    # Match the legacy script: rename original to grid_unfilt.txt, new → grid.txt.
    if str(new_grid_path) != str(grid_path):
        shutil.move(str(grid_path), str(grid_path.with_name("grid_unfilt.txt")))
        shutil.move(str(new_grid_path), str(grid_path))
    logger.info(f"Tomo-filtered grid → {grid_path}")


def run_grid_mask(p: Dict[str, Any]) -> None:
    """Apply a rectangular ``GridMask`` ``[xmin, xmax, ymin, ymax]`` filter."""
    if not p.get("GridMask") or not isinstance(p["GridMask"], list) or len(p["GridMask"]) != 4:
        return
    grid_path = Path(p["resultFolder"]) / "grid.txt"
    if not grid_path.exists():
        return
    pts = np.genfromtxt(str(grid_path), skip_header=1, delimiter=" ")
    m = p["GridMask"]
    keep = (pts[:, 2] >= m[0]) & (pts[:, 2] <= m[1]) & (pts[:, 3] >= m[2]) & (pts[:, 3] <= m[3])
    pts = pts[keep]
    shutil.move(str(grid_path), str(grid_path.with_name("grid_old.txt")))
    np.savetxt(
        str(grid_path), pts, fmt="%.6f", delimiter=" ",
        header=str(pts.shape[0]), comments="",
    )
    logger.info(f"GridMask kept {pts.shape[0]} grid points")


def run_diffr_spots(p: Dict[str, Any], param_file: str | Path) -> None:
    """Run :class:`midas_nf_preprocess.diffr_spots.DiffrSpotsPipeline`."""
    from midas_nf_preprocess.diffr_spots.cli import run as diffr_run
    args = Namespace(
        parameter_file=str(param_file),
        device=None, dtype=None, output_dir=None,
    )
    diffr_run(args)
    logger.info("Diffraction spots simulated")


def run_preprocessing(
    p: Dict[str, Any], param_file: str | Path, *,
    skip_hex_grid: bool = False,
    skip_diffr_spots: bool = False,
    ff_seed_orientations: bool = False,
    install_dir: Optional[str] = None,
) -> None:
    """Run the full preprocessing block (HKLs → seeds → grid → diffr spots)."""
    run_get_hkls(p, param_file)

    if ff_seed_orientations:
        run_seed_orientations_from_ff(p)
    elif not p.get("SeedOrientations") or not Path(p["SeedOrientations"]).exists():
        run_seed_orientations_from_cache(p, install_dir=install_dir)

    # Update param file with the actual orientation count (NF C codes need it).
    seed_path = p.get("SeedOrientations", "")
    if seed_path and Path(seed_path).exists():
        with open(seed_path) as f:
            n_orient = sum(1 for _ in f)
        from .params import update_param_file
        update_param_file(param_file, {"NrOrientations": str(n_orient)})

    if not skip_hex_grid:
        run_hex_grid(p, param_file)
        run_tomo_filter(p)
        run_grid_mask(p)

    if not skip_diffr_spots:
        run_diffr_spots(p, param_file)


# ---------------------------------------------------------------------------
#  Stage 2: image processing — ProcessImagesCombined per detector distance
# ---------------------------------------------------------------------------

def run_image_processing(p: Dict[str, Any], param_file: str | Path) -> None:
    """Loop over detector distances and run the combined median + peak
    pipeline (port of ``ProcessImagesCombined`` in
    :mod:`midas_nf_preprocess.process_images`).
    """
    from midas_nf_preprocess.process_images.cli import run as proc_run

    n_distances = int(p.get("nDistances", 1))
    for d in range(1, n_distances + 1):
        logger.info(f"ProcessImages: distance {d}/{n_distances}")
        args = Namespace(
            parameter_file=str(param_file),
            distance_nr=d,
            n_cpus=int(p.get("nCPUs", 1)),
            device=None, dtype=None,
        )
        proc_run(args)


# ---------------------------------------------------------------------------
#  Stage 3: orientation fitting (FitOrientationOMP equivalent)
# ---------------------------------------------------------------------------

def run_fitting(
    p: Dict[str, Any], param_file: str | Path, *,
    n_blocks: int = 1, block_nr: int = 0,
    n_cpus: int = 1, device: str = "auto",
    dtype: str = "float64",
    refine: str = "nm-batched",
) -> None:
    """Run :func:`midas_nf_fitorientation.fit_orientation_run`.

    ``dtype`` is the new (post-auto-detect) plumbing: when the pipeline
    CLI resolves ``--dtype auto`` it lands here as ``'float32'`` on
    cuda/mps and ``'float64'`` on cpu. ``refine`` selects the phase-2
    strategy — ``nm-batched`` is the safe default; ``nm-triton`` is
    auto-selected by midas-nf-fitorientation when device=cuda + Triton
    is available, and is the production-fastest path.
    """
    import torch
    from midas_nf_fitorientation import fit_orientation_run

    # Delete stale MicFileBinary so fitting starts from a clean slate.
    mic_bin = p.get("MicFileBinary")
    if mic_bin:
        bin_path = Path(p["resultFolder"]) / mic_bin
        if bin_path.exists():
            bin_path.unlink()
            logger.info(f"Removed stale MicFileBinary: {bin_path}")

    # Map "float32" / "float64" / "fp32" / "fp64" → torch.dtype.
    dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float64": torch.float64, "fp64": torch.float64,
    }
    torch_dtype = dtype_map.get(str(dtype), torch.float64)

    fit_orientation_run(
        str(param_file), block_nr=block_nr, n_blocks=n_blocks,
        n_cpus=n_cpus, device=device, dtype=torch_dtype,
        refine=refine, verbose=False,
    )


# ---------------------------------------------------------------------------
#  Stage 4: post-process (ParseMic) + consolidation
# ---------------------------------------------------------------------------

def run_parse_mic(p: Dict[str, Any]) -> dict:
    """Run :func:`midas_nf_pipeline.parse_mic.parse_mic`."""
    from .parse_mic import ParseMicParams, parse_mic

    rf = p["resultFolder"]
    mic_bin = p.get("MicFileBinary", "")
    mic_text = p.get("MicFileText", "")
    if not mic_bin or not mic_text:
        raise ValueError("MicFileBinary and MicFileText required for parse_mic")

    bin_path = Path(rf) / mic_bin
    out_path = Path(rf) / (mic_text + ".mic")

    out = parse_mic(ParseMicParams(
        PhaseNr=int(p.get("PhaseNr", 1)),
        NumPhases=int(p.get("NumPhases", 1)),
        GlobalPosition=float(p.get("GlobalPosition", 0.0)),
        inputfile=str(bin_path),
        outputfile=str(out_path),
        nSaves=int(p.get("SaveNSolutions", 1)),
        SGNr=int(p.get("SpaceGroup", p.get("SGNr", 225))),
        GBAngle=float(p.get("GBAngle", 5.0)),
    ))
    logger.info(f"ParseMic produced {len(out)} output files")
    return out


def run_consolidate(p: Dict[str, Any], param_text: str, args_namespace=None,
                    output_path: Optional[str] = None,
                    resolution_label: Optional[str] = None) -> str:
    """Bundle outputs into a consolidated H5."""
    from .consolidate import generate_consolidated_hdf5

    rf = p["resultFolder"]
    mic_text_name = p.get("MicFileText", "")
    if not mic_text_name:
        raise ValueError("MicFileText not set")
    mic_text_path = Path(rf) / (mic_text_name + ".mic")
    return generate_consolidated_hdf5(
        str(mic_text_path),
        param_text=param_text,
        args_namespace=args_namespace,
        output_path=output_path,
        resolution_label=resolution_label,
    )


# ---------------------------------------------------------------------------
#  Multi-resolution helpers
# ---------------------------------------------------------------------------

def run_mic_to_grains(
    param_file: str | Path, mic_file: str | Path, out_file: str | Path,
    *, do_neighbor_search: int = 0, n_cpus: int = 1,
    min_conf_override: Optional[float] = None,
) -> int:
    """Run :func:`midas_nf_pipeline.mic2grains.mic_to_grains`."""
    from .mic2grains import Mic2GrainsParams, mic_to_grains
    return mic_to_grains(Mic2GrainsParams(
        param_file=param_file,
        mic_file=mic_file,
        out_file=out_file,
        do_neighbor_search=int(do_neighbor_search),
        n_cpus=int(n_cpus),
        min_conf_override=min_conf_override,
    ))
