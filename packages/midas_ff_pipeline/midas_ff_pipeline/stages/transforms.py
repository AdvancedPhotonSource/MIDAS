"""Detector transforms via ``midas-fit-setup``.

This stage produces the per-detector (or single) ``Spots.bin``,
``ExtraInfo.bin``, ``IDRings.csv``, and ``paramstest.txt``.

For multi-detector runs each detector goes into its own subdir;
``cross_det_merge`` then concatenates the per-detector spots.
"""
from __future__ import annotations

import time
from pathlib import Path

from ._base import StageContext, run_subprocess
from .._logging import LOG, stage_timer
from ..results import TransformsResult
from ..eta_coverage import (
    compute_panel_eta_coverage,
    total_coverage_per_ring,
    write_coverage_block,
)


def run(ctx: StageContext) -> TransformsResult:
    started = time.time()
    outputs: dict[str, str] = {}

    with stage_timer("transforms"):
        for det in ctx.detectors:
            zip_path = Path(det.zarr_path)
            stage_dir = ctx.stage_dir(det)
            cmd = [
                "midas-fit-setup", str(zip_path),
                "--result-folder", str(stage_dir),
                "--device", ctx.config.device,
                "--dtype", ctx.config.dtype,
            ]
            run_subprocess(
                cmd,
                cwd=stage_dir,
                stdout_path=ctx.log_dir / f"transforms_det{det.det_id}_out.csv",
                stderr_path=ctx.log_dir / f"transforms_det{det.det_id}_err.csv",
            )
            paramstest = ctx.stage_dir(det) / "paramstest.txt"
            if not paramstest.exists():
                raise FileNotFoundError(
                    f"transforms did not produce {paramstest} for det {det.det_id}"
                )
            outputs[str(paramstest)] = ""

    finished = time.time()
    paramstest_canonical: Path
    if ctx.is_multi_detector:
        # cross_det_merge will rewrite the canonical paramstest later
        paramstest_canonical = ctx.detector_dir(ctx.detectors[0]) / "paramstest.txt"
    else:
        paramstest_canonical = ctx.layer_dir / "paramstest.txt"

    # Patch each per-detector paramstest with OutputFolder + ResultFolder
    # so midas-index / midas-fit-grain / midas_process_grains find each
    # other's outputs in the conventional Output/ + Results/ layout.
    #
    # midas-index derives ``cwd = dirname(OutputFolder)`` to find
    # ``Spots.bin``. We *must* therefore set OutputFolder to
    # ``<stage_dir>/Output`` so dirname == ``<stage_dir>`` (where Spots.bin
    # actually lives). midas-fit-setup occasionally writes ``OutputFolder
    # <stage_dir>`` (no ``/Output`` suffix); rewrite that form here.
    #
    # Also compute and inject EtaCoverage_DetN rows for downstream
    # calc-radius / index / fit-grain consumers.
    for det in ctx.detectors:
        pt = ctx.stage_dir(det) / "paramstest.txt"
        if pt.exists():
            sd = ctx.stage_dir(det)
            target_out = str((sd / "Output").resolve())
            target_res = str((sd / "Results").resolve())
            new_lines: list[str] = []
            seen_out = False
            seen_res = False
            for raw in pt.read_text().splitlines():
                stripped = raw.strip()
                if stripped.startswith("OutputFolder"):
                    new_lines.append(f"OutputFolder {target_out}")
                    seen_out = True
                elif stripped.startswith("ResultFolder"):
                    new_lines.append(f"ResultFolder {target_res}")
                    seen_res = True
                else:
                    new_lines.append(raw)
            if not seen_out:
                new_lines.append(f"OutputFolder {target_out}")
            if not seen_res:
                new_lines.append(f"ResultFolder {target_res}")
            pt.write_text("\n".join(new_lines).rstrip() + "\n")
            (sd / "Output").mkdir(parents=True, exist_ok=True)
            (sd / "Results").mkdir(parents=True, exist_ok=True)
        _emit_eta_coverage(ctx, det)

    return TransformsResult(
        stage_name="transforms",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs=outputs,
        paramstest_path=str(paramstest_canonical),
        metrics={"n_detectors": len(ctx.detectors)},
    )


def _read_paramstest_kv(pt: Path) -> dict[str, list[str]]:
    """Loose paramstest reader returning {key: [tokens-after-key, ...]}."""
    out: dict[str, list[str]] = {}
    if not pt.exists():
        return out
    for raw in pt.read_text().splitlines():
        line = raw.split("#", 1)[0].strip().rstrip(";").rstrip()
        if not line:
            continue
        toks = [t.rstrip(";") for t in line.split()]
        out.setdefault(toks[0], []).append(" ".join(toks[1:]))
    return out


def _emit_eta_coverage(ctx: StageContext, det) -> None:
    """Compute the per-(det, ring) η coverage from the panel's geometry +
    hkls.csv ring radii and append ``EtaCoverage_DetN`` rows to the
    detector's paramstest.

    Pixel-enumeration over the nominal (distortion-free) tilt-rotated
    panel — see ``midas_ff_pipeline.eta_coverage``.
    """
    pt = ctx.stage_dir(det) / "paramstest.txt"
    if not pt.exists():
        return
    kv = _read_paramstest_kv(pt)
    try:
        n_pixels = int(float(kv.get("NrPixels", ["2048"])[0].split()[0]))
    except (ValueError, IndexError):
        n_pixels = 2048
    try:
        px_um = float(kv.get("px", ["200"])[0].split()[0])
    except (ValueError, IndexError):
        px_um = 200.0
    try:
        width_um = float(kv.get("Width", ["1500"])[0].split()[0])
    except (ValueError, IndexError):
        width_um = 1500.0

    # Read ring radii from hkls.csv mirrored into the per-det dir.
    hkls_csv = ctx.stage_dir(det) / "hkls.csv"
    if not hkls_csv.exists():
        # Fallback to layer_dir/hkls.csv (single-detector case).
        hkls_csv = ctx.layer_dir / "hkls.csv"
    if not hkls_csv.exists():
        LOG.warning(
            "no hkls.csv for det %d — skipping EtaCoverage emission",
            det.det_id,
        )
        return
    radii_by_ring: dict[int, float] = {}
    with hkls_csv.open() as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.replace(",", " ").split()
            try:
                rn = int(float(toks[4]))
                rad = float(toks[10])
            except (IndexError, ValueError):
                continue
            radii_by_ring.setdefault(rn, rad)
    if not radii_by_ring:
        LOG.warning(
            "hkls.csv at %s has no parseable ring radii — skipping coverage",
            hkls_csv,
        )
        return

    arcs = compute_panel_eta_coverage(
        n_pixels=n_pixels,
        px_um=px_um,
        lsd_um=det.lsd,
        y_bc_px=det.y_bc,
        z_bc_px=det.z_bc,
        tx_deg=det.tx,
        ty_deg=det.ty,
        tz_deg=det.tz,
        ring_radii_um=sorted(radii_by_ring.items()),
        width_um=width_um,
    )
    write_coverage_block(pt, det.det_id, arcs)
    cov = total_coverage_per_ring(arcs)
    cov_str = " ".join(f"r{rn}={c:.1f}°" for rn, c in sorted(cov.items()))
    LOG.info("  η coverage det %d: %s", det.det_id, cov_str)


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.stage_dir(d) / "paramstest.txt" for d in ctx.detectors]
