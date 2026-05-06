"""``midas-ff-pipeline reprocess`` (gap #6).

Reruns ``merge_overlaps`` and the ``consolidation`` stage on a layer
directory that has already been processed (i.e. has ``Grains.csv``,
``SpotMatrix.csv``, ``InputAllExtraInfoFittingAll.csv``, …, plus a
``*.MIDAS.zip``). Mirrors ``ff_MIDAS.py -reprocess 1``.

This is useful when you have older runs that pre-date ``MergeMap.csv``
and the consolidated grain↔peak HDF5, or when you want to refresh
either after fixing an upstream bug without re-running the whole
pipeline.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ._logging import LOG, configure_logging
from .config import LayerSelection, MachineConfig, PipelineConfig
from .detector import DetectorConfig
from .stages._base import StageContext
from .stages import consolidation as _consolidation


def _find_zarrs(layer_dir: Path) -> list[Path]:
    return sorted(layer_dir.glob("*.MIDAS.zip"))


def _layer_nr_from_dir(layer_dir: Path) -> int:
    name = layer_dir.name
    if name.startswith("LayerNr_"):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 1
    return 1


def reprocess_dir(layer_dir: Path, *,
                  n_cpus: int = 8,
                  device: str = "cuda",
                  dtype: str = "float64") -> None:
    """Re-run merge + consolidation in a single layer directory."""
    configure_logging()
    layer_dir = layer_dir.resolve()
    LOG.info("=== reprocess: %s ===", layer_dir)

    zips = _find_zarrs(layer_dir)
    if not zips:
        raise FileNotFoundError(
            f"No .MIDAS.zip in {layer_dir}; nothing to reprocess."
        )
    primary_zip = zips[0]

    # 1) Re-run merge_overlaps for every zarr in the layer dir.
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True, parents=True)
    for zp in zips:
        LOG.info("  merging peaks in %s", zp)
        cmd = [
            "midas-merge-peaks", str(zp),
            "--result-folder", str(layer_dir),
            "--device", device,
            "--dtype", dtype,
        ]
        with (log_dir / f"reprocess_merge_{zp.stem}_out.csv").open("w") as fout, \
             (log_dir / f"reprocess_merge_{zp.stem}_err.csv").open("w") as ferr:
            rc = subprocess.call(cmd, cwd=str(layer_dir),
                                 stdout=fout, stderr=ferr)
        if rc != 0:
            raise RuntimeError(
                f"midas-merge-peaks failed (rc={rc}) on {zp}; "
                f"see {log_dir}/reprocess_merge_{zp.stem}_err.csv"
            )

    merge_map = layer_dir / "MergeMap.csv"
    if merge_map.exists():
        LOG.info("  MergeMap.csv: %s", merge_map)
    else:
        LOG.warning("  MergeMap.csv missing — merger may not have written it.")

    # 2) Build a minimal StageContext + PipelineConfig and call consolidation.
    fake_params = layer_dir / "paramstest.txt"
    if not fake_params.exists():
        # consolidation only reads the paramstest if it exists; create an
        # empty one so the dataclass __post_init__ doesn't choke on a
        # missing path.
        fake_params.write_text("")
    cfg = PipelineConfig(
        result_dir=str(layer_dir.parent),
        params_file=str(fake_params),
        n_cpus=n_cpus,
        device=device,
        dtype=dtype,
        layer_selection=LayerSelection(start=_layer_nr_from_dir(layer_dir),
                                        end=_layer_nr_from_dir(layer_dir)),
        machine=MachineConfig(name="local"),
        generate_h5=True,
    )
    detectors = [
        DetectorConfig(det_id=i + 1, zarr_path=str(zp),
                        lsd=0.0, y_bc=0.0, z_bc=0.0)
        for i, zp in enumerate(zips)
    ]
    ctx = StageContext(
        config=cfg,
        detectors=detectors,
        layer_nr=_layer_nr_from_dir(layer_dir),
        layer_dir=layer_dir,
        log_dir=log_dir,
    )
    started = time.time()
    _consolidation.run(ctx)
    LOG.info("  reprocess done in %.1fs", time.time() - started)
