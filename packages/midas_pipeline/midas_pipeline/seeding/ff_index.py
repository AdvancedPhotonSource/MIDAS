"""Stage C of merged-FF seeding: run FF ``midas-index`` on the merged spot file.

Drives ``midas-index`` (in FF mode, no scan-position filter) against
the single merged ``InputAllExtraInfoFittingAll.csv`` produced by
:func:`midas_pipeline.seeding.merge_all_scans`. The output is
``IndexBest.bin`` / ``IndexBestFull.bin`` per the FF indexer's
contract — which downstream ``ProcessGrains`` (FF) consumes to emit
``Grains.csv``.

The merged-FF plan §6 also calls out a couple of indexer knobs that
should differ from the standard FF run:

- ``MinNHKLs`` is halved (small grains that are under-illuminated in
  some scans have effective merged-completeness ≈ K · matched / total).
  We surface this as a ``min_n_hkls`` argument; -1 means "use
  ``cfg_min_n_hkls // 2`` from the input paramstest".

- ``MarginEta`` / ``MarginOme`` may need slight widening to absorb
  the post-alignment residuals. v1 leaves these at their input values
  (the synthetic fixture has no drift to compensate); future revisions
  can multiply by a configurable factor.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class FfIndexResult:
    """Summary of the FF index invocation on a merged spot file."""

    paramstest_path: Path
    index_best_bin: Path
    index_best_full_bin: Optional[Path] = None
    n_grains_indexed: int = 0
    extra: Dict = field(default_factory=dict)


def _rewrite_paramstest(
    paramstest_in: Path, paramstest_out: Path, *,
    min_n_hkls_override: int | None,
    output_folder: Path,
) -> int:
    """Copy paramstest with scan-mode fields stripped and MinNHKLs adjusted.

    Returns the resolved MinNHKLs that landed in the derived file.
    """
    scan_keys = {"nScans", "BeamSize", "ScanPosTol", "nStepsToMerge",
                 "OutputFolder", "MinNHKLs"}
    cfg_min_n_hkls = -1
    kept_lines: list[str] = []
    for raw in paramstest_in.read_text().splitlines():
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            kept_lines.append(raw)
            continue
        key = stripped.split()[0]
        if key == "MinNHKLs":
            try:
                cfg_min_n_hkls = int(stripped.split()[1])
            except (IndexError, ValueError):
                pass
            continue
        if key in scan_keys:
            continue
        kept_lines.append(raw)

    if min_n_hkls_override is not None and min_n_hkls_override > 0:
        resolved = int(min_n_hkls_override)
    elif cfg_min_n_hkls > 0:
        resolved = cfg_min_n_hkls // 2
    else:
        resolved = -1

    if resolved > 0:
        kept_lines.append(f"MinNHKLs {resolved}")
    kept_lines.append(f"OutputFolder {output_folder}")
    paramstest_out.write_text("\n".join(kept_lines) + "\n")
    return resolved


def run_ff_indexer_on_merged(
    *,
    layer_dir: str | Path,
    merged_csv: str | Path,
    paramstest_in: str | Path,
    min_n_hkls: int = -1,                # -1 ⇒ cfg_min_n_hkls // 2
    output_folder: Optional[str | Path] = None,
    n_cpus: int = 1,
    device: str = "cpu",
    dtype: str = "float64",
    indexer_group_size: int = 4,
) -> FfIndexResult:
    """Run the FF indexer on the merged spot file.

    Builds a derived ``paramstest_merged.txt`` (scan-aware keys stripped,
    ``OutputFolder`` redirected, ``MinNHKLs`` halved or overridden), then
    shells out to ``python -m midas_index`` — same kernel + same CLI as
    the production FF path in :mod:`midas_pipeline.stages.indexing`.

    Parameters
    ----------
    layer_dir : path
        Working directory for the seeding run.
    merged_csv : path
        Output of :func:`merge_all_scans` (single
        ``InputAllExtraInfoFittingAll.csv``).
    paramstest_in : path
        Source ``paramstest.txt`` for the regular PF run.
    min_n_hkls : int
        Override for ``MinNHKLs``. ``-1`` defers to
        ``cfg_min_n_hkls // 2`` per plan §6.
    output_folder : optional path
        Indexer output directory. Defaults to
        ``<layer_dir>/Output_MergedFFSeeding``.
    n_cpus, device, dtype, indexer_group_size :
        Standard midas-index invocation knobs.

    Returns
    -------
    :class:`FfIndexResult` — paths to ``paramstest_merged.txt``,
    ``IndexBest.bin``, and (when present) ``IndexBestFull.bin``.
    Grain count is left at 0; the caller fills it in after
    ``ProcessGrains`` runs.
    """
    layer_dir = Path(layer_dir)
    merged_csv = Path(merged_csv)
    paramstest_in = Path(paramstest_in)
    if not merged_csv.exists():
        raise FileNotFoundError(f"merged spot file missing: {merged_csv}")
    if not paramstest_in.exists():
        raise FileNotFoundError(f"paramstest_in missing: {paramstest_in}")

    if output_folder is None:
        output_folder = layer_dir / "Output_MergedFFSeeding"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    paramstest_out = layer_dir / "paramstest_merged.txt"
    _rewrite_paramstest(
        paramstest_in, paramstest_out,
        min_n_hkls_override=min_n_hkls,
        output_folder=output_folder,
    )

    # Seed count comes from the SpotsToIndex.csv produced upstream; if
    # none, derive from the merged CSV row count (one seed per row).
    spots_to_index = layer_dir / "SpotsToIndex.csv"
    if spots_to_index.exists():
        n_seeds = sum(1 for line in spots_to_index.open() if line.strip())
    else:
        n_seeds = max(1, sum(1 for line in merged_csv.open() if line.strip()) - 1)

    cmd = [
        sys.executable, "-m", "midas_index",
        str(paramstest_out),
        "0", "1",                              # block_nr, n_blocks
        str(n_seeds),
        str(n_cpus),
        "--device", device,
        "--dtype", dtype,
        "--group-size", str(indexer_group_size),
    ]
    subprocess.run(cmd, cwd=str(layer_dir), check=True)

    index_best = output_folder / "IndexBest.bin"
    if not index_best.exists():
        index_best = layer_dir / "IndexBest.bin"
    index_best_full = index_best.with_name("IndexBestFull.bin")
    return FfIndexResult(
        paramstest_path=paramstest_out,
        index_best_bin=index_best,
        index_best_full_bin=index_best_full if index_best_full.exists() else None,
        n_grains_indexed=0,
        extra={"n_seeds": n_seeds, "output_folder": str(output_folder)},
    )
