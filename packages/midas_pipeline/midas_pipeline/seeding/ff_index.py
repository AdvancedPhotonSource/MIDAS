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

**Status**: thin shell. The actual indexer invocation is a
subprocess call to ``midas-index`` (or a programmatic call to
``midas_index.Indexer.run``) — we orchestrate the paramstest rewrite
+ shell out, then collect the output paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class FfIndexResult:
    """Summary of the FF index invocation on a merged spot file."""

    paramstest_path: Path
    index_best_bin: Path
    n_grains_indexed: int = 0
    extra: Dict = field(default_factory=dict)


def run_ff_indexer_on_merged(
    *,
    layer_dir: str | Path,
    merged_csv: str | Path,
    paramstest_in: str | Path,
    min_n_hkls: int = -1,                # -1 ⇒ cfg_min_n_hkls // 2
    output_folder: Optional[str | Path] = None,
) -> FfIndexResult:
    """Run the FF indexer on the merged spot file.

    Parameters
    ----------
    layer_dir : path
        Working directory.
    merged_csv : path
        Output of :func:`merge_all_scans` (single
        ``InputAllExtraInfoFittingAll.csv``).
    paramstest_in : path
        Source ``paramstest.txt`` for the regular PF run; this
        function writes a derived ``paramstest_merged.txt`` with
        scan-aware fields stripped and ``MinNHKLs`` halved (or
        overridden via ``min_n_hkls > 0``).
    min_n_hkls : int
        Override for ``MinNHKLs`` in the derived paramstest. ``-1``
        defers to ``cfg_min_n_hkls // 2`` per plan §6.
    output_folder : optional path
        Indexer output directory. Defaults to
        ``<layer_dir>/Output_MergedFFSeeding``.

    Returns
    -------
    :class:`FfIndexResult` — paths to the FF indexer's output, plus
    a placeholder grain count (filled in by the caller after
    ``ProcessGrains`` runs).

    Status
    ------
    Scaffold. Wiring the actual indexer invocation requires the
    transformed-paramstest writer + a ``midas-index`` subprocess call
    (or ``Indexer.from_param_file(...).run(block_nr=0, n_blocks=1)``).
    Raises ``NotImplementedError`` until that wiring lands.
    """
    raise NotImplementedError(
        "run_ff_indexer_on_merged: paramstest rewrite + midas-index "
        "invocation not yet wired up. See plan §6 + §16 Q1 (Option B "
        "orchestrator) for the integration plan."
    )
