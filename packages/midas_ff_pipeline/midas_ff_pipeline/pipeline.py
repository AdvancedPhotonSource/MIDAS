"""Pipeline orchestrator — drives all stages with optional resume.

Multi-layer, multi-detector, with provenance. Stage execution order:

  hkl → peakfit → merge_overlaps → calc_radius → transforms
       → cross_det_merge → binning → indexing → refinement → process_grains

Each stage is exposed as ``Pipeline.run_<stage>()`` and the full run is
``Pipeline.run()``. Resume picks up from the last completed stage by
default, or from a named stage when ``resume_from_stage`` is set.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from ._logging import LOG, configure_logging
from .config import PipelineConfig
from .detector import DetectorConfig
from .provenance import ProvenanceStore
from .results import LayerResult
from .seeding import (
    apply_raw_dir_override,
    patch_params_with_grains,
    resolve_grains_file_for_layer,
)
from .stages import (
    StageContext,
    binning,
    calc_radius,
    consolidation,
    cross_det_merge,
    hkl,
    index_stage,
    merge_overlaps,
    peakfit,
    process_grains,
    refine,
    transforms,
    zip_convert,
)
from .stages import global_powder


# ---- canonical stage order ----
STAGE_ORDER: list[tuple[str, object]] = [
    # zip_convert is a no-op when every detector already has a valid zarr
    # (the common notebook + multi-det case). It only fires for raw → zarr
    # ingestion runs.
    ("zip_convert", zip_convert),
    ("hkl", hkl),
    ("peakfit", peakfit),
    ("merge_overlaps", merge_overlaps),
    ("calc_radius", calc_radius),
    ("transforms", transforms),
    # cross_det_merge concatenates per-detector InputAll.csv +
    # InputAllExtraInfoFittingAll.csv (with renumbered SpotID + det_id
    # side-car) into the layer_dir. Binning then runs once globally on the
    # merged input. For single-detector runs cross_det_merge is a thin
    # tag-only no-op.
    ("cross_det_merge", cross_det_merge),
    # global_powder rebuilds the per-(ring, η) intensity model from all
    # panels' obs spots and rewrites GrainRadius in the merged InputAll.csv.
    # No-op for single-detector. Multi-distance overlapping panels get
    # averaged; gaps in coverage drop out of the powder estimator.
    ("global_powder", global_powder),
    ("binning", binning),
    ("indexing", index_stage),
    ("refinement", refine),
    ("process_grains", process_grains),
    # consolidation is a no-op unless ``config.generate_h5`` is set, but
    # we keep it in the canonical stage order so resume/inspect see it.
    ("consolidation", consolidation),
]
STAGE_NAMES = [n for n, _ in STAGE_ORDER]


class Pipeline:
    """Drive the FF-HEDM pipeline from raw zarr to ``Grains.csv``.

    Construct with a ``PipelineConfig`` and (optionally) a list of
    ``DetectorConfig`` objects. If ``detectors`` is omitted, the pipeline
    discovers them from ``config.detectors_json`` or from
    ``DetParams`` rows in ``config.params_file``, falling back to a
    single-detector setup derived from the global geometry keys.
    """

    def __init__(self, *,
                 config: PipelineConfig,
                 detectors: Optional[Sequence[DetectorConfig]] = None) -> None:
        configure_logging(getattr(logging, config.log_level.upper(), logging.INFO))
        self.config = config
        self.detectors: list[DetectorConfig] = list(
            detectors if detectors is not None else self._discover_detectors()
        )
        if not self.detectors:
            raise ValueError("no detectors configured for pipeline")

        # Per-layer state, populated by ``run()``
        self.layer_results: list[LayerResult] = []

        # Convenience: the most recent layer's stage results, surfaced
        # for notebook usage as ``pipe.peakfit_result`` etc.
        self._current_ctx: Optional[StageContext] = None
        self._current_layer_result: Optional[LayerResult] = None

    # --- detector discovery ---

    def _discover_detectors(self) -> list[DetectorConfig]:
        if self.config.detectors_json:
            return DetectorConfig.load_many(self.config.detectors_json)
        # Try DetParams rows in paramstest.
        try:
            dets = DetectorConfig.load_from_paramstest(
                self.config.params_file, zarr_path=self.config.zarr_path,
            )
        except (FileNotFoundError, ValueError):
            dets = []
        if dets:
            # If load_from_paramstest didn't get a zarr but the config has one,
            # patch each detector to use the same zarr.
            if self.config.zarr_path:
                for d in dets:
                    if not d.zarr_path:
                        d.zarr_path = self.config.zarr_path
            return dets
        # Single detector from global geometry keys.
        return [DetectorConfig.single_from_paramstest(
            self.config.params_file, zarr_path=self.config.zarr_path,
        )]

    # --- public top-level run ---

    def run(self) -> list[LayerResult]:
        """Run every selected layer end-to-end. Returns the per-layer results."""
        from .dispatch import configure_dispatch
        n_cpus, n_nodes = configure_dispatch(
            machine=self.config.machine.name,
            n_nodes=self.config.machine.n_nodes,
            n_cpus=self.config.n_cpus,
        )
        # Push back the resolved values so stages see the cluster-aware n_cpus.
        self.config.n_cpus = n_cpus
        self.config.machine.n_nodes = n_nodes

        layer_nrs = self.config.layer_selection.layers()
        LOG.info("=== midas-ff-pipeline: %d detector(s), %d layer(s) [%s] ===",
                 len(self.detectors), len(layer_nrs), ",".join(str(n) for n in layer_nrs))
        for layer_nr in layer_nrs:
            self._run_layer(layer_nr)
        return self.layer_results

    # --- per-layer drive ---

    def _make_context(self, layer_nr: int) -> StageContext:
        layer_dir = self.config.layer_dir(layer_nr)
        layer_dir.mkdir(parents=True, exist_ok=True)
        log_dir = layer_dir / "midas_log"
        log_dir.mkdir(exist_ok=True, parents=True)
        ctx = StageContext(
            config=self.config,
            detectors=self.detectors,
            layer_nr=layer_nr,
            layer_dir=layer_dir,
            log_dir=log_dir,
        )
        # Pre-create per-detector dirs for multi-det runs.
        if ctx.is_multi_detector:
            for det in self.detectors:
                ctx.detector_dir(det).mkdir(parents=True, exist_ok=True)
        return ctx

    def _run_layer(self, layer_nr: int) -> LayerResult:
        ctx = self._make_context(layer_nr)
        provenance = ProvenanceStore(ctx.layer_dir)
        layer_result = LayerResult(layer_nr=layer_nr, layer_dir=str(ctx.layer_dir))
        self._current_ctx = ctx
        self._current_layer_result = layer_result

        # Resolve per-layer seed grains (gaps #3, #4). nf_result_dir wins
        # over a generic --grains-file when ``GrainsLayer{N}.csv`` exists.
        seed_grains = resolve_grains_file_for_layer(
            layer_nr=layer_nr,
            grains_file=self.config.grains_file,
            nf_result_dir=self.config.nf_result_dir,
        )

        # --raw-dir override (gap #5) — patch the *global* parameter file
        # in-place so downstream zip_convert / midas-fit-setup pick up the
        # new RawFolder. Only applied when raw_dir is set; idempotent.
        if self.config.raw_dir:
            apply_raw_dir_override(Path(self.config.params_file), self.config.raw_dir)

        # Determine which stages to run, honoring ``only_stages`` / ``skip_stages``.
        active_stages = self._select_stages()

        # Resume bookkeeping.
        invalidate_from = None
        if self.config.resume == "from" and self.config.resume_from_stage:
            invalidate_from = self.config.resume_from_stage
            if invalidate_from not in [s for s, _ in active_stages]:
                raise ValueError(
                    f"resume_from_stage={invalidate_from!r} not in active stages "
                    f"{[s for s, _ in active_stages]}"
                )

        invalidating = invalidate_from is None
        for stage_name, stage_module in active_stages:
            if not invalidating and stage_name == invalidate_from:
                invalidating = True
            if not invalidating:
                continue                              # pre-resume stage: leave alone
            if invalidate_from and stage_name == invalidate_from:
                provenance.invalidate(stage_name)

            self._run_stage_with_resume(stage_name, stage_module, ctx, provenance, layer_result)

            # Seeding hook: patch the canonical paramstest with the chosen
            # GrainsFile + MinNrSpots=1 once a paramstest exists. Runs after
            # transforms (single-det) and after cross_det_merge (multi-det)
            # so both code paths converge here.
            if seed_grains and stage_name in ("transforms", "cross_det_merge"):
                pt = ctx.layer_dir / "paramstest.txt"
                if pt.exists():
                    patch_params_with_grains(pt, seed_grains)
                    LOG.info("  layer %d: GrainsFile patched into %s", layer_nr, pt)

        self.layer_results.append(layer_result)
        LOG.info("=== layer %d done: %d grains in %.1fs ===",
                 layer_nr, layer_result.n_grains, layer_result.total_duration_s())
        return layer_result

    def _select_stages(self) -> list[tuple[str, object]]:
        if self.config.only_stages:
            allowed = set(self.config.only_stages)
            stages = [(n, m) for n, m in STAGE_ORDER if n in allowed]
        else:
            stages = list(STAGE_ORDER)
        if self.config.skip_stages:
            skip = set(self.config.skip_stages)
            stages = [(n, m) for n, m in stages if n not in skip]
        return stages

    def _run_stage_with_resume(self, stage_name: str, stage_module,
                               ctx: StageContext,
                               provenance: ProvenanceStore,
                               layer_result: LayerResult) -> None:
        # Resume: skip if stage is already complete and outputs intact.
        if self.config.resume in ("auto", "from"):
            expected = stage_module.expected_outputs(ctx)
            recorded = provenance.read(stage_name)
            if recorded and recorded.get("status") == "complete":
                if all(Path(p).exists() for p in expected):
                    LOG.info("  · %s — already complete, skipping (resume)", stage_name)
                    self._set_layer_result_field(layer_result, stage_name, None)
                    # Still rebuild the merged_paramstest pointer for downstream stages.
                    if stage_name == "cross_det_merge":
                        ctx.merged_paramstest = ctx.layer_dir / "paramstest.txt"
                    return

        # Run the stage.
        result = stage_module.run(ctx)
        self._set_layer_result_field(layer_result, stage_name, result)

        # Record into provenance.
        try:
            provenance.record(
                stage_name,
                status="complete",
                started_at=result.started_at,
                finished_at=result.finished_at,
                duration_s=result.duration_s,
                inputs=result.inputs,
                outputs=result.outputs,
                metrics=result.metrics,
            )
        except Exception:
            LOG.exception("provenance record failed for %s", stage_name)

    @staticmethod
    def _set_layer_result_field(layer_result: LayerResult, stage_name: str, value) -> None:
        # Map stage names to LayerResult attribute names.
        attr = {
            "zip_convert": "zip_convert",
            "hkl": "hkl",
            "peakfit": "peakfit",
            "merge_overlaps": "merge_overlaps",
            "calc_radius": "calc_radius",
            "transforms": "transforms",
            "cross_det_merge": "cross_det_merge",
            "global_powder": "global_powder",
            "binning": "binning",
            "indexing": "indexing",
            "refinement": "refinement",
            "process_grains": "process_grains",
            "consolidation": "consolidation",
        }.get(stage_name)
        if attr is None or value is None:
            return
        if hasattr(layer_result, attr):
            setattr(layer_result, attr, value)

    # ----- per-stage methods (notebook style) ------

    def _ensure_ctx(self) -> StageContext:
        if self._current_ctx is None:
            # Default: run the first selected layer through ``_make_context``.
            layer_nr = self.config.layer_selection.start
            self._current_ctx = self._make_context(layer_nr)
            self._current_layer_result = LayerResult(
                layer_nr=layer_nr, layer_dir=str(self._current_ctx.layer_dir),
            )
        return self._current_ctx

    def _run_one(self, stage_name: str, stage_module):
        ctx = self._ensure_ctx()
        provenance = ProvenanceStore(ctx.layer_dir)
        assert self._current_layer_result is not None
        self._run_stage_with_resume(stage_name, stage_module, ctx, provenance,
                                    self._current_layer_result)
        return getattr(self._current_layer_result, stage_name, None)

    def run_zip_convert(self):       return self._run_one("zip_convert", zip_convert)
    def run_hkl(self):               return self._run_one("hkl", hkl)
    def run_peakfit(self):           return self._run_one("peakfit", peakfit)
    def run_merge_overlaps(self):    return self._run_one("merge_overlaps", merge_overlaps)
    def run_calc_radius(self):       return self._run_one("calc_radius", calc_radius)
    def run_transforms(self):        return self._run_one("transforms", transforms)
    def run_cross_det_merge(self):   return self._run_one("cross_det_merge", cross_det_merge)
    def run_global_powder(self):     return self._run_one("global_powder", global_powder)
    def run_binning(self):           return self._run_one("binning", binning)
    def run_indexing(self):          return self._run_one("indexing", index_stage)
    def run_index(self):             return self._run_one("indexing", index_stage)
    def run_refine(self):            return self._run_one("refinement", refine)
    def run_process_grains(self):    return self._run_one("process_grains", process_grains)
    def run_consolidation(self):     return self._run_one("consolidation", consolidation)

    @property
    def layer_dir(self) -> Optional[Path]:
        """Active layer's directory, set after the first stage call."""
        if self._current_ctx is not None:
            return self._current_ctx.layer_dir
        if self.layer_results:
            return Path(self.layer_results[-1].layer_dir)
        return None

    # --- result accessors used by notebooks ---

    @property
    def layer_result(self) -> Optional[LayerResult]:
        """Most-recent layer's roll-up. ``None`` until at least one stage runs."""
        if self._current_layer_result is not None:
            return self._current_layer_result
        return self.layer_results[-1] if self.layer_results else None

    @property
    def hkl_result(self):           return self.layer_result.hkl if self.layer_result else None
    @property
    def peakfit_result(self):       return self.layer_result.peakfit if self.layer_result else None
    @property
    def transforms_result(self):    return self.layer_result.transforms if self.layer_result else None
    @property
    def cross_det_merge_result(self):
        return self.layer_result.cross_det_merge if self.layer_result else None
    @property
    def index_result(self):         return self.layer_result.indexing if self.layer_result else None
    @property
    def refine_result(self):        return self.layer_result.refinement if self.layer_result else None
    @property
    def process_grains_result(self):
        return self.layer_result.process_grains if self.layer_result else None

    # --- inspection helpers ---

    def status(self) -> dict:
        """Summarise resume state of every layer in the run.

        Useful both as the return value for notebooks (`pipe.status()`)
        and as the ``midas-ff-pipeline status`` CLI subcommand backend.
        """
        out = {"result_dir": self.config.result_dir,
               "layers": []}
        for layer_nr in self.config.layer_selection.layers():
            layer_dir = self.config.layer_dir(layer_nr)
            store = ProvenanceStore(layer_dir)
            stages = store.all_stages()
            out["layers"].append({
                "layer_nr": layer_nr,
                "layer_dir": str(layer_dir),
                "stages": stages,
                "complete": all(
                    stages.get(name, {}).get("status") == "complete"
                    for name in STAGE_NAMES
                ),
            })
        return out
