"""Smoke tests: package imports cleanly and exposes the planned surface."""

import torch


def test_import_top_level():
    import midas_index

    assert midas_index.__version__
    assert hasattr(midas_index, "Indexer")
    assert hasattr(midas_index, "IndexerParams")
    assert hasattr(midas_index, "IndexerResult")
    assert hasattr(midas_index, "SeedResult")


def test_import_subpackages():
    from midas_index import compute, io  # noqa: F401
    from midas_index.benchmarks import bench_seed  # noqa: F401
    from midas_index.compute import (  # noqa: F401
        binning,
        constants,
        forward_adapter,
        matching,
        orientation_grid,
        position_grid,
        reduce,
        rotation,
        seeds,
    )
    from midas_index.io import binary, csv, output, params  # noqa: F401
    from midas_index.io import consolidated  # noqa: F401


def test_resolve_device_and_dtype_defaults():
    from midas_index.device import resolve_device, resolve_dtype

    cpu = resolve_device("cpu")
    assert cpu.type == "cpu"
    assert resolve_dtype(cpu, None) is torch.float64
    assert resolve_dtype(cpu, "float32") is torch.float32


def test_indexer_construction_defers_to_pipeline():
    from midas_index import Indexer, IndexerParams

    ind = Indexer(IndexerParams(), device="cpu")
    assert ind.device.type == "cpu"
    assert ind.dtype is torch.float64
