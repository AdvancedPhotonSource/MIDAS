"""Smoke tests — just import everything and check the public API surface."""

import importlib
import pytest


def test_imports():
    for module in [
        "midas_transforms",
        "midas_transforms.cli",
        "midas_transforms.device",
        "midas_transforms.params",
        "midas_transforms.pipeline",
        "midas_transforms.merge",
        "midas_transforms.merge.core",
        "midas_transforms.radius",
        "midas_transforms.radius.core",
        "midas_transforms.fit_setup",
        "midas_transforms.fit_setup.core",
        "midas_transforms.fit_setup.transform",
        "midas_transforms.fit_setup.refine",
        "midas_transforms.bin_data",
        "midas_transforms.bin_data.core",
        "midas_transforms.io",
        "midas_transforms.io.csv",
        "midas_transforms.io.binary",
        "midas_transforms.io.zarr_io",
    ]:
        importlib.import_module(module)


def test_public_api():
    import midas_transforms as mt

    assert hasattr(mt, "merge_overlapping_peaks")
    assert hasattr(mt, "calc_radius")
    assert hasattr(mt, "fit_setup")
    assert hasattr(mt, "bin_data")
    assert hasattr(mt, "Pipeline")
    assert isinstance(mt.__version__, str)


def test_cli_entry_points_callable():
    """The console-script handlers should be importable and callable."""
    from midas_transforms.cli import (
        main, merge_main, radius_main, fit_setup_main, bin_data_main, pipeline_main,
    )
    # `--version` triggers argparse's built-in SystemExit(0); accept that.
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_device_resolution():
    import torch
    from midas_transforms.device import resolve_device, resolve_dtype

    cpu = resolve_device("cpu")
    assert cpu == torch.device("cpu")
    assert resolve_dtype(cpu, None) == torch.float64
    assert resolve_dtype(cpu, "float32") == torch.float32
