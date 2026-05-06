"""Auto-detect resolver tests for the NF pipeline CLI."""
from __future__ import annotations

from midas_nf_pipeline.cli import _resolve_nf_dtype


def test_resolve_nf_dtype_explicit_aliases():
    # Explicit fp32/fp64 (and float32/float64) all pass through to the
    # canonical torch-friendly form.
    assert _resolve_nf_dtype("cpu", "fp32") == "float32"
    assert _resolve_nf_dtype("cpu", "fp64") == "float64"
    assert _resolve_nf_dtype("cuda", "float32") == "float32"
    assert _resolve_nf_dtype("cuda", "float64") == "float64"


def test_resolve_nf_dtype_auto_explicit_device():
    # When the device is explicit (not 'auto'), dtype-auto resolves
    # purely from the device.
    assert _resolve_nf_dtype("cuda", "auto") == "float32"
    assert _resolve_nf_dtype("mps", "auto") == "float32"
    assert _resolve_nf_dtype("cpu", "auto") == "float64"


def test_resolve_nf_dtype_auto_device_auto():
    # When both are auto, the resolver introspects torch.cuda.is_available().
    # This test must pass on any machine, so we just check the result is
    # one of the two valid strings.
    out = _resolve_nf_dtype("auto", "auto")
    assert out in ("float32", "float64")
