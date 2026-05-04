"""Tests for device/dtype resolution."""

from __future__ import annotations

import os

import pytest
import torch

from midas_nf_preprocess.device import (
    ENV_DEVICE,
    ENV_DTYPE,
    apply_cpu_threads,
    resolve_device,
    resolve_dtype,
)


def test_resolve_device_explicit_string():
    d = resolve_device("cpu")
    assert d.type == "cpu"


def test_resolve_device_explicit_torch_device():
    in_d = torch.device("cpu")
    out_d = resolve_device(in_d)
    assert out_d is in_d


def test_resolve_device_env_var(monkeypatch):
    monkeypatch.setenv(ENV_DEVICE, "cpu")
    d = resolve_device(None)
    assert d.type == "cpu"


def test_resolve_device_arg_overrides_env(monkeypatch):
    monkeypatch.setenv(ENV_DEVICE, "cpu")
    d = resolve_device("cpu")  # explicit
    assert d.type == "cpu"


def test_resolve_dtype_default_cpu():
    d = resolve_dtype(torch.device("cpu"), None)
    assert d == torch.float64


def test_resolve_dtype_default_cuda_pretend():
    # Even without CUDA available, calling with a cuda device should pick fp32.
    d = resolve_dtype(torch.device("cuda"), None)
    assert d == torch.float32


def test_resolve_dtype_default_mps_pretend():
    d = resolve_dtype(torch.device("mps"), None)
    assert d == torch.float32


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("single", torch.float32),
        ("float64", torch.float64),
        ("fp64", torch.float64),
        ("double", torch.float64),
        ("FP32", torch.float32),
        ("Float64", torch.float64),
    ],
)
def test_resolve_dtype_string_aliases(alias, expected):
    d = resolve_dtype(torch.device("cpu"), alias)
    assert d == expected


def test_resolve_dtype_torch_dtype():
    d = resolve_dtype(torch.device("cpu"), torch.float32)
    assert d == torch.float32


def test_resolve_dtype_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype(torch.device("cpu"), "int32")


def test_resolve_dtype_env_var(monkeypatch):
    monkeypatch.setenv(ENV_DTYPE, "fp32")
    d = resolve_dtype(torch.device("cpu"), None)
    assert d == torch.float32


def test_apply_cpu_threads_no_op_on_gpu():
    # Should not raise even with a non-cpu device; just verifies no-op semantics.
    apply_cpu_threads(4, torch.device("cuda"))
    apply_cpu_threads(4, torch.device("mps"))


def test_apply_cpu_threads_sets_on_cpu():
    original = torch.get_num_threads()
    try:
        apply_cpu_threads(2, torch.device("cpu"))
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(original)
