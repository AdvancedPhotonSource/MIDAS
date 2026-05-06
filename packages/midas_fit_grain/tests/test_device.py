"""Device / dtype precedence tests."""

from __future__ import annotations

import os

import pytest
import torch

from midas_fit_grain.device import (
    apply_cpu_threads,
    resolve_device,
    resolve_dtype,
)


def test_explicit_device_wins(monkeypatch):
    monkeypatch.setenv("MIDAS_FIT_GRAIN_DEVICE", "cuda")
    assert resolve_device("cpu").type == "cpu"


def test_env_device_used(monkeypatch):
    monkeypatch.setenv("MIDAS_FIT_GRAIN_DEVICE", "cpu")
    assert resolve_device(None).type == "cpu"


def test_auto_device(monkeypatch):
    monkeypatch.delenv("MIDAS_FIT_GRAIN_DEVICE", raising=False)
    d = resolve_device(None)
    assert d.type in ("cpu", "cuda", "mps")


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("single", torch.float32),
        ("float64", torch.float64),
        ("fp64", torch.float64),
        ("double", torch.float64),
        ("DOUBLE", torch.float64),
    ],
)
def test_dtype_aliases(alias, expected):
    assert resolve_dtype(torch.device("cpu"), alias) == expected


def test_dtype_default_per_device():
    assert resolve_dtype(torch.device("cpu"), None) == torch.float64
    assert resolve_dtype(torch.device("cuda"), None) == torch.float32
    assert resolve_dtype(torch.device("mps"), None) == torch.float32


def test_dtype_unknown_raises():
    with pytest.raises(ValueError):
        resolve_dtype(torch.device("cpu"), "bf16")


def test_apply_cpu_threads_cpu_only():
    initial = torch.get_num_threads()
    apply_cpu_threads(2, torch.device("cpu"))
    assert torch.get_num_threads() == 2
    apply_cpu_threads(99, torch.device("cuda"))  # should be a no-op
    assert torch.get_num_threads() == 2
    torch.set_num_threads(initial)
