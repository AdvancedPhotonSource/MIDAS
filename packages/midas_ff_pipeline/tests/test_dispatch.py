"""Unit tests for cluster dispatch (gap #8) and sr-midas shim (gap #9)."""
from __future__ import annotations

import pytest

from midas_ff_pipeline import dispatch, sr_midas


def test_dispatch_unknown_machine_raises():
    with pytest.raises(ValueError):
        dispatch.configure_dispatch(machine="not-a-real-cluster",
                                    n_nodes=1, n_cpus=4)


def test_dispatch_local_resolves_defaults():
    """parsl may or may not be installed — either way, the function should
    return a sane (n_cpus, n_nodes) pair without raising."""
    n_cpus, n_nodes = dispatch.configure_dispatch(
        machine="local", n_nodes=2, n_cpus=12,
    )
    # User overrides win when > 0.
    assert n_cpus == 12
    assert n_nodes == 2


def test_dispatch_local_falls_back_to_default_when_zero():
    n_cpus, n_nodes = dispatch.configure_dispatch(
        machine="local", n_nodes=0, n_cpus=0,
    )
    # Defaults from dispatch._DEFAULTS["local"] = (8, 1).
    assert n_cpus == 8
    assert n_nodes == 1


def test_sr_midas_status_no_crash_when_unavailable():
    # The smoke check is just that calling the helper doesn't raise.
    class _DummyLogger:
        def info(self, *a, **kw):
            pass
        def warning(self, *a, **kw):
            pass

    sr_midas.log_status(_DummyLogger(), run_sr=False)
    sr_midas.log_status(_DummyLogger(), run_sr=True)
