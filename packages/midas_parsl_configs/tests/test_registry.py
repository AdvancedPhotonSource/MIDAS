"""Registry: builtin discovery, env-var setup, user override priority."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from midas_parsl_configs import registry


def test_available_includes_all_builtins():
    cfgs = registry.available_configs()
    for name in ["local", "umich", "purdue", "polaris",
                 "marquette", "orthrosall", "orthrosnew", "adhoc"]:
        assert name in cfgs


def test_load_local_config_returns_parsl_config():
    cfg = registry.load_config("local", n_nodes=1, n_cpus=4,
                               script_dir="/tmp")
    # We don't depend on parsl being installed here, but the bundled
    # localConfig.py imports it at top level — so if this import works
    # we already know parsl is available.
    from parsl.config import Config  # type: ignore
    assert isinstance(cfg, Config)


def test_load_unknown_raises():
    with pytest.raises(KeyError):
        registry.load_config("does-not-exist", n_nodes=1, n_cpus=4)


def test_load_orthrosnew_picks_correct_attribute():
    """orthrosAllConfig.py defines BOTH ``orthrosNewConfig`` and
    ``orthrosAllConfig``. Verify each short name resolves to the matching
    module attribute, not the wrong one.

    The bundled module imports ``AdHocProvider`` / ``SSHChannel`` which
    are removed in parsl 2024+, so we skip when those aren't available.
    """
    try:
        from parsl.providers import AdHocProvider  # noqa: F401
        from parsl.channels import SSHChannel  # noqa: F401
    except ImportError:
        pytest.skip("parsl ≥ 2024 dropped AdHocProvider/SSHChannel; "
                    "bundled orthros config is unsupported here.")
    cfg_new = registry.load_config("orthrosnew", n_nodes=1, n_cpus=4,
                                   script_dir="/tmp")
    cfg_all = registry.load_config("orthrosall", n_nodes=1, n_cpus=4,
                                   script_dir="/tmp")
    labels_new = [e.label for e in cfg_new.executors]
    labels_all = [e.label for e in cfg_all.executors]
    assert labels_new == ["orthrosnew"]
    assert labels_all == ["orthrosall"]


def test_user_dir_override_takes_priority(tmp_path: Path, monkeypatch):
    udir = tmp_path / "userconfigs"
    udir.mkdir()
    # Drop a fake "localConfig" that exposes a sentinel attribute
    # so we can prove load_config picked the user version, not the bundled
    # ThreadPoolExecutor one.
    (udir / "localConfig.py").write_text(
        "from parsl.config import Config\n"
        "from parsl.executors import ThreadPoolExecutor\n"
        "\n"
        "localConfig = Config(executors=[ThreadPoolExecutor(label='user-override')])\n"
        "config = localConfig\n"
    )
    monkeypatch.setenv("MIDAS_PARSL_CONFIGS_DIR", str(udir))
    cfg = registry.load_config("local", n_nodes=1, n_cpus=4)
    assert any(e.label == "user-override" for e in cfg.executors)


def test_load_sets_required_env_vars(monkeypatch):
    monkeypatch.delenv("MIDAS_SCRIPT_DIR", raising=False)
    monkeypatch.delenv("nNodes", raising=False)
    registry.load_config("local", n_nodes=4, n_cpus=8, script_dir="/tmp/foo")
    assert os.environ.get("MIDAS_SCRIPT_DIR") == "/tmp/foo"
    assert os.environ.get("nNodes") == "4"


def test_normalize_strips_config_suffix():
    assert registry._normalize("UMichConfig") == "umich"
    assert registry._normalize("uMichConfig.py") == "umich"
    assert registry._normalize("UMICH") == "umich"
