"""Smoke tests for the ``midas-parsl-configs`` CLI."""
from __future__ import annotations

import textwrap
from pathlib import Path

from midas_parsl_configs.cli import main


def test_cli_list(capsys):
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    for n in ("local", "umich", "polaris"):
        assert n in out


def test_cli_path_prints_user_dir(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setenv("MIDAS_PARSL_CONFIGS_DIR", str(tmp_path))
    rc = main(["path"])
    assert rc == 0
    assert str(tmp_path) in capsys.readouterr().out


def test_cli_generate_writes_file(tmp_path: Path, monkeypatch):
    submit = tmp_path / "submit.sh"
    submit.write_text(textwrap.dedent("""\
        #!/bin/bash
        #SBATCH --partition=shared
        #SBATCH --account=acme
        #SBATCH --time=01:00:00
        #SBATCH --nodes=2
        module load python
    """))
    user_dir = tmp_path / "u"
    monkeypatch.setenv("MIDAS_PARSL_CONFIGS_DIR", str(user_dir))
    rc = main(["generate", str(submit), "--name", "demo"])
    assert rc == 0
    assert (user_dir / "demoConfig.py").exists()


def test_cli_generate_print_only(tmp_path: Path, capsys):
    submit = tmp_path / "submit.sh"
    submit.write_text(
        "#!/bin/bash\n"
        "#SBATCH --partition=shared\n"
        "#SBATCH --time=00:30:00\n"
    )
    rc = main(["generate", str(submit), "--name", "demo", "--print"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SlurmProvider" in out
    assert "shared" in out
