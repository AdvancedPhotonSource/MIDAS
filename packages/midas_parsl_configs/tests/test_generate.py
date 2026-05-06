"""SLURM/PBS submit-script → parsl Config module generation."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from midas_parsl_configs import generate, registry


# ---- SLURM ------------------------------------------------------------------


SLURM_SCRIPT = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name=midas-ff
    #SBATCH --partition=batch
    #SBATCH --account=mygroup
    #SBATCH --time=12:00:00
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=64
    #SBATCH --gres=gpu:a100:4
    #SBATCH --constraint=nvme

    module load python/3.12
    source activate midas
    cd $SLURM_SUBMIT_DIR
""")


def test_parse_slurm_basic(tmp_path: Path):
    p = tmp_path / "submit.sh"
    p.write_text(SLURM_SCRIPT)
    info = generate.parse_submit_script(p)
    assert info.scheduler == "slurm"
    assert info.partition == "batch"
    assert info.account == "mygroup"
    assert info.walltime == "12:00:00"
    assert info.nodes_per_block == 4
    assert info.cores_per_worker == 64        # ntasks-per-node
    assert info.n_gpus_per_node == 4
    assert info.label == "midas-ff"
    assert any("--gres=gpu:a100:4" in s for s in info.extra_scheduler_options)
    assert any("--constraint=nvme" in s for s in info.extra_scheduler_options)
    # worker_init lines were captured from the body
    assert any("module load" in line for line in info.worker_init)
    assert any("activate midas" in line for line in info.worker_init)


def test_build_slurm_config_module(tmp_path: Path):
    p = tmp_path / "submit.sh"
    p.write_text(SLURM_SCRIPT)
    info = generate.parse_submit_script(p)
    src = generate.build_config_module(name="mycluster", info=info)
    assert "SlurmProvider(" in src
    assert "partition='batch'" in src
    assert "walltime='12:00:00'" in src
    assert "mycluster" in src
    assert "--account=mygroup" in src       # picked up via scheduler_options


def test_write_user_config_round_trip(tmp_path: Path, monkeypatch):
    udir = tmp_path / "u"
    monkeypatch.setenv("MIDAS_PARSL_CONFIGS_DIR", str(udir))
    p = tmp_path / "submit.sh"
    p.write_text(SLURM_SCRIPT)
    info = generate.parse_submit_script(p)
    src = generate.build_config_module(name="mycluster", info=info)
    out = generate.write_user_config("mycluster", src)
    assert out.exists()
    assert out.parent == udir
    # Registry should now find it.
    assert "mycluster" in registry.available_configs()


# ---- PBS --------------------------------------------------------------------


PBS_SCRIPT = textwrap.dedent("""\
    #!/bin/bash
    #PBS -N midas-pbs
    #PBS -q prod
    #PBS -A myalloc
    #PBS -l walltime=4:30:00
    #PBS -l select=2:ncpus=64:ngpus=4
    #PBS -l filesystems=home:eagle:grand

    source /home/me/venv/bin/activate
""")


def test_parse_pbs_basic(tmp_path: Path):
    p = tmp_path / "submit.pbs"
    p.write_text(PBS_SCRIPT)
    info = generate.parse_submit_script(p)
    assert info.scheduler == "pbs"
    assert info.queue == "prod"
    assert info.account == "myalloc"
    assert info.walltime == "4:30:00"
    assert info.nodes_per_block == 2
    assert info.cores_per_worker == 64
    assert info.n_gpus_per_node == 4
    assert info.label == "midas-pbs"


def test_build_pbs_config_module(tmp_path: Path):
    p = tmp_path / "submit.pbs"
    p.write_text(PBS_SCRIPT)
    info = generate.parse_submit_script(p)
    src = generate.build_config_module(name="aurora", info=info)
    assert "PBSProProvider(" in src
    assert "queue='prod'" in src
    assert "account='myalloc'" in src
    assert "walltime='4:30:00'" in src
    assert "filesystems=home:eagle:grand" in src
