"""Convert a SLURM / PBS submit script into a Parsl Config module.

Usage::

    from midas_parsl_configs import (
        parse_submit_script, build_config_module, write_user_config,
    )

    info = parse_submit_script("/path/to/submit.sh")
    src  = build_config_module(name="mycluster", info=info)
    path = write_user_config("mycluster", src)
    # path is now ~/.midas/parsl_configs/mycluster_Config.py and
    # midas_parsl_configs.load_config("mycluster") returns its Config.

CLI: ``midas-parsl-configs generate /path/to/submit.sh --name mycluster``.

We extract only the directives that map cleanly onto the Parsl
``SlurmProvider`` / ``PBSProProvider``. Anything we can't interpret
becomes a passthrough ``scheduler_options`` line so the user can refine
the generated module by hand.

Supported directives:

  SLURM (``#SBATCH``):
    -p / --partition         → ``partition``
    -A / --account           → ``scheduler_options`` ('#SBATCH --account=…')
    -t / --time              → ``walltime``
    -N / --nodes             → ``nodes_per_block`` (and ``max_blocks`` ↔ nNodes)
    -n / --ntasks            → cores_per_worker hint
    --cpus-per-task / -c     → cores_per_worker hint
    --ntasks-per-node        → cores_per_worker hint
    --mem                    → kept as a comment in scheduler_options
    --constraint             → scheduler_options
    --gres                   → scheduler_options (and detect GPU count)
    --qos                    → scheduler_options

  PBS (``#PBS``):
    -q                       → queue
    -A                       → account
    -l walltime=…            → walltime
    -l select=…:ncpus=…      → nodes_per_block + cores_per_node
    -l filesystems=…         → scheduler_options
    -N                       → label

Worker-init lines (``module load …``, ``conda activate …``, ``source …``)
are picked up from the body of the script (anything outside ``#SBATCH``
/ ``#PBS`` blocks that's not a shebang or comment) and concatenated into
the ``worker_init`` argument.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---- parser -----------------------------------------------------------------


@dataclass
class SubmitScriptInfo:
    scheduler: str = "slurm"               # "slurm" | "pbs"
    label: Optional[str] = None
    partition: Optional[str] = None         # SLURM partition
    queue: Optional[str] = None             # PBS queue
    account: Optional[str] = None
    walltime: Optional[str] = None
    nodes_per_block: int = 1
    cores_per_worker: int = 1
    n_gpus_per_node: int = 0
    worker_init: list[str] = field(default_factory=list)
    extra_scheduler_options: list[str] = field(default_factory=list)
    raw_directives: list[str] = field(default_factory=list)
    source: Optional[str] = None


_SBATCH_RE = re.compile(r"^\s*#SBATCH\s+(.*?)\s*$", re.IGNORECASE)
_PBS_RE = re.compile(r"^\s*#PBS\s+(.*?)\s*$", re.IGNORECASE)
_GRES_GPU_RE = re.compile(r"gpu(?::[^:=]+)?:(\d+)", re.IGNORECASE)


def _split_kv(arg: str) -> tuple[str, Optional[str]]:
    """Split SLURM-style ``--key=value`` or ``-k value`` directives.

    Returns ``(key_without_dashes, value_or_None)``. ``value`` is ``None``
    when the directive is a bare flag (e.g. ``--exclusive``).
    """
    arg = arg.strip()
    if "=" in arg and arg.startswith("-"):
        head, _, tail = arg.partition("=")
        return head.lstrip("-"), tail.strip()
    if arg.startswith("--"):
        return arg.lstrip("-"), None
    if arg.startswith("-") and len(arg) >= 2:
        # short form: ``-p shared``
        return arg[1], None
    return arg, None


def _parse_sbatch_directives(directive: str, info: SubmitScriptInfo) -> None:
    """Apply the contents of one ``#SBATCH …`` line."""
    info.raw_directives.append(f"#SBATCH {directive}")
    parts = directive.split()
    if not parts:
        return

    # Recombine so '-p shared' and '--partition=shared' are both handled.
    if parts[0] in ("-p",) and len(parts) >= 2:
        info.partition = parts[1]
        return
    if parts[0] in ("-A",) and len(parts) >= 2:
        info.account = parts[1]
        return
    if parts[0] in ("-t",) and len(parts) >= 2:
        info.walltime = parts[1]
        return
    if parts[0] in ("-N",) and len(parts) >= 2:
        try:
            info.nodes_per_block = int(parts[1])
        except ValueError:
            pass
        return
    if parts[0] in ("-J", "-J="):
        if len(parts) >= 2:
            info.label = parts[1]
        return
    if parts[0] in ("-c",) and len(parts) >= 2:
        try:
            info.cores_per_worker = max(info.cores_per_worker, int(parts[1]))
        except ValueError:
            pass
        return

    key, value = _split_kv(directive)
    key = key.lower()
    if key == "partition" and value:
        info.partition = value
    elif key == "account" and value:
        info.account = value
    elif key in ("time", "walltime") and value:
        info.walltime = value
    elif key in ("nodes",) and value:
        try:
            info.nodes_per_block = int(value)
        except ValueError:
            pass
    elif key in ("ntasks",) and value:
        try:
            info.cores_per_worker = max(info.cores_per_worker, int(value))
        except ValueError:
            pass
    elif key in ("cpus-per-task",) and value:
        try:
            info.cores_per_worker = max(info.cores_per_worker, int(value))
        except ValueError:
            pass
    elif key in ("ntasks-per-node",) and value:
        try:
            info.cores_per_worker = max(info.cores_per_worker, int(value))
        except ValueError:
            pass
    elif key in ("gres",) and value:
        m = _GRES_GPU_RE.search(value)
        if m:
            info.n_gpus_per_node = int(m.group(1))
        info.extra_scheduler_options.append(f"#SBATCH --gres={value}")
    elif key in ("constraint",) and value:
        info.extra_scheduler_options.append(f"#SBATCH --constraint={value}")
    elif key in ("qos",) and value:
        info.extra_scheduler_options.append(f"#SBATCH --qos={value}")
    elif key in ("mem", "mem-per-cpu") and value:
        info.extra_scheduler_options.append(f"#SBATCH --{key}={value}")
    elif key in ("job-name", "j") and value:
        info.label = value
    elif key in ("exclusive",):
        info.extra_scheduler_options.append("#SBATCH --exclusive")


def _parse_pbs_directives(directive: str, info: SubmitScriptInfo) -> None:
    info.raw_directives.append(f"#PBS {directive}")
    parts = directive.split(None, 1)
    if not parts:
        return
    flag = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if flag == "-q" and rest:
        info.queue = rest.strip()
        return
    if flag == "-A" and rest:
        info.account = rest.strip()
        return
    if flag == "-N" and rest:
        info.label = rest.strip()
        return
    if flag == "-l" and rest:
        # rest looks like 'walltime=4:00:00' or 'select=2:ncpus=64:mpiprocs=64'
        for piece in rest.split(","):
            kv = piece.strip()
            if kv.startswith("walltime="):
                info.walltime = kv.split("=", 1)[1]
            elif kv.startswith("select="):
                # select=2:ncpus=64
                m_nodes = re.match(r"select=(\d+)", kv)
                if m_nodes:
                    try:
                        info.nodes_per_block = int(m_nodes.group(1))
                    except ValueError:
                        pass
                m_cpus = re.search(r"ncpus=(\d+)", rest)
                if m_cpus:
                    try:
                        info.cores_per_worker = max(info.cores_per_worker,
                                                    int(m_cpus.group(1)))
                    except ValueError:
                        pass
                m_gpus = re.search(r"ngpus=(\d+)", rest)
                if m_gpus:
                    try:
                        info.n_gpus_per_node = int(m_gpus.group(1))
                    except ValueError:
                        pass
            elif kv.startswith("filesystems="):
                info.extra_scheduler_options.append(f"#PBS -l {kv}")


def parse_submit_script(path: str | Path) -> SubmitScriptInfo:
    """Parse a SLURM or PBS submit script.

    Auto-detects SLURM vs PBS by which set of directives shows up first.
    Lines outside the directive header that aren't shebangs/comments
    accumulate into ``info.worker_init`` so ``module load`` / ``conda
    activate`` lines round-trip into the parsl ``worker_init`` argument.
    """
    text = Path(path).read_text()
    info = SubmitScriptInfo(source=str(Path(path).resolve()))

    in_header = True
    body_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        m_sbatch = _SBATCH_RE.match(line)
        m_pbs = _PBS_RE.match(line)
        if m_sbatch:
            in_header = False                     # body comes after the header block
            info.scheduler = "slurm"
            _parse_sbatch_directives(m_sbatch.group(1), info)
            continue
        if m_pbs:
            in_header = False
            info.scheduler = "pbs"
            _parse_pbs_directives(m_pbs.group(1), info)
            continue
        if in_header and (line.startswith("#") or not line.strip()):
            continue                              # shebang / blank / comment
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        body_lines.append(line.strip())

    info.worker_init = body_lines
    return info


# ---- module emitter ---------------------------------------------------------


_SLURM_TEMPLATE = '''\
"""Auto-generated parsl config from {source}.

Edit by hand if anything looks off — all SLURM directives that we
couldn't translate live in ``scheduler_options``.
"""
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
import os

SCRIPTDIR = os.environ.get("MIDAS_SCRIPT_DIR")
nNodes = int(os.environ.get("nNodes", "{nodes}"))


{varname} = Config(
    executors=[
        HighThroughputExecutor(
            label={label!r},
            cores_per_worker={cores},
            max_workers_per_node=1,
            provider=SlurmProvider(
                nodes_per_block={nodes},
                init_blocks=1,
                min_blocks=1,
                max_blocks=nNodes,
{partition_block}                scheduler_options={scheduler_options!r},
                worker_init={worker_init!r},
                walltime={walltime!r},
                cmd_timeout=120,
            ),
        )
    ]
)

# Alias so ``midas_parsl_configs.load_config({short!r})`` finds the object
# under the conventional name.
config = {varname}
'''


_PBS_TEMPLATE = '''\
"""Auto-generated parsl config from {source}.

Edit by hand if anything looks off — all PBS directives that we couldn't
translate live in ``scheduler_options``.
"""
from parsl.config import Config
from parsl.providers import PBSProProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
import os

SCRIPTDIR = os.environ.get("MIDAS_SCRIPT_DIR")
nNodes = int(os.environ.get("nNodes", "{nodes}"))


{varname} = Config(
    executors=[
        HighThroughputExecutor(
            label={label!r},
            cores_per_worker={cores},
            max_workers_per_node=1,
            provider=PBSProProvider(
                launcher=MpiExecLauncher(bind_cmd="--cpu-bind",
                                         overrides="--depth={cores} --ppn 1"),
                queue={queue!r},
                account={account!r},
                nodes_per_block=nNodes,
                init_blocks=1,
                min_blocks=1,
                max_blocks=1,
                scheduler_options={scheduler_options!r},
                worker_init={worker_init!r},
                walltime={walltime!r},
                cmd_timeout=120,
            ),
        )
    ]
)

config = {varname}
'''


def build_config_module(*, name: str, info: SubmitScriptInfo) -> str:
    """Return the source of a Parsl config module derived from ``info``."""
    short = name.strip()
    varname = f"{short}Config"
    label = info.label or short
    walltime = info.walltime or "01:00:00"
    cores = max(info.cores_per_worker or 1, 1)
    nodes = max(info.nodes_per_block or 1, 1)
    sched_opts = list(info.extra_scheduler_options)
    if info.scheduler == "slurm" and info.account:
        sched_opts.insert(0, f"#SBATCH --account={info.account}")
    sched_opts_str = "\n".join(sched_opts)

    # Worker-init: keep it terse.
    worker_init = "; ".join(info.worker_init).strip() or "true"

    if info.scheduler == "pbs":
        return _PBS_TEMPLATE.format(
            source=info.source or "<unknown>",
            varname=varname,
            label=label,
            cores=cores,
            nodes=nodes,
            queue=info.queue or "default",
            account=info.account or "",
            scheduler_options=sched_opts_str,
            worker_init=worker_init,
            walltime=walltime,
        )

    # SLURM (default).
    if info.partition:
        partition_block = f"                partition={info.partition!r},\n"
    else:
        partition_block = ""

    return _SLURM_TEMPLATE.format(
        source=info.source or "<unknown>",
        varname=varname,
        label=label,
        cores=cores,
        nodes=nodes,
        partition_block=partition_block,
        scheduler_options=sched_opts_str,
        worker_init=worker_init,
        walltime=walltime,
        short=short,
    )


def write_user_config(name: str, source: str,
                      *, target_dir: Optional[Path] = None) -> Path:
    """Write a generated module into the user config dir.

    Returns the path. Filename is ``<Name>Config.py`` so the registry
    can pick it up by short name. Uses ``user_configs_dir()`` unless an
    explicit ``target_dir`` is passed.
    """
    from .registry import user_configs_dir
    target = (target_dir or user_configs_dir()).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    out = target / f"{name.strip()}Config.py"
    out.write_text(textwrap.dedent(source).rstrip() + "\n")
    return out
