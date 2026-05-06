# midas-parsl-configs

Bundled + user-extensible Parsl configs for MIDAS pipelines.

This package owns the cluster Parsl configs that used to live under
`FF_HEDM/workflows/*Config.py` and `NF_HEDM/workflows/*Config.py`, plus
a registry + a SLURM/PBS submit-script-to-config generator so users
don't have to hand-edit Python to bring a new cluster online.

## What it gives you

```python
from midas_parsl_configs import load_config, available_configs

available_configs()
# {'local': 'builtin', 'umich': 'builtin', 'polaris': 'builtin',
#  'mycluster': 'user:/Users/me/.midas/parsl_configs/myclusterConfig.py', ...}

cfg = load_config("polaris", n_nodes=4, n_cpus=64,
                  script_dir="/path/to/run")
parsl.load(config=cfg)
```

`load_config` sets the env vars the bundled configs read at import time
(`MIDAS_SCRIPT_DIR`, `nNodes`, `PROJECT_NAME`, `QUEUE_NAME`,
`CONDA_LOC`) before importing the module so you get a runnable Config
back.

## Bundled configs

| name        | scheduler            | source |
|-------------|----------------------|--------|
| `local`     | ThreadPoolExecutor   | local development |
| `adhoc`     | AdHoc + SSH          | beam-time pool of workers |
| `orthrosnew`| AdHoc + SSH          | APS Orthros (puppy80…) |
| `orthrosall`| AdHoc + SSH          | APS Orthros (pup0100…) |
| `umich`     | SlurmProvider        | UMich GreatLakes |
| `marquette` | SlurmProvider        | Marquette HPC |
| `purdue`    | SlurmProvider        | Purdue Halstead |
| `polaris`   | PBSProProvider       | ALCF Polaris |

## Bringing your own cluster online

```bash
midas-parsl-configs generate /path/to/your_submit.sh --name mycluster
```

This parses `#SBATCH` / `#PBS` directives + the body's `module load` /
`conda activate` / `source` lines, then writes
`~/.midas/parsl_configs/myclusterConfig.py`. From then on
`midas-parsl-configs list` and any pipeline driver that resolves
through the registry (`midas-ff-pipeline`, in particular) can use
`--machine mycluster`.

Override the user dir with `MIDAS_PARSL_CONFIGS_DIR` if you want the
file written somewhere else.

User configs override bundled configs of the same name — useful when
you're iterating on a cluster's setup without losing the upstream
default.

## CLI

```
midas-parsl-configs list                                    # show every visible config + source
midas-parsl-configs generate <submit.sh> --name <short>     # write user config from submit script
midas-parsl-configs generate <submit.sh> --name x --print   # dry run, prints to stdout
midas-parsl-configs show <name>                             # repr() of the resolved Config
midas-parsl-configs path                                    # path to the user config dir
```

## What `generate` translates

SLURM (`#SBATCH`):

  - `--partition / -p`      → `partition`
  - `--account / -A`        → `scheduler_options` (`#SBATCH --account=…`)
  - `--time / -t`           → `walltime`
  - `--nodes / -N`          → `nodes_per_block`
  - `--ntasks-per-node`     → `cores_per_worker`
  - `--cpus-per-task / -c`  → `cores_per_worker`
  - `--gres`                → `scheduler_options` + GPU count detection
  - `--constraint / --qos / --mem` → passed through as `scheduler_options`
  - `--job-name / -J`       → executor label

PBS (`#PBS`):

  - `-q`                    → queue
  - `-A`                    → account
  - `-l walltime=…`         → walltime
  - `-l select=N:ncpus=M:ngpus=K` → `nodes_per_block` + `cores_per_worker` + GPU count
  - `-l filesystems=…`      → scheduler_options
  - `-N`                    → executor label

Anything we can't translate becomes a passthrough scheduler_options
line so you can refine the generated module by hand if needed.
