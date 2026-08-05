# Machine-local NMR environments

These scripts keep Python environments on each VM's local disk while leaving
code, datasets, checkpoints, and benchmark outputs on the shared project
filesystem.

Run the setup commands below from the repository root.

They create four independent environments:

| Directory | Python | Purpose |
| --- | --- | --- |
| `nmr_venv` | 3.11 | Canonical conversion, project analysis, and benchmark notebooks |
| `ultranmr_venv` | 3.11 | UltraNMR |
| `nmrtrans_venv` | 3.10 | NMRTrans |
| `nmrpeak_venv` | 3.10 | NMRPeak and pinned Uni-Core |

PyArrow is installed in every environment. It streams canonical Parquet input
and writes embedding batches incrementally, including when a benchmark runs
outside the main project environment.

RDKit is installed in the main `nmr_venv`; use that environment for schema-v2
canonicalization because its structure fields are derived from SMILES.

## CPU VM

Choose a path that is physically stored on the VM's local disk:

```bash
./scripts/envs_scr/setup_cpu_envs.sh /home/nicola-greco/.venvs/nmr
```

## GPU VM

The NVIDIA driver and a working `nvidia-smi` must already be present in the VM
image. The script lets `uv` choose a compatible PyTorch CUDA backend:

```bash
./scripts/envs_scr/setup_gpu_envs.sh /home/nicola-greco/.venvs/nmr
```

To force a backend:

```bash
./scripts/envs_scr/setup_gpu_envs.sh /home/nicola-greco/.venvs/nmr \
  --torch-backend cu121
```

Do not force one backend for all environments unless every pinned PyTorch
version publishes wheels for it. The default `auto` is safer because UltraNMR,
NMRPeak, and NMRTrans use different PyTorch releases.

## Install or repair one environment

```bash
./scripts/envs_scr/setup_cpu_envs.sh /home/nicola-greco/.venvs/nmr \
  --only ultranmr
```

Valid names are `main`, `ultranmr`, `nmrtrans`, `nmrpeak`, and `shell`.

Existing valid environments are reused and updated. Invalid directories are
never deleted automatically: move them aside and rerun the script.

## Activate an environment

By default, either setup script installs a machine-local `nmr-env` shell
function in `~/.bashrc`. Open a new terminal, or load it immediately:

```bash
source ~/.bashrc
```

Then use:

```bash
nmr-env main
nmr-env ultranmr
nmr-env nmrtrans
nmr-env nmrpeak
nmr-env off
nmr-env list
```

Switching environments automatically deactivates the current one. The helper
configuration is stored in `~/.config/nmr/envs.sh`, so each VM can point to its
own local environment root.

To leave `~/.bashrc` unchanged:

```bash
./scripts/envs_scr/setup_cpu_envs.sh /path/to/local/venvs --no-shell-helper
```

If the environments already exist and you only want to install or update the
helper:

```bash
./scripts/envs_scr/setup_cpu_envs.sh /home/nicola-greco/.venvs/nmr --only shell
```

## Notes

- The scripts install `uv` through Astral's official installer if needed.
- They do not install system packages, NVIDIA drivers, or a CUDA toolkit.
- PyTorch wheels provide their CUDA runtime, but the host NVIDIA driver must be
  compatible.
- NMRPeak's upstream `apex==0.9.10.dev0` requirement is excluded because the
  package under that name on PyPI is an unrelated web toolkit. NMRPeak has no
  direct Apex import.
- Uni-Core is pinned to commit
  `ace6fae1c8479a9751f2bb1e1d6e4047427bc134`. Its optional fused CUDA
  extensions remain disabled, matching the simple NMRPeak installation path and
  avoiding a hard dependency on a local `nvcc` toolkit.
- Installed package inventories are written under `VENV_ROOT/manifests/`.
