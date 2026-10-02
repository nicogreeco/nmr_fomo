# NMR environments

These scripts create separate Python environments for the repository.

Run the setup commands below from the repository root.

They create five independent environments:

| Directory | Python | Purpose |
| --- | --- | --- |
| `nmr_venv` | 3.11 | Canonical conversion, project analysis, and benchmark notebooks |
| `ultranmr_venv` | 3.11 | UltraNMR |
| `nmrtrans_venv` | 3.10 | NMRTrans |
| `nmrpeak_venv` | 3.10 | NMRPeak and pinned Uni-Core |
| `unimol2_venv` | 3.11 | UniMol2 molecular embeddings |

PyArrow is installed in every environment. It streams canonical Parquet input
and writes embedding batches incrementally, including when a benchmark runs
outside the main project environment.

RDKit, DVC with the S3 remote extra, and the YAML libraries used by the pipeline
generator are installed in the main `nmr_venv`. Use that environment for
schema-v2 canonicalization and for `dvc dag`, `dvc repro`, and `dvc push`.

## CPU VM

```bash
./scripts/envs_scr/setup_cpu_envs.sh /path/to/nmr-envs
```

## GPU VM

The NVIDIA driver and a working `nvidia-smi` must already be present in the VM
image. The script lets `uv` choose a compatible PyTorch CUDA backend:

```bash
./scripts/envs_scr/setup_gpu_envs.sh /path/to/nmr-envs
```

To force a backend:

```bash
./scripts/envs_scr/setup_gpu_envs.sh /path/to/nmr-envs \
  --torch-backend cu121
```

Do not force one backend for all environments unless every pinned PyTorch
version publishes wheels for it. The default `auto` is safer because UltraNMR,
NMRPeak, NMRTrans, and UniMol2 may use different PyTorch releases.

## Install or repair one environment

```bash
./scripts/envs_scr/setup_cpu_envs.sh /path/to/nmr-envs \
  --only ultranmr
```

Valid names are `main`, `ultranmr`, `nmrtrans`, `nmrpeak`, `unimol2`, and `shell`.

Existing valid environments are reused and updated. Invalid directories are
never deleted automatically: move them aside and rerun the script.

## Activate an environment

By default, either setup script installs an `nmr-env` shell function in
`~/.bashrc`. Open a new terminal, or load it immediately:

```bash
source ~/.bashrc
```

Then use:

```bash
nmr-env main
nmr-env ultranmr
nmr-env nmrtrans
nmr-env nmrpeak
nmr-env unimol2
nmr-env off
nmr-env list
```

Switching environments automatically deactivates the current one. The helper
configuration is stored in `~/.config/nmr/envs.sh`.

To leave `~/.bashrc` unchanged:

```bash
./scripts/envs_scr/setup_cpu_envs.sh /path/to/nmr-envs --no-shell-helper
```

If the environments already exist and you only want to install or update the
helper:

```bash
./scripts/envs_scr/setup_cpu_envs.sh /path/to/nmr-envs --only shell
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
- UniMol2 installs `unimol_tools==0.1.6`. Its official model weights are
  downloaded on first use; set `UNIMOL_WEIGHT_DIR` to place that cache on a
  specific local or shared path.
- Installed package inventories are written under `VENV_ROOT/manifests/`.
- Re-run the main-environment setup after data-pipeline requirement changes;
  existing valid environments are updated in place.
