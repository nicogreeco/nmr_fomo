#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
    cat <<'EOF'
Usage:
  setup_cpu_envs.sh VENV_ROOT [options]
  setup_gpu_envs.sh VENV_ROOT [options]

Create machine-local Python environments for the NMR project and paper models.

Arguments:
  VENV_ROOT                Local directory that will contain all environments.

Options:
  --project-root PATH      Project root containing models/ (default: script parent).
  --only NAME              Install only one environment:
                           main, ultranmr, nmrtrans, nmrpeak, unimol2, shell, or all.
  --torch-backend NAME     uv PyTorch backend. Defaults to cpu on the CPU script
                           and auto on the GPU script.
  --no-shell-helper        Do not install the nmr-env helper in ~/.bashrc.
  -h, --help               Show this help.

Environment overrides:
  MAIN_TORCH_VERSION       Default: 2.6.0
  UNICORE_REF              Default: pinned commit verified with NMRPeak
  ALLOW_SHARED_VENVS=1     Allow VENV_ROOT inside/on the project filesystem.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '\n[%s] %s\n' "$MODE" "$*"
}

MODE="${1:-}"
case "$MODE" in
    cpu|gpu) shift ;;
    -h|--help|"")
        usage
        exit 0
        ;;
    *) die "Internal mode must be cpu or gpu." ;;
esac

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

VENV_ROOT="${1:-}"
[[ -n "$VENV_ROOT" ]] || {
    usage
    exit 2
}
shift

PROJECT_ROOT="$DEFAULT_PROJECT_ROOT"
ONLY="all"
INSTALL_SHELL_HELPER="1"
TORCH_BACKEND="cpu"
[[ "$MODE" == "gpu" ]] && TORCH_BACKEND="auto"

while (($#)); do
    case "$1" in
        --project-root)
            [[ $# -ge 2 ]] || die "--project-root requires a path."
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --only)
            [[ $# -ge 2 ]] || die "--only requires a name."
            ONLY="$2"
            shift 2
            ;;
        --torch-backend)
            [[ $# -ge 2 ]] || die "--torch-backend requires a backend."
            TORCH_BACKEND="$2"
            shift 2
            ;;
        --no-shell-helper)
            INSTALL_SHELL_HELPER="0"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

case "$ONLY" in
    all|main|ultranmr|nmrtrans|nmrpeak|unimol2|shell) ;;
    *) die "--only must be main, ultranmr, nmrtrans, nmrpeak, unimol2, shell, or all." ;;
esac

if [[ "$ONLY" == "shell" && "$INSTALL_SHELL_HELPER" != "1" ]]; then
    die "--only shell cannot be combined with --no-shell-helper."
fi

command -v git >/dev/null 2>&1 || die "git is required. Install it before running this script."

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    command -v curl >/dev/null 2>&1 || die "uv is missing and curl is unavailable."
    note "uv is missing; installing it with Astral's official installer."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [[ -x "${HOME}/.local/bin/uv" ]] || die "uv installation completed but its executable was not found."
    UV_BIN="${HOME}/.local/bin/uv"
fi

mkdir -p "$PROJECT_ROOT" "$VENV_ROOT"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"
VENV_ROOT="$(cd -- "$VENV_ROOT" && pwd -P)"

case "${VENV_ROOT}/" in
    "${PROJECT_ROOT}/"*)
        [[ "${ALLOW_SHARED_VENVS:-0}" == "1" ]] ||
            die "VENV_ROOT is inside the shared project. Choose a machine-local path."
        ;;
esac

PROJECT_DEVICE="$(stat -c '%d' "$PROJECT_ROOT")"
VENV_DEVICE="$(stat -c '%d' "$VENV_ROOT")"
if [[ "$PROJECT_DEVICE" == "$VENV_DEVICE" && "${ALLOW_SHARED_VENVS:-0}" != "1" ]]; then
    die "VENV_ROOT appears to be on the same filesystem as the shared project. Set ALLOW_SHARED_VENVS=1 only if this is intentional."
fi

if [[ "$MODE" == "gpu" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 ||
        die "nvidia-smi is unavailable. Install/enable the NVIDIA driver on the GPU VM first."
    nvidia-smi >/dev/null ||
        die "nvidia-smi cannot communicate with a GPU."
fi

MAIN_TORCH_VERSION="${MAIN_TORCH_VERSION:-2.6.0}"
UNICORE_REF="${UNICORE_REF:-ace6fae1c8479a9751f2bb1e1d6e4047427bc134}"

MAIN_ENV="${VENV_ROOT}/nmr_venv"
ULTRANMR_ENV="${VENV_ROOT}/ultranmr_venv"
NMRTRANS_ENV="${VENV_ROOT}/nmrtrans_venv"
NMRPEAK_ENV="${VENV_ROOT}/nmrpeak_venv"
UNIMOL2_ENV="${VENV_ROOT}/unimol2_venv"
SOURCE_ROOT="${VENV_ROOT}/.bootstrap-sources"
MANIFEST_ROOT="${VENV_ROOT}/manifests"

ULTRANMR_REPO="${PROJECT_ROOT}/models/UltraNMR"
NMRTRANS_REPO="${PROJECT_ROOT}/models/NMRTrans"
NMRPEAK_REPO="${PROJECT_ROOT}/models/NMRPeak"

ensure_repo() {
    local repo="$1"
    local marker="$2"
    [[ -f "${repo}/${marker}" ]] ||
        die "Expected repository file not found: ${repo}/${marker}"
}

ensure_venv() {
    local env_path="$1"
    local python_version="$2"

    if [[ -e "$env_path" && ! -x "${env_path}/bin/python" ]]; then
        die "${env_path} exists but is not a valid virtual environment. Move it aside and retry."
    fi

    if [[ ! -e "$env_path" ]]; then
        note "Creating ${env_path} with Python ${python_version}"
        "$UV_BIN" venv --python "$python_version" "$env_path"
    else
        note "Reusing ${env_path}"
    fi

    local actual_version
    actual_version="$("${env_path}/bin/python" -c 'import platform; print(platform.python_version())')"
    [[ "$actual_version" == "${python_version}."* ]] ||
        die "${env_path} uses Python ${actual_version}; expected ${python_version}. Move it aside and retry."
}

install_build_tools() {
    local python="$1"
    "$UV_BIN" pip install --python "$python" setuptools wheel
}

install_parquet_support() {
    local python="$1"
    "$UV_BIN" pip install --python "$python" "pyarrow>=17,<26"
}

install_torch() {
    local python="$1"
    shift
    "$UV_BIN" pip install \
        --python "$python" \
        --torch-backend "$TORCH_BACKEND" \
        "$@"
}

verify_torch() {
    local python="$1"
    local env_name="$2"

    "$python" -c 'import torch; print("torch", torch.__version__, "cuda build", torch.version.cuda, "cuda available", torch.cuda.is_available())'

    if [[ "$MODE" == "gpu" ]]; then
        "$python" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else "PyTorch cannot access the GPU")'
    else
        "$python" -c 'import torch, sys; sys.exit(0 if torch.version.cuda is None else "Expected a CPU-only PyTorch build")'
    fi

    note "${env_name} validation passed"
}

record_manifest() {
    local env_name="$1"
    local python="$2"
    mkdir -p "$MANIFEST_ROOT"
    "$UV_BIN" pip freeze --python "$python" > "${MANIFEST_ROOT}/${env_name}-${MODE}.txt"
}

install_shell_helper() {
    local helper_dir="${HOME}/.config/nmr"
    local helper_file="${helper_dir}/envs.sh"
    local bashrc="${HOME}/.bashrc"
    local source_line="[ -f \"${helper_file}\" ] && source \"${helper_file}\""
    local helper_tmp

    mkdir -p "$helper_dir"
    helper_tmp="$(mktemp "${helper_dir}/envs.sh.tmp.XXXXXX")"

    {
        printf '# Generated by %s. Re-run the setup script to update.\n' "$0"
        printf 'export NMR_VENV_ROOT=%q\n\n' "$VENV_ROOT"
        cat <<'EOF'
nmr-env() {
    local env_dir

    case "${1:-}" in
        main)     env_dir="${NMR_VENV_ROOT}/nmr_venv" ;;
        ultranmr) env_dir="${NMR_VENV_ROOT}/ultranmr_venv" ;;
        nmrtrans) env_dir="${NMR_VENV_ROOT}/nmrtrans_venv" ;;
        nmrpeak)  env_dir="${NMR_VENV_ROOT}/nmrpeak_venv" ;;
        unimol2)  env_dir="${NMR_VENV_ROOT}/unimol2_venv" ;;
        off)
            if declare -F deactivate >/dev/null 2>&1; then
                deactivate
            fi
            return 0
            ;;
        list)
            printf 'main      %s\n' "${NMR_VENV_ROOT}/nmr_venv"
            printf 'ultranmr  %s\n' "${NMR_VENV_ROOT}/ultranmr_venv"
            printf 'nmrtrans  %s\n' "${NMR_VENV_ROOT}/nmrtrans_venv"
            printf 'nmrpeak   %s\n' "${NMR_VENV_ROOT}/nmrpeak_venv"
            printf 'unimol2   %s\n' "${NMR_VENV_ROOT}/unimol2_venv"
            return 0
            ;;
        *)
            printf 'Usage: nmr-env {main|ultranmr|nmrtrans|nmrpeak|unimol2|off|list}\n' >&2
            return 2
            ;;
    esac

    if [[ ! -f "${env_dir}/bin/activate" ]]; then
        printf 'Environment not found: %s\n' "$env_dir" >&2
        return 1
    fi

    if declare -F deactivate >/dev/null 2>&1; then
        deactivate
    fi
    source "${env_dir}/bin/activate"
}
EOF
    } > "$helper_tmp"

    chmod 0644 "$helper_tmp"
    mv -f -- "$helper_tmp" "$helper_file"
    touch "$bashrc"

    if ! grep -Fqx "$source_line" "$bashrc"; then
        {
            printf '\n# >>> nmr-env helper >>>\n'
            printf '%s\n' "$source_line"
            printf '# <<< nmr-env helper <<<\n'
        } >> "$bashrc"
    fi

    note "Installed shell helper: ${helper_file}"
}

install_main() {
    ensure_venv "$MAIN_ENV" "3.11"
    local python="${MAIN_ENV}/bin/python"
    install_torch "$python" "torch==${MAIN_TORCH_VERSION}"
    "$UV_BIN" pip install \
        --python "$python" \
        -r "${SCRIPT_DIR}/requirements/nmr-main.txt"
    "$python" -c 'import lz4, numpy, pandas, pyarrow, rdkit, sklearn, torch'
    verify_torch "$python" "nmr_venv"
    record_manifest "nmr_venv" "$python"
}

install_ultranmr() {
    ensure_repo "$ULTRANMR_REPO" "setup.py"
    ensure_venv "$ULTRANMR_ENV" "3.11"
    local python="${ULTRANMR_ENV}/bin/python"
    install_build_tools "$python"
    "$UV_BIN" pip install \
        --python "$python" \
        --torch-backend "$TORCH_BACKEND" \
        --editable "$ULTRANMR_REPO"
    install_parquet_support "$python"
    "$python" -c 'import pyarrow, rdkit, torch, transformers'
    verify_torch "$python" "ultranmr_venv"
    record_manifest "ultranmr_venv" "$python"
}

install_nmrtrans() {
    ensure_repo "$NMRTRANS_REPO" "uv.lock"
    ensure_venv "$NMRTRANS_ENV" "3.10"
    local python="${NMRTRANS_ENV}/bin/python"
    local locked_requirements
    locked_requirements="$(mktemp)"

    "$UV_BIN" export \
        --project "$NMRTRANS_REPO" \
        --locked \
        --no-dev \
        --prune torch \
        --no-emit-project \
        --no-hashes \
        --output-file "$locked_requirements"

    install_torch "$python" "torch==2.6.0"
    "$UV_BIN" pip install --python "$python" -r "$locked_requirements"
    "$UV_BIN" pip install --python "$python" --no-deps --editable "$NMRTRANS_REPO"
    install_parquet_support "$python"
    rm -f -- "$locked_requirements"

    "$python" -c 'import lightning, pyarrow, torch, transformers'
    verify_torch "$python" "nmrtrans_venv"
    record_manifest "nmrtrans_venv" "$python"
}

install_unicore() {
    local python="$1"
    local source_dir="${SOURCE_ROOT}/Uni-Core-${UNICORE_REF:0:12}"

    mkdir -p "$SOURCE_ROOT"
    if [[ ! -d "${source_dir}/.git" ]]; then
        note "Cloning pinned Uni-Core source"
        git clone --filter=blob:none https://github.com/dptech-corp/Uni-Core.git "$source_dir"
        git -C "$source_dir" checkout --detach "$UNICORE_REF"
    fi

    local current_ref
    current_ref="$(git -C "$source_dir" rev-parse HEAD)"
    [[ "$current_ref" == "$UNICORE_REF" ]] ||
        die "Uni-Core source at ${source_dir} is not at the expected commit."

    # The pinned Uni-Core revision disables optional CUDA extensions by default.
    # PyTorch can still use the GPU; enabling fused extensions additionally
    # requires a matching local CUDA toolkit and nvcc.
    "$UV_BIN" pip install \
        --python "$python" \
        --no-build-isolation \
        "$source_dir"
}

install_nmrpeak() {
    ensure_repo "$NMRPEAK_REPO" "requirements.txt"
    ensure_venv "$NMRPEAK_ENV" "3.10"
    local python="${NMRPEAK_ENV}/bin/python"
    install_build_tools "$python"
    install_torch \
        "$python" \
        "torch==2.3.0" \
        "torchvision==0.18.0" \
        "torchaudio==2.3.0"

    "$UV_BIN" pip install \
        --python "$python" \
        --torch-backend "$TORCH_BACKEND" \
        --requirements "${NMRPEAK_REPO}/requirements.txt" \
        --excludes "${SCRIPT_DIR}/requirements/nmrpeak-excludes.txt"

    install_unicore "$python"
    install_parquet_support "$python"

    "$python" -c 'import pathlib, site, sys; pathlib.Path(site.getsitepackages()[0], "nmrpeak_repo.pth").write_text(sys.argv[1] + "\n")' "$NMRPEAK_REPO"
    "$python" -c 'import faiss, pyarrow, rdkit, torch, transformers, unicore'
    verify_torch "$python" "nmrpeak_venv"
    record_manifest "nmrpeak_venv" "$python"
}

install_unimol2() {
    ensure_venv "$UNIMOL2_ENV" "3.11"
    local python="${UNIMOL2_ENV}/bin/python"
    install_torch "$python" "torch==2.6.0"
    "$UV_BIN" pip install \
        --python "$python" \
        --torch-backend "$TORCH_BACKEND" \
        "unimol_tools==0.1.6"
    install_parquet_support "$python"
    "$python" -c 'import importlib.metadata, pyarrow, rdkit, torch, unimol_tools; assert importlib.metadata.version("unimol_tools") == "0.1.6"'
    verify_torch "$python" "unimol2_venv"
    record_manifest "unimol2_venv" "$python"
}

note "Project root: ${PROJECT_ROOT}"
note "Environment root: ${VENV_ROOT}"
note "PyTorch backend: ${TORCH_BACKEND}"

if [[ "$ONLY" == "all" || "$ONLY" == "main" ]]; then
    install_main
fi
if [[ "$ONLY" == "all" || "$ONLY" == "ultranmr" ]]; then
    install_ultranmr
fi
if [[ "$ONLY" == "all" || "$ONLY" == "nmrtrans" ]]; then
    install_nmrtrans
fi
if [[ "$ONLY" == "all" || "$ONLY" == "nmrpeak" ]]; then
    install_nmrpeak
fi

if [[ "$ONLY" == "all" || "$ONLY" == "unimol2" ]]; then
    install_unimol2
fi
if [[ "$INSTALL_SHELL_HELPER" == "1" ]]; then
    install_shell_helper
fi

note "Setup completed."
if [[ "$INSTALL_SHELL_HELPER" == "1" ]]; then
    printf '\nOpen a new shell or run: source %q\n' "${HOME}/.bashrc"
    printf 'Then activate with: nmr-env {main|ultranmr|nmrtrans|nmrpeak|unimol2}\n'
else
    printf '\nActivation commands:\n'
    printf '  source %q\n' "${MAIN_ENV}/bin/activate"
    printf '  source %q\n' "${ULTRANMR_ENV}/bin/activate"
    printf '  source %q\n' "${NMRTRANS_ENV}/bin/activate"
    printf '  source %q\n' "${NMRPEAK_ENV}/bin/activate"
    printf '  source %q\n' "${UNIMOL2_ENV}/bin/activate"
fi
