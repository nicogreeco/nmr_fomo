#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="$REPO_ROOT/datasets/raw"

NMRGYM_REPO="meaw0415/NMRGym"
NMRGYM_REVISION="b1172058c1961d10d3005aec3eec93e5b0e4342e"
NMRTRANS_REPO="little1d/NMRTrans-Data"
NMRTRANS_REVISION="658e28c074b340435b7edc4c5f6a7b76974dcba8"
SIMNMR_REPO="yqj01/SimNMR-PubChem"
SIMNMR_REVISION="d915caf02834759858afccfccbb7cff1a377bbc7"

ZENODO_URL="https://zenodo.org/records/19122815/files/data.zip?download=1"
ZENODO_MD5="b27be622059908c9f07b6e9ba3ff641f"
ADMET_URL="https://dataverse.harvard.edu/api/access/datafile/4426004"
ADMET_FALLBACK_URL="https://zenodo.org/api/records/20180944/files/tdc_admet_group_2026-03-24.tar.gz/content"
ADMET_FALLBACK_SHA256="8ac217bd8c316d04d15ab2ef5173ef6e9a084e156dfcd16c0ac21e2ed6e4590b"
SANGSTER_URL="https://zenodo.org/records/19387552/files/Datasets.xlsx?download=1"
SANGSTER_SHA256="a5f998a690ff48d820afe11b23d2522e95ff11764f6d13c773749750f3ea8d42"

STEP_INDEX=0
STEP_TOTAL=0

show_help() {
    cat <<'EOF'
Download the public raw releases into the paths expected by dvc.yaml.

Usage:
  scripts/data/download_raw_datasets.sh [--no-verify] TARGET [TARGET ...]

Targets:
  admet sangster_logp mst_nmr nmrexp nmrgym nmrtrans simnmr_pubchem all

Examples:
  scripts/data/download_raw_datasets.sh nmrtrans nmrgym
  scripts/data/download_raw_datasets.sh all

By default, the selected datasets are checked against their committed .dvc
pointers. Use --no-verify to skip this potentially long hash pass. Existing
completed targets are never overwritten. An interrupted SimNMR download can be
resumed by running the same command again.
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_new_target() {
    local target="$1"
    [[ ! -e "$target" ]] || die "$target already exists; refusing to overwrite it"
}

check_size() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(stat -c %s "$path")"
    [[ "$actual" == "$expected" ]] || die "$path has $actual bytes; expected $expected"
}

announce_step() {
    local count="$1"
    local message="$2"
    local first=$((STEP_INDEX + 1))

    STEP_INDEX=$((STEP_INDEX + count))
    if (( count == 1 )); then
        printf '\n[%d/%d] %s\n' "$first" "$STEP_TOTAL" "$message"
    else
        printf '\n[%d-%d/%d] %s\n' "$first" "$STEP_INDEX" "$STEP_TOTAL" "$message"
    fi
}

download_hf_files() {
    local repo="$1"
    local revision="$2"
    local target="$3"
    shift 3

    require_new_target "$target"
    local staging
    staging="$(mktemp -d)"
    trap 'rm -rf -- "$staging"' RETURN

    printf 'Retrieving %s (%d files)...\n' "$(basename "$target")" "$#"
    local filename
    for filename in "$@"; do
        hf download "$repo" "$filename" \
            --repo-type dataset \
            --revision "$revision" \
            --local-dir "$staging" \
            --quiet >/dev/null
    done

    mkdir -p "$(dirname "$target")"
    mkdir -p "$target"
    for filename in "$@"; do
        mv "$staging/$filename" "$target/$(basename "$filename")"
    done
    printf 'Completed %s.\n' "$(basename "$target")"

    rm -rf -- "$staging"
    trap - RETURN
}

download_zenodo_sources() {
    local want_mst="$1"
    local want_nmrexp="$2"
    local archive staging

    [[ "$want_mst" == 0 ]] || require_new_target "$RAW_ROOT/mst_nmr"
    [[ "$want_nmrexp" == 0 ]] || require_new_target "$RAW_ROOT/nmrexp"

    staging="$(mktemp -d)"
    archive="$staging/data.zip"
    trap 'rm -rf -- "$staging"' RETURN

    printf 'Retrieving the shared NMRPeak Zenodo archive...\n'
    curl --fail --location --retry 3 --output "$archive" "$ZENODO_URL"
    printf '%s  %s\n' "$ZENODO_MD5" "$archive" | md5sum --check --status \
        || die "the Zenodo archive checksum does not match release 19122815"

    if [[ "$want_mst" == 1 ]]; then
        mkdir -p "$staging/mst_nmr"
        unzip -p "$archive" data/MST_NMR/lmdb_dataset/train.lmdb > "$staging/mst_nmr/train.lmdb"
        unzip -p "$archive" data/MST_NMR/lmdb_dataset/valid.lmdb > "$staging/mst_nmr/valid.lmdb"
        unzip -p "$archive" data/MST_NMR/lmdb_dataset/test.lmdb > "$staging/mst_nmr/test.lmdb"
        mv "$staging/mst_nmr" "$RAW_ROOT/mst_nmr"
        printf 'Completed mst_nmr.\n'
    fi

    if [[ "$want_nmrexp" == 1 ]]; then
        mkdir -p "$staging/nmrexp"
        unzip -p "$archive" data/NMRexp/lmdb_dataset/train.lmdb > "$staging/nmrexp/train.lmdb"
        unzip -p "$archive" data/NMRexp/lmdb_dataset/valid.lmdb > "$staging/nmrexp/valid.lmdb"
        unzip -p "$archive" data/NMRexp/lmdb_dataset/test.lmdb > "$staging/nmrexp/test.lmdb"
        mv "$staging/nmrexp" "$RAW_ROOT/nmrexp"
        printf 'Completed nmrexp.\n'
    fi

    rm -rf -- "$staging"
    trap - RETURN
}

download_admet() {
    local target="$RAW_ROOT/admet"
    require_new_target "$target"

    local archive archive_format staging extracted source_dir csv_count
    staging="$(mktemp -d)"
    archive="$staging/admet_group.zip"
    archive_format="zip"
    extracted="$staging/extracted"
    trap 'rm -rf -- "$staging"' RETURN

    printf 'Retrieving the TDC ADMET benchmark group from the official PyTDC endpoint...\n'
    local http_status
    http_status="$(curl --fail --location --retry 3 --progress-bar \
        --output "$archive" --write-out '%{http_code}' "$ADMET_URL")"
    if [[ "$http_status" == 202 ]]; then
        printf 'Dataverse returned an AWS WAF challenge; using the DVC-matching public snapshot.\n'
        archive="$staging/admet_group.tar.gz"
        archive_format="tar.gz"
        curl --fail --location --retry 3 --progress-bar \
            --output "$archive" "$ADMET_FALLBACK_URL"
        printf '%s  %s\n' "$ADMET_FALLBACK_SHA256" "$archive" | sha256sum --check --status \
            || die "the ADMET fallback archive checksum does not match"
    else
        printf 'Received HTTP %s; validating archive...\n' "$http_status"
        if [[ ! -s "$archive" ]] || ! unzip -tq "$archive" >/dev/null 2>&1; then
            die "Harvard Dataverse did not return a valid ADMET ZIP"
        fi
    fi

    mkdir -p "$extracted"
    if [[ "$archive_format" == "zip" ]]; then
        unzip -q "$archive" -d "$extracted"
    else
        tar --extract --gzip --file "$archive" --directory "$extracted"
    fi
    source_dir="$(find "$extracted" -type d -name admet_group -print -quit)"
    [[ -n "$source_dir" ]] || die "the ADMET archive does not contain an admet_group directory"
    rm -f -- "$source_dir/.DS_Store"
    csv_count="$(find "$source_dir" -type f -name '*.csv' | wc -l)"
    [[ "$csv_count" == 44 ]] || die "the ADMET archive contains $csv_count CSV files; expected 44"

    mv "$source_dir" "$target"
    printf 'Completed admet.\n'

    rm -rf -- "$staging"
    trap - RETURN
}

download_sangster_logp() {
    local target="$RAW_ROOT/sangster_logp"
    require_new_target "$target"

    local staging workbook
    staging="$(mktemp -d)"
    workbook="$staging/Datasets.xlsx"
    trap 'rm -rf -- "$staging"' RETURN

    printf 'Retrieving the Sangster logP workbook from Zenodo...\n'
    curl --fail --location --retry 3 --progress-bar \
        --output "$workbook" "$SANGSTER_URL"
    printf '%s  %s\n' "$SANGSTER_SHA256" "$workbook" | sha256sum --check --status \
        || die "the Sangster workbook checksum does not match release 19387552"
    mkdir -p "$target"
    mv "$workbook" "$target/Datasets.xlsx"
    printf 'Completed sangster_logp.\n'

    rm -rf -- "$staging"
    trap - RETURN
}

download_simnmr() {
    local target_dir="$RAW_ROOT/simnmr_pubchem"
    local final_file="$target_dir/metadata/PubChem_merged_id.lmdb"
    local partial_file="$final_file.partial"
    local state_file="$target_dir/.download_state"
    local completed_parts=0

    if [[ -f "$final_file" ]]; then
        check_size "$final_file" 400000000000
        printf 'simnmr_pubchem is already complete; keeping it unchanged.\n'
        return
    fi

    if [[ -e "$target_dir" && ! -f "$partial_file" ]]; then
        die "$target_dir exists but has no resumable partial LMDB"
    fi
    mkdir -p "$target_dir/metadata"

    if [[ -f "$partial_file" ]]; then
        [[ -f "$state_file" ]] || die "$partial_file exists without $state_file"
        read -r completed_parts recorded_size < "$state_file"
        check_size "$partial_file" "$recorded_size"
        printf 'Resuming SimNMR after part %s.\n' "$completed_parts"
    fi

    local part_number part_name staging downloaded cumulative_size
    for part_number in $(seq 1 38); do
        (( part_number <= completed_parts )) && continue
        part_name="metadata/PubChem_merged_id.lmdb.part-$(printf '%03d' "$part_number")"
        staging="$(mktemp -d)"
        printf 'Downloading SimNMR part %d/38...\n' "$part_number"
        hf download "$SIMNMR_REPO" "$part_name" \
            --repo-type dataset \
            --revision "$SIMNMR_REVISION" \
            --local-dir "$staging" \
            --quiet >/dev/null
        downloaded="$staging/$part_name"
        dd if="$downloaded" of="$partial_file" bs=64M oflag=append conv=notrunc status=none
        cumulative_size="$(stat -c %s "$partial_file")"
        printf '%d %s\n' "$part_number" "$cumulative_size" > "$state_file"
        rm -rf -- "$staging"
    done

    check_size "$partial_file" 400000000000
    mv "$partial_file" "$final_file"
    rm -f -- "$state_file"
    printf 'Completed simnmr_pubchem.\n'
}

verify_downloads() {
    local backup_dir pointer source mismatch=0
    backup_dir="$(mktemp -d)"

    for pointer in "$@"; do
        cp "$pointer" "$backup_dir/$(basename "$pointer")"
        source="${pointer%.dvc}"
        dvc add --no-commit "$source" --quiet
        if ! cmp -s "$pointer" "$backup_dir/$(basename "$pointer")"; then
            printf 'DVC hash mismatch: %s\n' "$source" >&2
            mismatch=1
        fi
        cp "$backup_dir/$(basename "$pointer")" "$pointer"
    done

    rm -rf -- "$backup_dir"
    (( mismatch == 0 )) || die "downloaded files do not match the committed DVC pointers"
    printf 'DVC verification completed successfully.\n'
}

main() {
    local verify=1
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        show_help
        return
    fi
    if [[ "${1:-}" == "--no-verify" ]]; then
        verify=0
        shift
    fi
    (( $# > 0 )) || { show_help >&2; exit 2; }

    mkdir -p "$RAW_ROOT"
    cd "$REPO_ROOT"

    declare -A selected=()
    local target
    for target in "$@"; do
        if [[ "$target" == all ]]; then
            selected[admet]=1
            selected[sangster_logp]=1
            selected[mst_nmr]=1
            selected[nmrexp]=1
            selected[nmrgym]=1
            selected[nmrtrans]=1
            selected[simnmr_pubchem]=1
        else
            case "$target" in
                admet|sangster_logp|mst_nmr|nmrexp|nmrgym|nmrtrans|simnmr_pubchem)
                    selected["$target"]=1
                    ;;
                *)
                    die "unknown target: $target"
                    ;;
            esac
        fi
    done

    local want_mst="${selected[mst_nmr]:-0}"
    local want_nmrexp="${selected[nmrexp]:-0}"
    local selected_names=()
    for target in admet sangster_logp mst_nmr nmrexp nmrgym nmrtrans simnmr_pubchem; do
        [[ "${selected[$target]:-0}" == 0 ]] || selected_names+=("$target")
    done
    STEP_TOTAL="${#selected_names[@]}"
    printf 'Selected %d dataset(s): %s\n' "$STEP_TOTAL" "${selected_names[*]}"

    if [[ "$want_mst" == 1 || "$want_nmrexp" == 1 || "${selected[admet]:-0}" == 1 || "${selected[sangster_logp]:-0}" == 1 ]]; then
        require_command curl
    fi
    if [[ "$want_mst" == 1 || "$want_nmrexp" == 1 || "${selected[admet]:-0}" == 1 ]]; then
        require_command unzip
    fi
    if [[ "${selected[admet]:-0}" == 1 ]]; then
        require_command sha256sum
        require_command tar
    fi
    if [[ "${selected[sangster_logp]:-0}" == 1 ]]; then
        require_command sha256sum
    fi
    if [[ "$want_mst" == 1 || "$want_nmrexp" == 1 ]]; then
        require_command md5sum
    fi
    if [[ "${selected[nmrgym]:-0}" == 1 || "${selected[nmrtrans]:-0}" == 1 || "${selected[simnmr_pubchem]:-0}" == 1 ]]; then
        require_command hf
    fi
    if [[ "$verify" == 1 ]]; then
        require_command dvc
    fi

    if [[ "$want_mst" == 1 || "$want_nmrexp" == 1 ]]; then
        if [[ "$want_mst" == 1 && "$want_nmrexp" == 1 ]]; then
            announce_step 2 "Downloading mst_nmr and nmrexp from one shared release"
        elif [[ "$want_mst" == 1 ]]; then
            announce_step 1 "Downloading mst_nmr"
        else
            announce_step 1 "Downloading nmrexp"
        fi
        download_zenodo_sources "$want_mst" "$want_nmrexp"
    fi
    if [[ "${selected[admet]:-0}" == 1 ]]; then
        announce_step 1 "Downloading admet"
        download_admet
    fi
    if [[ "${selected[sangster_logp]:-0}" == 1 ]]; then
        announce_step 1 "Downloading sangster_logp"
        download_sangster_logp
    fi
    if [[ "${selected[nmrgym]:-0}" == 1 ]]; then
        announce_step 1 "Downloading nmrgym"
        download_hf_files \
            "$NMRGYM_REPO" "$NMRGYM_REVISION" "$RAW_ROOT/nmrgym" \
            NMRGym_train_balanced_dedup.pkl \
            NMRGym_val_balanced_dedup.pkl \
            NMRGym_test_balanced_dedup.pkl
    fi
    if [[ "${selected[nmrtrans]:-0}" == 1 ]]; then
        announce_step 1 "Downloading nmrtrans"
        download_hf_files \
            "$NMRTRANS_REPO" "$NMRTRANS_REVISION" "$RAW_ROOT/nmrtrans" \
            train.pkl.lz4 val.pkl.lz4 test.pkl.lz4
    fi
    if [[ "${selected[simnmr_pubchem]:-0}" == 1 ]]; then
        announce_step 1 "Downloading simnmr_pubchem"
        download_simnmr
    fi

    printf '\nCompleted %d selected dataset download(s).\n' "$STEP_TOTAL"

    if [[ "$verify" == 1 ]]; then
        printf 'Verifying selected datasets against DVC pointers...\n'
        local pointers=()
        for target in admet sangster_logp mst_nmr nmrexp nmrgym nmrtrans simnmr_pubchem; do
            [[ "${selected[$target]:-0}" == 0 ]] || pointers+=("datasets/raw/$target.dvc")
        done
        verify_downloads "${pointers[@]}"
    else
        printf 'Skipped DVC verification (--no-verify).\n'
    fi
}

main "$@"
