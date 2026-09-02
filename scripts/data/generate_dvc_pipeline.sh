#!/usr/bin/env bash
# Generate the explicit root DVC data-processing pipeline without running it.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

for required_path in .dvc params.yaml datasets/raw scripts/data; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done
if ! command -v dvc >/dev/null 2>&1; then
    echo "DVC is not available. Activate the main project environment first." >&2
    exit 1
fi
if ! command -v python >/dev/null 2>&1; then
    echo "Python is not available. Activate the main project environment first." >&2
    exit 1
fi

canonical_shared_deps=(
    scripts/data/__init__.py
    scripts/data/console.py
    scripts/data/canonicalize/common.py
    scripts/data/schema.py
    scripts/data/validation.py
    scripts/data/reporting.py
    scripts/envs_scr/requirements/nmr-main.txt
)
postprocess_shared_deps=(
    scripts/data/__init__.py
    scripts/data/console.py
    scripts/data/postprocess/common.py
    scripts/data/canonicalize/common.py
    scripts/data/schema.py
    scripts/data/validation.py
    scripts/data/reporting.py
    scripts/envs_scr/requirements/nmr-main.txt
)

canonical_dep_args=()
for dependency in "${canonical_shared_deps[@]}"; do
    canonical_dep_args+=(-d "${dependency}")
done
postprocess_dep_args=()
for dependency in "${postprocess_shared_deps[@]}"; do
    postprocess_dep_args+=(-d "${dependency}")
done

# These outputs remain cached locally and recorded in dvc.lock, but a normal
# dvc push does not upload them. Final cleaned data and all small reports use
# the default push policy.
local_only_outputs=()

add_canonical_stage() {
    local stage_name="$1"
    local converter="$2"
    local input_path="$3"
    local output_path="$4"
    local report_path="${output_path%.parquet}_report.json"
    local command

    command="PYTHONPATH=scripts python ${converter} ${input_path} ${output_path}"
    command+=" --row-group-size \${canonicalize.row_group_size}"
    command+=" --report-output ${report_path}"
    command+=" --quiet"

    dvc stage add --force --name "${stage_name}" \
        --deps "${input_path}" \
        --deps "${converter}" \
        "${canonical_dep_args[@]}" \
        --params canonicalize.row_group_size \
        --outs "${output_path}" \
        --outs "${report_path}" \
        "${command}"

    local_only_outputs+=("${output_path}")
}

add_canonical_stage \
    canonicalize_mst_train \
    scripts/data/canonicalize/convert_mst_nmr.py \
    datasets/raw/mst_nmr/train.lmdb \
    datasets/canonical/mst_nmr/train.parquet
add_canonical_stage \
    canonicalize_mst_val \
    scripts/data/canonicalize/convert_mst_nmr.py \
    datasets/raw/mst_nmr/valid.lmdb \
    datasets/canonical/mst_nmr/val.parquet
add_canonical_stage \
    canonicalize_mst_test \
    scripts/data/canonicalize/convert_mst_nmr.py \
    datasets/raw/mst_nmr/test.lmdb \
    datasets/canonical/mst_nmr/test.parquet

add_canonical_stage \
    canonicalize_nmrexp_train \
    scripts/data/canonicalize/convert_nmrexp.py \
    datasets/raw/nmrexp/train.lmdb \
    datasets/canonical/nmrexp/train.parquet
add_canonical_stage \
    canonicalize_nmrexp_val \
    scripts/data/canonicalize/convert_nmrexp.py \
    datasets/raw/nmrexp/valid.lmdb \
    datasets/canonical/nmrexp/val.parquet
add_canonical_stage \
    canonicalize_nmrexp_test \
    scripts/data/canonicalize/convert_nmrexp.py \
    datasets/raw/nmrexp/test.lmdb \
    datasets/canonical/nmrexp/test.parquet

add_canonical_stage \
    canonicalize_nmrtrans_train \
    scripts/data/canonicalize/convert_nmrtrans.py \
    datasets/raw/nmrtrans/train.pkl.lz4 \
    datasets/canonical/nmrtrans/train.parquet
add_canonical_stage \
    canonicalize_nmrtrans_val \
    scripts/data/canonicalize/convert_nmrtrans.py \
    datasets/raw/nmrtrans/val.pkl.lz4 \
    datasets/canonical/nmrtrans/val.parquet
add_canonical_stage \
    canonicalize_nmrtrans_test \
    scripts/data/canonicalize/convert_nmrtrans.py \
    datasets/raw/nmrtrans/test.pkl.lz4 \
    datasets/canonical/nmrtrans/test.parquet

add_canonical_stage \
    canonicalize_nmrgym \
    scripts/data/canonicalize/convert_nmrgym.py \
    datasets/raw/nmrgym \
    datasets/canonical/nmrgym/all.parquet

simnmr_output=datasets/canonical/simnmr/all.parquet
simnmr_report=datasets/canonical/simnmr/all_report.json
simnmr_rejections=datasets/canonical/simnmr/all_chemical_metadata_rejections.jsonl
simnmr_command="PYTHONPATH=scripts python"
simnmr_command+=" scripts/data/canonicalize/convert_nmrsolver.py"
simnmr_command+=" datasets/raw/simnmr_pubchem/metadata/PubChem_merged_id.lmdb"
simnmr_command+=" ${simnmr_output}"
simnmr_command+=" --row-group-size \${canonicalize.row_group_size}"
simnmr_command+=" --workers \${canonicalize.simnmr_workers}"
simnmr_command+=" --records-per-task \${canonicalize.simnmr_records_per_task}"
simnmr_command+=" --report-output ${simnmr_report}"
simnmr_command+=" --quiet"
dvc stage add --force --name canonicalize_simnmr \
    --deps datasets/raw/simnmr_pubchem/metadata/PubChem_merged_id.lmdb \
    --deps scripts/data/canonicalize/convert_nmrsolver.py \
    "${canonical_dep_args[@]}" \
    --params canonicalize.row_group_size \
    --params canonicalize.simnmr_workers \
    --params canonicalize.simnmr_records_per_task \
    --outs "${simnmr_output}" \
    --outs "${simnmr_rejections}" \
    --outs "${simnmr_report}" \
    "${simnmr_command}"
local_only_outputs+=("${simnmr_output}" "${simnmr_rejections}")

add_merge_stage() {
    local stage_name="$1"
    local output_path="$2"
    shift 2
    local input_paths=("$@")
    local report_path="${output_path%.parquet}_report.json"
    local stage_args=(
        stage add --force --name "${stage_name}"
        --deps scripts/data/postprocess/merge_datasets.py
        "${postprocess_dep_args[@]}"
        --params postprocess.batch_size
        --outs "${output_path}"
        --outs "${report_path}"
    )
    local command="PYTHONPATH=scripts python"
    command+=" scripts/data/postprocess/merge_datasets.py"

    for input_path in "${input_paths[@]}"; do
        stage_args+=(--deps "${input_path}")
        command+=" ${input_path}"
    done
    command+=" --output ${output_path}"
    command+=" --batch-size \${postprocess.batch_size}"
    command+=" --report-output ${report_path}"
    command+=" --quiet"

    dvc "${stage_args[@]}" "${command}"
    local_only_outputs+=("${output_path}")
}

add_merge_stage \
    merge_rich_train_val \
    datasets/intermediate/rich_train_val.parquet \
    datasets/canonical/mst_nmr/train.parquet \
    datasets/canonical/mst_nmr/val.parquet \
    datasets/canonical/nmrexp/train.parquet \
    datasets/canonical/nmrexp/val.parquet \
    datasets/canonical/nmrtrans/train.parquet \
    datasets/canonical/nmrtrans/val.parquet

add_merge_stage \
    merge_rich_test \
    datasets/intermediate/rich_test.parquet \
    datasets/canonical/mst_nmr/test.parquet \
    datasets/canonical/nmrexp/test.parquet \
    datasets/canonical/nmrtrans/test.parquet

move_test_output=datasets/intermediate/test_after_simnmr.parquet
move_train_output=datasets/intermediate/train_val_extended.parquet
move_report=datasets/intermediate/move_simnmr_overlaps_to_train_report.json
move_command="PYTHONPATH=scripts python"
move_command+=" scripts/data/postprocess/move_benchmark_overlaps_to_train.py"
move_command+=" datasets/intermediate/rich_test.parquet"
move_command+=" datasets/canonical/simnmr/all.parquet"
move_command+=" datasets/intermediate/rich_train_val.parquet"
move_command+=" --test-output ${move_test_output}"
move_command+=" --train-output ${move_train_output}"
move_command+=" --batch-size \${postprocess.batch_size}"
move_command+=" --report-output ${move_report}"
move_command+=" --quiet"
dvc stage add --force --name move_simnmr_overlaps_to_train \
    --deps datasets/intermediate/rich_test.parquet \
    --deps datasets/canonical/simnmr/all.parquet \
    --deps datasets/intermediate/rich_train_val.parquet \
    --deps scripts/data/postprocess/move_benchmark_overlaps_to_train.py \
    "${postprocess_dep_args[@]}" \
    --params postprocess.batch_size \
    --outs "${move_test_output}" \
    --outs "${move_train_output}" \
    --outs "${move_report}" \
    "${move_command}"
local_only_outputs+=("${move_test_output}" "${move_train_output}")

disjoint_output=datasets/intermediate/test_benchmark_preclean.parquet
disjoint_report=datasets/intermediate/remove_benchmark_overlaps_report.json
disjoint_command="PYTHONPATH=scripts python"
disjoint_command+=" scripts/data/postprocess/remove_benchmark_overlaps.py"
disjoint_command+=" datasets/intermediate/test_after_simnmr.parquet"
disjoint_command+=" datasets/intermediate/train_val_extended.parquet"
disjoint_command+=" datasets/canonical/nmrgym/all.parquet"
disjoint_command+=" --output ${disjoint_output}"
disjoint_command+=" --batch-size \${postprocess.batch_size}"
disjoint_command+=" --report-output ${disjoint_report}"
disjoint_command+=" --quiet"
dvc stage add --force --name remove_benchmark_overlaps \
    --deps datasets/intermediate/test_after_simnmr.parquet \
    --deps datasets/intermediate/train_val_extended.parquet \
    --deps datasets/canonical/nmrgym/all.parquet \
    --deps scripts/data/postprocess/remove_benchmark_overlaps.py \
    "${postprocess_dep_args[@]}" \
    --params postprocess.batch_size \
    --outs "${disjoint_output}" \
    --outs "${disjoint_report}" \
    "${disjoint_command}"
local_only_outputs+=("${disjoint_output}")

add_filter_stage() {
    local stage_name="$1"
    local input_path="$2"
    local output_path="$3"
    local removed_path="$4"
    local final_output="$5"
    local report_path="${output_path%.parquet}_report.json"
    local command

    command="PYTHONPATH=scripts python"
    command+=" scripts/data/postprocess/filter_dataset.py"
    command+=" ${input_path}"
    command+=" --output ${output_path}"
    command+=" --removed-output ${removed_path}"
    command+=" --batch-size \${postprocess.batch_size}"
    command+=" --report-output ${report_path}"
    command+=" --quiet"

    dvc stage add --force --name "${stage_name}" \
        --deps "${input_path}" \
        --deps scripts/data/postprocess/filter_dataset.py \
        "${postprocess_dep_args[@]}" \
        --params postprocess.batch_size \
        --outs "${output_path}" \
        --outs "${removed_path}" \
        --outs "${report_path}" \
        "${command}"

    if [[ "${final_output}" != true ]]; then
        local_only_outputs+=("${output_path}")
    fi
    local_only_outputs+=("${removed_path}")
}

add_filter_stage \
    filter_train \
    datasets/intermediate/train_val_extended.parquet \
    datasets/intermediate/train_val_filtered.parquet \
    datasets/intermediate/audits/train_val_removed.parquet \
    false
add_filter_stage \
    filter_test \
    datasets/intermediate/test_benchmark_preclean.parquet \
    datasets/cleaned/test_benchmark.parquet \
    datasets/intermediate/audits/test_benchmark_removed.parquet \
    true
add_filter_stage \
    filter_simnmr \
    datasets/canonical/simnmr/all.parquet \
    datasets/cleaned/simnmr.parquet \
    datasets/intermediate/audits/simnmr_removed.parquet \
    true
add_filter_stage \
    filter_nmrgym \
    datasets/canonical/nmrgym/all.parquet \
    datasets/cleaned/nmrgym.parquet \
    datasets/intermediate/audits/nmrgym_removed.parquet \
    true

admet_deps=()
for endpoint in solubility_aqsoldb ld50_zhu ames lipophilicity_astrazeneca; do
    for split in train_val test; do
        admet_deps+=("datasets/raw/admet/${endpoint}/${split}.csv")
    done
done
admet_outputs=(datasets/cleaned/rich.parquet)
for endpoint in solubility_aqsoldb ld50_zhu ames lipophilicity_astrazeneca sangster_logp; do
    for split in train_val test; do
        admet_outputs+=("datasets/cleaned/admet/${endpoint}/${split}.parquet")
        admet_outputs+=("datasets/cleaned/admet/${endpoint}/${split}.csv")
    done
done
admet_report=datasets/cleaned/admet/preparation_report.json
admet_outputs+=("${admet_report}")
prepare_admet_args=(
    stage add --force --name prepare_admet
    --deps datasets/intermediate/train_val_filtered.parquet
    --deps datasets/raw/sangster_logp/Datasets.xlsx
    --deps scripts/data/postprocess/prepare_admet_datasets.py
    "${postprocess_dep_args[@]}"
    --params postprocess.batch_size
    --params admet.sangster_train_fraction
    --params admet.sangster_seed
)
for dependency in "${admet_deps[@]}"; do
    prepare_admet_args+=(--deps "${dependency}")
done
for output_path in "${admet_outputs[@]}"; do
    prepare_admet_args+=(--outs "${output_path}")
done
prepare_admet_command="PYTHONPATH=scripts python"
prepare_admet_command+=" scripts/data/postprocess/prepare_admet_datasets.py"
prepare_admet_command+=" datasets/intermediate/train_val_filtered.parquet"
prepare_admet_command+=" --tdc-root datasets/raw/admet"
prepare_admet_command+=" --sangster-workbook datasets/raw/sangster_logp/Datasets.xlsx"
prepare_admet_command+=" --output-root datasets/cleaned/admet"
prepare_admet_command+=" --train-output datasets/cleaned/rich.parquet"
prepare_admet_command+=" --sangster-train-fraction \${admet.sangster_train_fraction}"
prepare_admet_command+=" --sangster-seed \${admet.sangster_seed}"
prepare_admet_command+=" --batch-size \${postprocess.batch_size}"
prepare_admet_command+=" --report-output ${admet_report}"
prepare_admet_command+=" --quiet"
dvc "${prepare_admet_args[@]}" "${prepare_admet_command}"

add_molecular_properties_stage() {
    local stage_label="$1"
    local dataset_name="$2"
    local input_path="datasets/cleaned/${dataset_name}.parquet"
    local output_path="datasets/cleaned/${dataset_name}_mol_properties.parquet"
    local report_path="datasets/cleaned/${dataset_name}_mol_properties_report.json"
    local command

    command="PYTHONPATH=scripts python"
    command+=" scripts/data/postprocess/calculate_mol_properties.py"
    command+=" ${input_path}"
    command+=" --output ${output_path}"
    command+=" --records-per-task \${mol_properties.records_per_task}"
    command+=" --workers \${mol_properties.workers}"
    command+=" --report-output ${report_path}"
    command+=" --quiet"

    dvc stage add --force --name "calculate_${stage_label}_properties" \
        --deps "${input_path}" \
        --deps scripts/data/postprocess/calculate_mol_properties.py \
        "${postprocess_dep_args[@]}" \
        --params mol_properties.records_per_task \
        --params mol_properties.workers \
        --outs "${output_path}" \
        --outs "${report_path}" \
        "${command}"
}

add_molecular_properties_stage rich rich
add_molecular_properties_stage test test_benchmark
add_molecular_properties_stage simnmr simnmr
add_molecular_properties_stage nmrgym nmrgym

split_output_root=datasets/train_splits
split_report=${split_output_root}/split_report.json
split_outputs=()
for source_name in simnmr rich nmrgym; do
    split_outputs+=("${split_output_root}/${source_name}_train.parquet")
    split_outputs+=("${split_output_root}/${source_name}_train_mol_properties.parquet")
    split_outputs+=("${split_output_root}/${source_name}_val.parquet")
    split_outputs+=("${split_output_root}/${source_name}_val_mol_properties.parquet")
done
split_args=(
    stage add --force --name split_foundation_datasets
    --deps datasets/cleaned/simnmr.parquet
    --deps datasets/cleaned/simnmr_mol_properties.parquet
    --deps datasets/cleaned/rich.parquet
    --deps datasets/cleaned/rich_mol_properties.parquet
    --deps datasets/cleaned/nmrgym.parquet
    --deps datasets/cleaned/nmrgym_mol_properties.parquet
    --deps scripts/data/postprocess/split_foundation_datasets.py
    "${postprocess_dep_args[@]}"
    --params foundation_splits.seed
    --params foundation_splits.batch_size
    --params foundation_splits.simnmr_validation_records
    --params foundation_splits.rich_validation_records
    --params foundation_splits.nmrgym_validation_records
    --outs "${split_report}"
)
for output_path in "${split_outputs[@]}"; do
    split_args+=(--outs-no-cache "${output_path}")
done
split_command="PYTHONPATH=scripts python"
split_command+=" scripts/data/postprocess/split_foundation_datasets.py"
split_command+=" --cleaned-root datasets/cleaned"
split_command+=" --output-root ${split_output_root}"
split_command+=" --seed \${foundation_splits.seed}"
split_command+=" --batch-size \${foundation_splits.batch_size}"
split_command+=" --simnmr-validation-records \${foundation_splits.simnmr_validation_records}"
split_command+=" --rich-validation-records \${foundation_splits.rich_validation_records}"
split_command+=" --nmrgym-validation-records \${foundation_splits.nmrgym_validation_records}"
split_command+=" --report-output ${split_report}"
split_command+=" --overwrite --quiet"
dvc "${split_args[@]}" "${split_command}"

probe_output_root=datasets/train_splits/maccs_probe
probe_report=${probe_output_root}/split_report.json
probe_outputs=(
    ${probe_output_root}/train.parquet
    ${probe_output_root}/train_mol_properties.parquet
    ${probe_output_root}/eval.parquet
    ${probe_output_root}/eval_mol_properties.parquet
)
probe_args=(
    stage add --force --name split_maccs_probe
    --deps datasets/train_splits/rich_val.parquet
    --deps datasets/train_splits/rich_val_mol_properties.parquet
    --deps scripts/data/postprocess/split_maccs_probe.py
    --deps scripts/data/postprocess/split_foundation_datasets.py
    "${postprocess_dep_args[@]}"
    --params foundation_splits.seed
    --params postprocess.batch_size
    --params maccs_probe.train_records
    --params maccs_probe.eval_records
    --outs "${probe_report}"
)
for output_path in "${probe_outputs[@]}"; do
    probe_args+=(--outs-no-cache "${output_path}")
done
probe_command="PYTHONPATH=scripts python"
probe_command+=" scripts/data/postprocess/split_maccs_probe.py"
probe_command+=" datasets/train_splits/rich_val.parquet"
probe_command+=" datasets/train_splits/rich_val_mol_properties.parquet"
probe_command+=" --output-root ${probe_output_root}"
probe_command+=" --train-records \${maccs_probe.train_records}"
probe_command+=" --eval-records \${maccs_probe.eval_records}"
probe_command+=" --seed \${foundation_splits.seed}"
probe_command+=" --batch-size \${postprocess.batch_size}"
probe_command+=" --report-output ${probe_report}"
probe_command+=" --overwrite --quiet"
dvc "${probe_args[@]}" "${probe_command}"

analytics_deps=(
    datasets/cleaned/rich.parquet
    datasets/cleaned/test_benchmark.parquet
    datasets/cleaned/simnmr.parquet
    datasets/cleaned/nmrgym.parquet
    datasets/cleaned/rich_mol_properties.parquet
    datasets/cleaned/test_benchmark_mol_properties.parquet
    datasets/cleaned/simnmr_mol_properties.parquet
    datasets/cleaned/nmrgym_mol_properties.parquet
)
for endpoint in solubility_aqsoldb ld50_zhu ames lipophilicity_astrazeneca sangster_logp; do
    for split in train_val test; do
        analytics_deps+=("datasets/cleaned/admet/${endpoint}/${split}.parquet")
        analytics_deps+=("datasets/cleaned/admet/${endpoint}/${split}.csv")
    done
done
analytics_dir=datasets/cleaned/analytics
main_analytics_report=datasets/cleaned/analyze_main_report.json
simnmr_analytics_report=datasets/cleaned/analyze_simnmr_report.json
nmrgym_analytics_report=datasets/cleaned/analyze_nmrgym_report.json
analytics_args=(
    stage add --force --name analyze_cleaned_collection
    --deps scripts/data/postprocess/analyze_cleaned_datasets.py
    --deps scripts/data/__init__.py
    --deps scripts/data/console.py
    --deps scripts/data/reporting.py
    --deps scripts/envs_scr/requirements/nmr-main.txt
    --params analytics.sample_per_group
    --outs "${analytics_dir}"
    --outs "${main_analytics_report}"
    --outs "${simnmr_analytics_report}"
    --outs "${nmrgym_analytics_report}"
)
for dependency in "${analytics_deps[@]}"; do
    analytics_args+=(--deps "${dependency}")
done
analytics_command="PYTHONPATH=scripts python"
analytics_command+=" scripts/data/postprocess/analyze_cleaned_datasets.py"
analytics_command+=" --cleaned-root datasets/cleaned"
analytics_command+=" --sample-per-group \${analytics.sample_per_group}"
analytics_command+=" --report-output ${main_analytics_report}"
analytics_command+=" --quiet"
analytics_command+=" && PYTHONPATH=scripts python"
analytics_command+=" scripts/data/postprocess/analyze_cleaned_datasets.py"
analytics_command+=" --nmrsolver-parquet datasets/cleaned/simnmr.parquet"
analytics_command+=" --sample-per-group \${analytics.sample_per_group}"
analytics_command+=" --report-output ${simnmr_analytics_report}"
analytics_command+=" --quiet"
analytics_command+=" && PYTHONPATH=scripts python"
analytics_command+=" scripts/data/postprocess/analyze_cleaned_datasets.py"
analytics_command+=" --nmrgym-parquet datasets/cleaned/nmrgym.parquet"
analytics_command+=" --sample-per-group \${analytics.sample_per_group}"
analytics_command+=" --report-output ${nmrgym_analytics_report}"
analytics_command+=" --quiet"
dvc "${analytics_args[@]}" "${analytics_command}"

# dvc stage add has no flag for the output-level push field. Apply that one
# policy after DVC has generated and validated all explicit stages.
python - dvc.yaml "${local_only_outputs[@]}" <<'PY'
from pathlib import Path
import sys

from ruamel.yaml import YAML

yaml_path = Path(sys.argv[1])
local_paths = set(sys.argv[2:])
yaml = YAML()
yaml.preserve_quotes = True
with yaml_path.open(encoding="utf-8") as handle:
    document = yaml.load(handle)

found = set()
for stage in document.get("stages", {}).values():
    outputs = stage.get("outs", [])
    for index, output in enumerate(outputs):
        if isinstance(output, str):
            output_path = output
            settings = {}
        elif len(output) == 1:
            output_path = next(iter(output))
            settings = dict(output[output_path] or {})
        else:
            continue
        if output_path not in local_paths:
            continue
        settings["push"] = False
        outputs[index] = {output_path: settings}
        found.add(output_path)

missing = sorted(local_paths - found)
if missing:
    raise RuntimeError("DVC outputs not found while setting push=false: " + ", ".join(missing))

with yaml_path.open("w", encoding="utf-8") as handle:
    yaml.dump(document, handle)

# ruamel may leave spaces at folded-line boundaries; keep generated YAML clean.
text = yaml_path.read_text(encoding="utf-8")
yaml_path.write_text(
    "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
    encoding="utf-8",
)
PY

dvc dag --dot >/dev/null
echo "Updated dvc.yaml. No pipeline stage was executed."
