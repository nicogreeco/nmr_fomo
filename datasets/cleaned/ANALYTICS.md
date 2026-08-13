# Canonical NMR Dataset Collection — Analytics

[Back to the main data card](README.md)

These files are generated from the final Parquets and their aligned molecular-
property CSVs. Statistics use every eligible row; plots use a deterministic
sample per source and limit only the visible range to `Q1 - 2.5×IQR` through
`Q3 + 2.5×IQR`.

The rich train and benchmark tables are stratified into MST-NMR, NMRexp, and
NMRTrans/NMRSpec. SimNMR and NMRGym are analyzed separately because their
intentionally missing rich annotations would otherwise obscure comparisons.

## Generated outputs

| Scope | Inventory | Molecular properties | Functional groups | Peak statistics | Annotation completeness | Multiplicity |
|---|---|---|---|---|---|---|
| Train/validation | [CSV](analytics/train_val/train_val_source_inventory.csv) | [CSV](analytics/train_val/train_val_molecular_property_summary.csv) | [CSV](analytics/train_val/train_val_functional_group_prevalence.csv) | [CSV](analytics/train_val/train_val_peak_summary.csv) | [CSV](analytics/train_val/train_val_proton_annotation_completeness.csv) | [CSV](analytics/train_val/train_val_multiplicity_distribution.csv) |
| Benchmark test | [CSV](analytics/test_benchmark/test_benchmark_source_inventory.csv) | [CSV](analytics/test_benchmark/test_benchmark_molecular_property_summary.csv) | [CSV](analytics/test_benchmark/test_benchmark_functional_group_prevalence.csv) | [CSV](analytics/test_benchmark/test_benchmark_peak_summary.csv) | [CSV](analytics/test_benchmark/test_benchmark_proton_annotation_completeness.csv) | [CSV](analytics/test_benchmark/test_benchmark_multiplicity_distribution.csv) |
| SimNMR-PubChem | [CSV](analytics/simnmr/nmrsolver_source_inventory.csv) | [CSV](analytics/simnmr/nmrsolver_molecular_property_summary.csv) | [CSV](analytics/simnmr/nmrsolver_functional_group_prevalence.csv) | [CSV](analytics/simnmr/nmrsolver_peak_summary.csv) | [CSV](analytics/simnmr/nmrsolver_proton_annotation_completeness.csv) | [CSV](analytics/simnmr/nmrsolver_multiplicity_distribution.csv) |
| NMRGym | [CSV](analytics/nmrgym/nmrgym_source_inventory.csv) | [CSV](analytics/nmrgym/nmrgym_molecular_property_summary.csv) | [CSV](analytics/nmrgym/nmrgym_functional_group_prevalence.csv) | [CSV](analytics/nmrgym/nmrgym_peak_summary.csv) | [CSV](analytics/nmrgym/nmrgym_proton_annotation_completeness.csv) | [CSV](analytics/nmrgym/nmrgym_multiplicity_distribution.csv) |
| ADMET | [CSV](analytics/admet/admet_source_inventory.csv) | [CSV](analytics/admet/admet_molecular_property_summary.csv) | [CSV](analytics/admet/admet_functional_group_prevalence.csv) | [CSV](analytics/admet/admet_peak_summary.csv) | [CSV](analytics/admet/admet_proton_annotation_completeness.csv) | [CSV](analytics/admet/admet_multiplicity_distribution.csv) |

The combined rich-source inventory is
[`source_inventory.csv`](analytics/collection/source_inventory.csv), and the ADMET target
summary is [`admet_summary.csv`](analytics/admet/admet_summary.csv). Each scope also
contains molecular-property and peak-distribution PNGs generated from the same
final records.

## Reproducing the analytics

From the project repository and main environment:

```bash
dvc repro analyze_cleaned_collection
```

The stage runs the maintained main, SimNMR, and NMRGym analysis modes and
records their inputs and outputs in `dvc.lock`.
