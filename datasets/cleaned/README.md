# Canonical NMR Dataset Collection — Data Card

**Dataset release:** v1  
**Canonical schema:** v2  
**Core modalities:** `1H` and `13C` NMR

## Overview

This collection supports representation learning and benchmarking from paired
`1H` and `13C` NMR peak lists. It brings together simulated and literature-
derived spectra in one model-independent schema while preserving which peak
attributes were actually available in each source.

One row is one spectrum record, identified by `record_id`. A molecule can have
more than one record when it has distinct spectra, sources, or measurement
conditions. Those records are retained unless they are exact copies under the
defined duplicate rule.

The collection has two complementary parts:

- a rich peak-table corpus from MST-NMR, NMRexp, and NMRTrans/NMRSpec;
- a large SimNMR-PubChem (NMR-Solver) corpus of simulated shifts, kept as a
  separate shift-only component because it does not report multiplicity,
  coupling, or peak-range annotations.


## Sources and coverage

| Source | Domain | Peak information | Role in the collection |
|---|---|---|---|
| MST-NMR | Simulated | Paired `1H` and `13C` peak lists. Proton shifts, integration, multiplicity, ranges, and available J couplings; additional simulated carbon fields. | Rich simulated training data. |
| NMRexp | Literature-derived experimental reports | Proton annotations when reported; carbon shifts. Some records contain only one modality. | Rich experimental training data. |
| NMRTrans / NMRSpec | Literature-derived experimental reports | Paired `1H` and `13C` lists. Proton shifts, integration, multiplicity, ranges, and J couplings; carbon shifts. | Rich experimental training data. |
| SimNMR-PubChem / NMR-Solver | Simulated | Predicted `1H` and `13C` shifts grouped by supplied atom-equivalence classes; proton integration is available. Multiplicity, J couplings, and reported ranges are unavailable. | Large shift-only pretraining corpus. |


## Canonical representation

Each record contains provenance, RDKit-derived chemical metadata, and optional
peak lists:

| Level | Main fields | Convention |
|---|---|---|
| Record | `record_id`, `source`, `smiles`, `smiles_canonical`, `molecular_formula`, `atoms`, `nmr_frequency`, `nmr_solvent` | The original selected SMILES is retained; canonical SMILES, formula, and atom order are recalculated with RDKit. Experimental conditions, such as frequency and solvet are reported when available in the original source.   |
| `1H` peak | `shift`, `integration`, `multiplicity_raw`, `multiplicity`, `j_values`, `range_min`, `range_max`, `range_half_span` | Shift ranges are in ppm and J values in Hz. `multiplicity_raw` preserves the source text; `multiplicity` is the common vocabulary. |
| `13C` peak | `shift`, `integral`, `intensity`, `width` | Carbon shift is common to all rich sources; the other fields are source-specific. |
| NMR-Solver provenance | `equivalence_class`, `member_shifts` on `1H` peaks | Populated only when the source genuinely supplies atom-level equivalence information. |

The H and C modality fields are always lists; `[]` means that no usable peaks
are present for that nucleus. In the rich peak-table datasets, every proton
peak also has a J list and `[]` means that no numerical coupling was retained
or reported. `j_values = null` is reserved for shift-only datasets, where J was
not measured or generated. A numeric `J=0` remains a supplied value and is not
used as a missing-value marker.

Multiplicity uses a compact common vocabulary. The aliases `p → quint`,
`hept → sept`, and `brd → bd` are normalised without losing meaning; unsupported
reported labels become `<unk>` while the original label remains in
`multiplicity_raw`.

## How the collection is produced

The processing is designed to preserve source evidence first and create clean,
leakage-aware derived datasets second:

```text
source conversion → canonical merge → split protection / property matching
                  → quality cleaning → molecular-property calculation
```

1. **Canonicalisation.** Source-specific converters map each release into the
   common schema, derive structure metadata with RDKit, and preserve unavailable
   annotations as missing. The relevant converters are
   [MST-NMR](../../scripts/data/canonicalize/convert_mst_nmr.py),
   [NMRexp](../../scripts/data/canonicalize/convert_nmrexp.py),
   [NMRTrans](../../scripts/data/canonicalize/convert_nmrtrans.py), and
   [NMR-Solver](../../scripts/data/canonicalize/convert_nmrsolver.py).

2. **Merge and split protection.** Compatible canonical records are combined by
   [merge_datasets.py](../../scripts/data/postprocess/merge_datasets.py). The
   benchmark test set is disjoined from train/validation using connectivity
   InChIKeys, a deliberately conservative rule that prevents the same molecular
   connectivity from appearing on both sides. Property matching instead uses
   full InChIKeys, so stereochemical distinctions are preserved whenever the
   source structure encodes them. The corresponding steps are implemented in
   [disjoin_benchmark_from_train.py](../../scripts/data/postprocess/disjoin_benchmark_from_train.py)
   and [extract_annotated_peaks.py](../../scripts/data/postprocess/extract_annotated_peaks.py).

3. **Quality cleaning.** A record is removed when it has no peaks in either
   modality, non-finite shifts, proton shifts outside `[-5, 20]` ppm, carbon
   shifts outside `[-50, 300]` ppm, more than 60 peaks in a modality, more than
   six J values in one proton peak, non-finite or negative J values, a supplied
   non-positive proton integration, or a multi-fragment canonical SMILES.
   Reported `J=0` and missing integration are retained. No drug-likeness or
   molecular-property threshold is used.

   Exact duplicate spectra are identified by canonical SMILES plus sorted exact
   `1H` and `13C` shift lists. The best-annotated record is retained with a
   deterministic ID tie-breaker. Records for the same molecule with different
   shift lists remain valid separate spectra. This targets clear extraction
   errors and identical mined spectra without discarding meaningful replicate
   measurements. The implementation is
   [filter_dataset.py](../../scripts/data/postprocess/filter_dataset.py).

4. **Molecular properties.** RDKit descriptors, functional-group flags, and
   fingerprints are calculated after cleaning from canonical SMILES. This makes
   the descriptor rows align one-to-one with the final NMR records. The
   implementation is
   [calculate_mol_properties.py](../../scripts/data/postprocess/calculate_mol_properties.py).

## Split design and downstream property tasks

The rich corpus has a train/validation collection for representation learning
and a connectivity-disjoint benchmark test collection. The benchmark split
removes 26,930 overlapping source-test records before quality cleaning; the
cleaning stage removes a further 31,250 train/validation records and 1,390
benchmark-test records.

For the ADMET benchmarks, records matched to a property endpoint are removed
from the general train/validation corpus using exact full-InChIKey equality.
This removes 6,858 NMR records from that pretraining collection. Each final
property CSV has exactly one row per NMR `record_id` and the same ID set as its
paired NMR records. Repeated agreeing labels are collapsed; discordant repeated
LD50 labels are excluded rather than averaged.

| Endpoint | Task | Train/validation records | Test records | Target distribution |
|---|---|---:|---:|---|
| Ames | Binary classification | 2,305 | 409 | Positive fraction: 36.9% / 42.5% |
| LD50 Zhu | Regression | 2,498 | 515 | Mean ± SD: 2.12 ± 0.67 / 2.37 ± 0.80 |
| AqSolDB solubility | Regression | 3,413 | 682 | Mean ± SD: −2.44 ± 1.87 / −2.95 ± 2.05 |

Values in the final column are train/validation followed by test. The endpoint
definition and target units are inherited from the corresponding TDC release.

## Final collection analytics

The fully annotated train/validation and benchmark-test records are distributed
as follows. NMRexp remains partly single-modality because the collection keeps
valid source records rather than requiring all data to fit one model.

| Source | Train/validation records | Train/validation with both `1H` + `13C` | Benchmark-test records | Benchmark-test with both |
|---|---:|---:|---:|---:|
| MST-NMR | 699,526 | 699,526 | 72,441 | 72,441 |
| NMRexp | 965,921 | 703,242 | 95,315 | 69,104 |
| NMRTrans / NMRSpec | 171,779 | 171,779 | 12,355 | 12,355 |
| **Total** | **1,837,226** | **1,574,547** | **180,111** | **153,900** |

The three rich sources have closely matched molecular size but different
chemical composition. NMRexp and NMRTrans are modestly more lipophilic and
less polar on average than MST-NMR; MST-NMR has more amine and heteroaromatic
motifs. 
| Source | Heavy atoms | Exact molecular weight | RDKit logP | TPSA |
|---|---:|---:|---:|---:|
| MST-NMR | 22.45 ± 6.83 | 325.55 ± 95.67 | 3.17 ± 1.59 | 61.81 ± 28.37 |
| NMRexp | 22.78 ± 7.22 | 326.27 ± 101.91 | 4.18 ± 1.81 | 42.90 ± 25.79 |
| NMRTrans / NMRSpec | 22.88 ± 6.96 | 325.66 ± 98.15 | 4.04 ± 1.72 | 46.86 ± 26.59 |

Peak statistics below use all train/validation records. 
| Source | `1H` peaks / record | `13C` peaks / record | `1H` shift (ppm) | `13C` shift (ppm) | J coupling (Hz) |
|---|---|---|---|---|---|
| MST-NMR | 9.23 ± 3.38 [3, 28] | 17.39 ± 8.37 [4, 60] | 5.01 ± 2.47 [−1.91, 10.00] | 107.68 ± 46.02 [−15.84, 229.91] | 5.76 ± 3.84 [0.46, 64.82] |
| NMRexp | 7.36 ± 3.87 [0, 28] | 12.92 ± 7.17 [0, 55] | 5.42 ± 2.50 [−2.94, 16.99] | 109.05 ± 45.04 [−35.54, 297.45] | 7.59 ± 6.29 [0.00, 738.00] |
| NMRTrans / NMRSpec | 9.04 ± 3.08 [1, 29] | 15.64 ± 5.47 [5, 60] | 5.23 ± 2.51 [−5.00, 20.00] | 106.91 ± 46.22 [0.01, 297.00] | 7.53 ± 4.20 [0.00, 50.00] |

The large NMRexp maximum J value is a retained source annotation. The cleaning
policy rejects malformed J values and excessive J-list length but does not
silently impose an upper numerical J threshold. The plots use robust visual
limits only; the summary tables retain all values.

Full generated tables are available for the
[source inventory](analytics/source_inventory.csv),
[peak statistics](analytics/train_val_peak_summary.csv),
[multiplicity distribution](analytics/train_val_multiplicity_distribution.csv),
[molecular descriptors](analytics/train_val_molecular_property_summary.csv),
[functional-group prevalence](analytics/train_val_functional_group_prevalence.csv),
and [ADMET targets](analytics/admet_summary.csv).

![Molecular-property distributions by source](analytics/train_val_molecular_property_distributions.png)

![Peak distributions by source](analytics/train_val_peak_distributions.png)

The violin plots use a deterministic sample of at most 20,000 records per
source. For readability only, their displayed range is limited to
`Q1 − 2.5×IQR` through `Q3 + 2.5×IQR`; tables are calculated on all records.
They can be regenerated with the molecular-property and peak-analysis cells at the end of
[test_notebook.ipynb](../../scripts/test_notebook.ipynb).

## Intended use and limitations

This collection is intended for peak-level NMR representation learning,
cross-source robustness studies, model-native preprocessing experiments, and
property prediction with the provided leakage-aware benchmarks. SimNMR-PubChem
is appropriate for shift-only pretraining or a staged curriculum; rich peak
tables are appropriate when using integration, multiplicity, ranges, or J
couplings.

It is not a collection of raw FID or frequency-domain traces. Literature data
are structured reports mined from publications, so solvent, field strength,
and annotation completeness vary by source. Carbon intensity should not be
treated as quantitative integration. RDKit canonicalisation standardises the
representation of a supplied structure but does not apply a full salt,
tautomer, protonation, or stereochemistry normalisation policy. A model should
therefore be evaluated by source and modality availability, rather than assuming
that all records are interchangeable measurements.

## Reproducibility

The collection uses canonical schema version 2, RDKit-derived structure fields,
and Zstandard-compressed Parquet. Source conversion, merging, split protection,
cleaning, descriptor calculation, and the analytics above are all implemented
in the linked scripts. The broader schema rationale and source-specific mapping
rules are documented in [Datasets](../../contex/Datasets.md),
[Canonicalization Implementation Notes](../../contex/Canonicalization_Implementation_Notes.md),
[Dataset Filtering and Processing](../../contex/Dataset_Filtering_and_Processing.md),
[Dataset Analysis](../../contex/Dataset%20Analysis.md), and
[Properties Dataset](../../contex/Properties%20Dataset.md).
