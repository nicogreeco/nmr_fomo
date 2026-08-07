# Canonical NMR Dataset Collection — Data Card

**Dataset release:** v1  
**Canonical schema:** v2  
**Spectral modalities:** `1H` and `13C` resonance-level peak lists

## Collection overview

This release brings several of the largest openly available processed NMR
corpora used by current deep-learning methods into one model-independent
schema. It combines simulated and literature-derived spectra while preserving
the provenance and annotation coverage of every source.

The collection has four functional components:

1. **`train_val`**, the common representation-learning pool built from the
   MST-NMR, NMRexp, and NMRTrans/NMRSpec train and validation partitions;
2. **`test_benchmark`**, the union of their published test partitions after a
   molecule-connectivity disjoin against all three rich train/validation pools;
3. **ADMET subsets**, exact molecule matches to three TDC endpoints, with the
   original TDC train/validation and test assignments;
4. **SimNMR-PubChem / NMR-Solver**, a much larger simulated shift-only
   component kept separate because its proton peaks contain shifts and
   equivalence-derived integration, while multiplicity, J coupling, and
   reported peak ranges are unavailable.

One Parquet row represents one spectrum record. A molecule may have multiple
records when spectra originate from different sources, simulations, reports,
or experimental conditions. Such records remain distinct unless their
canonical structure and both exact shift lists are identical under the
deduplication rule described below.

## Source provenance

| Component | Origin used by this collection | Reference |
|---|---|---|
| MST-NMR | Simulated paired spectra from the Multimodal Spectroscopic Dataset, using the processed NMRPeak LMDB release. | [MSD](https://doi.org/10.52202/079017-3996), [NMRPeak](https://arxiv.org/abs/2602.08752) |
| NMRexp | Experimental records mined from chemistry Supporting Information published between 2010 and 2024, using the quality-controlled NMRPeak subset. | [NMRexp](https://doi.org/10.1038/s41597-025-06245-5), [NMRPeak](https://arxiv.org/abs/2602.08752) |
| NMRTrans / NMRSpec | Experimental peak tables mined from chemistry Supporting Information published between 2013 and 2025; this collection uses the released 212,440-record model dataset. | [NMRTrans](https://arxiv.org/abs/2602.10158) |
| SimNMR-PubChem / NMR-Solver | PubChem-scale simulated atom-level `1H` and `13C` shifts grouped through supplied equivalence classes. | [NMR-Solver](https://doi.org/10.1038/s41467-026-71315-0) |
| ADMET labels | Ames, LD50 Zhu, and AqSolDB solubility endpoints with the official Therapeutics Data Commons splits. | [Property benchmark notes](../../contex/Properties%20Dataset.md) |

The source papers describe collection and upstream curation. This repository
starts from the released processed representations and records the additional
canonicalisation, split protection, and cleaning applied here.

## Final files

| Component | NMR records | Labels or derived features |
|---|---|---|
| Representation-learning pool | [`train_val.parquet`](train_val.parquet) | [`train_val_mol_properties.csv`](train_val_mol_properties.csv) |
| Connectivity-disjoint benchmark | [`test_benchmark.parquet`](test_benchmark.parquet) | [`test_benchmark_mol_properties.csv`](test_benchmark_mol_properties.csv) |
| SimNMR-PubChem shift-only pool | [`simnmr.parquet`](simnmr.parquet) | [`simnmr_mol_properties.csv`](simnmr_mol_properties.csv) |
| Ames | [`train_val.parquet`](admet/ames/train_val.parquet), [`test.parquet`](admet/ames/test.parquet) | Matched [train/validation](admet/ames/train_val.csv) and [test](admet/ames/test.csv) labels |
| LD50 Zhu | [`train_val.parquet`](admet/ld50_zhu/train_val.parquet), [`test.parquet`](admet/ld50_zhu/test.parquet) | Matched [train/validation](admet/ld50_zhu/train_val.csv) and [test](admet/ld50_zhu/test.csv) labels |
| AqSolDB solubility | [`train_val.parquet`](admet/solubility_aqsoldb/train_val.parquet), [`test.parquet`](admet/solubility_aqsoldb/test.parquet) | Matched [train/validation](admet/solubility_aqsoldb/train_val.csv) and [test](admet/solubility_aqsoldb/test.csv) labels |

Each molecular-property CSV has one row per final NMR `record_id`. It contains
RDKit descriptors, functional-group indicators, ECFP4 and MACCS fingerprints,
and an explicit RDKit processing status. Each ADMET CSV has the same unique
`record_id` set as its paired Parquet and contains the TDC molecule, label
`Y`, and source identifier.

## Canonical schema

The physical representation is Arrow/Parquet with nested lists and structs:

```text
CanonicalRecord
├── record and molecular provenance
├── atoms: list<string>
├── h_nmr_peaks: list<ProtonPeak>
│   ├── chemical shift and integration
│   ├── raw and harmonised multiplicity
│   ├── J-coupling list
│   ├── reported or reconstructed shift interval
│   └── optional equivalence-group provenance
└── c_nmr_peaks: list<CarbonPeak>
    └── shift plus optional source-specific simulated peak properties
```

The corresponding model-independent Python dataclasses are
[`CanonicalRecord`, `ProtonPeak`, and `CarbonPeak`](../../scripts/data/schema.py).
[`CanonicalParquetDataset`](../../scripts/data/dataset.py) streams one or more
Parquet files as validated `CanonicalRecord` objects, while
[`CanonicalNMRDataset`](../../scripts/data/dataset.py) provides the in-memory
equivalent for small collections.

### Record fields

| Field | Arrow type | Meaning |
|---|---|---|
| `record_id` | `string`, required | Stable source-qualified spectrum identifier; unique within each file. |
| `source` | `string` | Canonical source family. |
| `smiles` | `string` | Structure string selected from the source release. |
| `smiles_canonical` | `string` | Isomeric canonical SMILES recalculated with RDKit. |
| `molecular_formula` | `string` | Formula recalculated from the canonical RDKit molecule. |
| `nmr_frequency` | `string \| null` | Reported acquisition frequency when supplied. |
| `nmr_solvent` | `string \| null` | Reported solvent when supplied. |
| `atoms` | `list<string>` | Atom symbols in RDKit canonical-SMILES order, normally heavy atoms. |
| `h_nmr_peaks` | `list<struct<ProtonPeak>>` | Canonical proton resonances; always a list. |
| `c_nmr_peaks` | `list<struct<CarbonPeak>>` | Canonical carbon resonances; always a list. |

RDKit regenerates `smiles_canonical`, formula, and atom order from the selected
source SMILES. This gives every source the same structural representation and
records the RDKit version in the Parquet footer.

### Proton peak fields

| Field | Arrow type | Unit and convention |
|---|---|---|
| `shift` | `float64`, required | Chemical shift in ppm. |
| `integration` | `int64 \| null` | Number of represented protons when supplied or derivable. |
| `multiplicity_raw` | `string \| null` | Original source label. |
| `multiplicity` | `string \| null` | Harmonised label used across rich sources. |
| `j_values` | `list<float64> \| null` | J couplings in Hz. |
| `range_min` | `float64 \| null` | Lower endpoint of the reported shift interval, in ppm. |
| `range_max` | `float64 \| null` | Upper endpoint of the reported shift interval, in ppm. |
| `range_half_span` | `float64 \| null` | `(range_max - range_min) / 2`, in ppm. |
| `equivalence_class` | `int64 \| null` | Source equivalence-group identifier for atom-level simulated data. |
| `member_shifts` | `list<float64> \| null` | Original atom-level shifts represented by one grouped resonance. |

The canonical multiplicity vocabulary is:
`m, d, s, dd, t, ddd, q, dt, td, br, ddt, dq, tt, quint, dddd, qd, sept, ddp, ddq, bd, dqd`.
Lossless aliases are normalised as `p → quint`, `hept → sept`, and
`brd → bd`. Other supplied labels become `<unk>`, while their exact source
text remains available in `multiplicity_raw`.

### Carbon peak fields

| Field | Arrow type | Unit and convention |
|---|---|---|
| `shift` | `float64`, required | Chemical shift in ppm. |
| `integral` | `float64 \| null` | Source-supplied simulated carbon integral. |
| `intensity` | `float64 \| null` | Source-supplied simulated intensity. |
| `width` | `float64 \| null` | Source-supplied simulated carbon width in ppm. |

The three optional carbon quantities are populated by MST-NMR. Carbon peaks
from NMRexp, NMRTrans, and NMR-Solver carry the common shift field.

### Missing values

- The two modality columns are always lists. `[]` means that the record has no
  usable peaks for that nucleus.
- In the rich MST-NMR, NMRexp, and NMRTrans files, `j_values` is always a
  list. `[]` means that no numerical coupling is listed for that peak.
- Shift-only sources use `null` for annotations outside their source
  representation, including multiplicity, J values, and reported ranges.
- A numerical `J=0` remains a supplied value and can be masked during model
  preparation.
- After common cleaning, all 16,614,447 proton peaks in `train_val` and
  `test_benchmark` have populated positive integration, canonical
  multiplicity, and all three range fields.

Example streaming access:

```python
from data.dataset import CanonicalParquetDataset

records = CanonicalParquetDataset("datasets/cleaned/train_val.parquet")

for record in records:
    h_shifts = [peak.shift for peak in record.h_nmr_peaks]
    c_shifts = [peak.shift for peak in record.c_nmr_peaks]
```

Run repository code with `PYTHONPATH=scripts` so the `data` package is
available.

## Source-specific representation

### MST-NMR through NMRPeak

MST-NMR is the simulated rich source. Its NMRPeak release provides paired
proton and carbon resonances. Proton `centroid`, `nH`, `category`, parsed
J values, `rangeMin`, and `rangeMax` map directly to the shared fields. If
`centroid` is absent, conversion uses `delta`, followed by the midpoint of
the two range endpoints. A source peak containing only one shift receives a
point interval:

```text
range_min = range_max = shift
range_half_span = 0
```

MST-NMR also supplies the optional carbon `integral`, `intensity`, and
`width (ppm)`. This carbon width is a source-simulated peak property and is
separate from the proton `range_half_span`.

Converter:
[`convert_mst_nmr.py`](../../scripts/data/canonicalize/convert_mst_nmr.py).

### NMRexp through NMRPeak

NMRexp contains experimental spectra mined from literature and processed by
NMRPeak. Its proton mapping is the same as MST-NMR, including point intervals
for single reported shifts. Carbon records supply shifts only. Frequency and
solvent are retained where they occur in the processed release.

NMRexp is the source with partial modality coverage: an experimental report may
contain only `1H` or only `13C`. Those records remain useful and use an
empty list for the unavailable modality.

Converter:
[`convert_nmrexp.py`](../../scripts/data/canonicalize/convert_nmrexp.py).

### NMRTrans / NMRSpec

NMRTrans uses a processed NMRSpec collection mined from chemistry-paper
Supporting Information published between 2013 and 2025. The released proton
token has the form:

```text
[shift, range_half_span, multiplicity, integration, J_values]
```

The field called `peak_width` by the upstream loader is the half-span of the
reported interval:

```text
range_min = shift - range_half_span
range_max = shift + range_half_span
range_half_span = (range_max - range_min) / 2
```

Carbon values are shift lists. The release contains paired modalities and does
not supply solvent or frequency fields.

Converter:
[`convert_nmrtrans.py`](../../scripts/data/canonicalize/convert_nmrtrans.py).

### SimNMR-PubChem / NMR-Solver

SimNMR-PubChem is a simulated PubChem-scale shift database used by NMR-Solver.
The source stores atom-level predictions in `nmr_predict`, element identities
in `atom_index`, and supplied equivalence groups in `equi_class`. Conversion
creates one resonance per nucleus and equivalence class:

```text
shift = mean(member_shifts)
1H integration = number of hydrogen members
```

For proton resonances, `equivalence_class` and `member_shifts` preserve the
atom-level source information. Multiplicity, J coupling, and reported ranges
are represented as unavailable annotations. Carbon equivalence groups are also
reduced to their mean shift. This shift-only representation is distributed as
a separate component so models can use it for large-scale shift pretraining or
a staged simulated-to-rich curriculum.

Converter:
[`convert_nmrsolver.py`](../../scripts/data/canonicalize/convert_nmrsolver.py).

## Processing and split rationale

The reproducible data flow is:

```text
source releases
    → source-specific canonicalisation
    → rich train/validation and source-test merges
    → benchmark molecule disjoin / ADMET exact matching
    → common quality cleaning and exact-shift deduplication
    → molecular descriptors and collection analytics
```

### 1. Canonicalisation and merge

Each converter translates the corresponding processed release into schema v2,
normalises multiplicity, recalculates molecular metadata, validates every
record, and writes provenance in the Parquet footer. The Arrow writer and
shared source mappings are implemented in
[`common.py`](../../scripts/data/canonicalize/common.py).

[`merge_datasets.py`](../../scripts/data/postprocess/merge_datasets.py)
concatenates compatible schema-v2 files while checking schema and RDKit
metadata:

- source train and validation partitions form the common `train_val` pool;
- source test partitions form the initial rich benchmark pool.

### 2. Connectivity-disjoint benchmark

The benchmark starts from the published test partitions of MST-NMR, NMRexp,
and NMRTrans/NMRSpec. A molecule appearing in any of the corresponding rich
train/validation pools would give a model source-dependent prior exposure,
even if its spectrum or record ID differed. Therefore
[`disjoin_benchmark_from_train.py`](../../scripts/data/postprocess/disjoin_benchmark_from_train.py)
removes benchmark records whose RDKit connectivity InChIKey occurs in the
merged train/validation pool.

The comparison uses the connectivity block rather than the full InChIKey. It
therefore applies a conservative molecule-level boundary that also groups
stereoisomers sharing the same connectivity. This removed 26,930 records before
common cleaning. The final `test_benchmark` is a shared test set whose
molecular connectivities are absent from every MST-NMR, NMRexp, and NMRTrans
train/validation partition represented in this collection.

### 3. ADMET matching and disjoin

The ADMET workflow begins from the official TDC `train_val` and `test`
partitions. [`admet_overlap_audit.ipynb`](../../scripts/data/postprocess/admet_overlap_audit.ipynb)
derives full RDKit InChIKeys for property and NMR structures. Full keys are
used here because stereochemical distinctions can affect measured properties.

[`extract_annotated_peaks.py`](../../scripts/data/postprocess/extract_annotated_peaks.py)
materialises all exact NMR matches for Ames, LD50 Zhu, and AqSolDB solubility.
The union of 6,858 matched NMR `record_id` values is removed from the common
pretraining pool before its quality cleaning. This separation prevents a
downstream labelled spectrum from also occurring in `train_val`.

Multiple spectra of one molecule remain separate supervised examples.
Repeated property rows are resolved at molecular-identity level:

- agreeing Ames labels and the one agreeing repeated LD50 value are collapsed;
- ten LD50 identities with discordant labels are removed, affecting 8
  train/validation and 5 test NMR records;
- the solubility matches contain no repeated property rows.

### 4. Common quality cleaning

[`filter_dataset.py`](../../scripts/data/postprocess/filter_dataset.py)
applies the same simple physical and structural checks to every rich canonical
file. A record is retained when at least one modality contains peaks and:

- all shifts are finite;
- proton shifts lie in `[-5, 20]` ppm;
- carbon shifts lie in `[-50, 300]` ppm;
- each modality contains at most 60 peaks;
- each proton peak contains at most six J values;
- J values are finite and non-negative;
- each supplied proton integration is positive;
- the canonical SMILES contains one connected fragment.

The filter keeps supplied `J=0` values. Molecular weight, logP, TPSA, and
drug-likeness descriptors do not participate in record acceptance.

The filter removed 31,250 records from the ADMET-disjoint train/validation pool
and 1,390 from the connectivity-disjoint benchmark. Reason counts can overlap
when one record violates more than one condition:

| Reason | Train/validation | Benchmark test |
|---|---:|---:|
| Exact duplicate shift signature | 20,081 | 239 |
| Non-positive supplied `1H` integration | 6,358 | 665 |
| More than 60 `13C` peaks | 2,015 | 215 |
| Multi-fragment canonical SMILES | 1,511 | 163 |
| `1H` shift outside range | 1,048 | 89 |
| `13C` shift outside range | 335 | 29 |
| More than six J values in one peak | 2 | 0 |

### 5. Exact-shift deduplication

Deduplication targets identical extracted spectra rather than all records of
the same molecule. Its exact identity key is:

```text
canonical SMILES
+ sorted exact 1H shift list
+ sorted exact 13C shift list
```

No rounding or tolerance is applied. An unavailable modality contributes an
empty shift list. A deterministic hash narrows candidate groups, after which
the actual SMILES and both shift lists are compared, so hash equality alone
never removes a row.

Within one exact group, selection prefers the record with more populated proton
integration, multiplicity, J-list, and range annotations, followed by fewer
missing or `<unk>` annotations and the lexicographically smallest
`record_id`. This preserves simulated/experimental pairs, replicate spectra,
and condition-dependent measurements whenever their shifts differ.

The pre-cleaning audit of 1,868,476 ADMET-disjoint train/validation rows found
20,128 exact groups, all pairs: 16,037 NMRexp–NMRTrans pairs and 4,091 internal
MST-NMR pairs. After validity filters, 20,081 duplicate rows remained eligible
for removal.

### 6. Molecular features and analytics

[`calculate_mol_properties.py`](../../scripts/data/postprocess/calculate_mol_properties.py)
runs after cleaning, so descriptor rows align with final record IDs. It
calculates exact molecular weight, RDKit logP, TPSA, HBA, HBD, rotatable bonds,
fraction Csp3, aromatic atom fraction, nine SMARTS functional-group flags,
radius-2 2,048-bit Morgan/ECFP4 fingerprints, and the 166 usable MACCS keys.

[`analyze_cleaned_datasets.py`](../../scripts/data/postprocess/analyze_cleaned_datasets.py)
reproduces the molecular and peak analyses for `train_val`,
`test_benchmark`, and the three ADMET cohorts. Summary CSVs use every record.
Violin plots use a deterministic sample of at most 20,000 records per group;
only their visible range is limited to
`Q1 − 2.5×IQR` through `Q3 + 2.5×IQR`.

## Dataset-specific composition

### Rich train/validation collection

`train_val.parquet` combines the original train and validation partitions of
the three rich sources after removing exact ADMET matches and applying common
cleaning. It contains 1,837,226 records and supports representation learning,
source-aware sampling, and later task-specific split construction.

### Connectivity-disjoint benchmark

`test_benchmark.parquet` contains the cleaned source-test records that pass
the connectivity disjoin. It contains 180,111 records. The source balance and
molecular-property distributions closely follow the train/validation
collection, making it suitable for cross-model comparison without an obvious
composition shift introduced by the disjoin step.

| Dataset | Source | Records | Unique canonical SMILES | With `1H` | With `13C` | With both |
|---|---|---:|---:|---:|---:|---:|
| Train/validation | MST-NMR | 699,526 | 699,469 | 699,526 | 699,526 | 699,526 |
| Train/validation | NMRexp | 965,921 | 965,921 | 844,532 | 824,631 | 703,242 |
| Train/validation | NMRTrans / NMRSpec | 171,779 | 171,779 | 171,779 | 171,779 | 171,779 |
| **Train/validation** | **All** | **1,837,226** | **1,772,773** | **1,715,837** | **1,695,936** | **1,574,547** |
| Benchmark test | MST-NMR | 72,441 | 72,441 | 72,441 | 72,441 | 72,441 |
| Benchmark test | NMRexp | 95,315 | 95,315 | 83,298 | 81,121 | 69,104 |
| Benchmark test | NMRTrans / NMRSpec | 12,355 | 12,355 | 12,355 | 12,355 | 12,355 |
| **Benchmark test** | **All** | **180,111** | **179,483** | **168,094** | **165,917** | **153,900** |

NMRexp accounts for all single-modality records. MST-NMR and NMRTrans remain
paired throughout both rich files.

### ADMET property subsets

Each endpoint keeps the official TDC split and contains spectra drawn from all
three rich sources:

| Endpoint | Split | MST-NMR | NMRexp | NMRTrans | Total |
|---|---|---:|---:|---:|---:|
| Ames | Train/validation | 827 | 1,104 | 374 | 2,305 |
| Ames | Test | 143 | 198 | 68 | 409 |
| LD50 Zhu | Train/validation | 1,148 | 1,035 | 315 | 2,498 |
| LD50 Zhu | Test | 265 | 198 | 52 | 515 |
| AqSolDB solubility | Train/validation | 1,538 | 1,418 | 457 | 3,413 |
| AqSolDB solubility | Test | 333 | 278 | 71 | 682 |

The same molecule can have several NMR records, so record counts exceed unique
property-SMILES counts:

| Endpoint | Task | Train/validation records / molecules | Test records / molecules | Target summary, train / test |
|---|---|---:|---:|---|
| Ames | Binary classification | 2,305 / 1,641 | 409 / 311 | Positive: 36.9% / 42.5% |
| LD50 Zhu | Regression | 2,498 / 1,853 | 515 / 396 | 2.12 ± 0.67 [−0.34, 5.51] / 2.37 ± 0.80 [0.29, 5.14] |
| AqSolDB solubility | Regression | 3,413 / 2,499 | 682 / 525 | −2.44 ± 1.87 [−10.10, 1.63] / −2.95 ± 2.05 [−8.70, 1.10] |

Target definitions and units follow the corresponding TDC releases.

### SimNMR-PubChem / NMR-Solver shift-only component

The source-level canonical scan contains 105,764,875 simulated records.
After source conversion and common cleaning, [`simnmr.parquet`](simnmr.parquet)
contains 105,509,616 records. Its 97,936,772 approximate unique canonical
SMILES indicate that it is a large molecule-level simulation collection with a
smaller number of repeated records. Both modalities are non-empty in
105,476,128 records (99.968%). Its shift-only annotation policy and much
larger scale make separate reporting and sampling appropriate.

## Measured analytics

All values below are `mean ± sample SD [min, max]` and use every eligible
value in the named final files.

### Molecular composition by source

Train/validation:

| Source | Heavy atoms | Exact molecular weight | RDKit logP | TPSA |
|---|---|---|---|---|
| MST-NMR | 22.45 ± 6.83 [5, 37] | 325.55 ± 95.67 [66.05, 1083.22] | 3.17 ± 1.59 [−7.94, 12.60] | 61.81 ± 28.37 [0, 317.12] |
| NMRexp | 22.78 ± 7.22 [3, 84] | 326.27 ± 101.91 [45.07, 1411.55] | 4.18 ± 1.81 [−10.72, 20.64] | 42.90 ± 25.79 [0, 434.52] |
| NMRTrans / NMRSpec | 22.88 ± 6.96 [1, 64] | 325.66 ± 98.15 [36.01, 898.80] | 4.04 ± 1.72 [−5.40, 18.93] | 46.86 ± 26.59 [0, 212.26] |

| Source | HBA | HBD | Rotatable bonds | Fraction Csp3 | Aromatic atom fraction |
|---|---|---|---|---|---|
| MST-NMR | 3.87 ± 1.72 [0, 17] | 1.05 ± 0.97 [0, 14] | 4.20 ± 2.59 [0, 30] | 0.357 ± 0.217 [0, 1] | 0.465 ± 0.210 [0, 1] |
| NMRexp | 2.82 ± 1.62 [0, 25] | 0.41 ± 0.64 [0, 12] | 4.15 ± 2.58 [0, 36] | 0.274 ± 0.199 [0, 1] | 0.525 ± 0.219 [0, 1] |
| NMRTrans / NMRSpec | 2.98 ± 1.64 [0, 15] | 0.51 ± 0.69 [0, 8] | 4.27 ± 2.68 [0, 48] | 0.297 ± 0.207 [0, 1] | 0.506 ± 0.221 [0, 1] |

Benchmark test:

| Source | Heavy atoms | Exact molecular weight | RDKit logP | TPSA |
|---|---|---|---|---|
| MST-NMR | 22.71 ± 6.74 [5, 35] | 329.33 ± 94.15 [70.04, 849.74] | 3.20 ± 1.58 [−6.04, 11.43] | 62.58 ± 28.32 [0, 272.13] |
| NMRexp | 23.12 ± 7.20 [3, 66] | 331.49 ± 101.68 [42.05, 1188.51] | 4.24 ± 1.82 [−5.94, 17.52] | 43.28 ± 25.88 [0, 376.16] |
| NMRTrans / NMRSpec | 23.81 ± 6.93 [6, 54] | 339.80 ± 97.45 [84.06, 770.00] | 4.14 ± 1.78 [−3.53, 17.20] | 50.20 ± 27.12 [0, 182.71] |

| Source | HBA | HBD | Rotatable bonds | Fraction Csp3 | Aromatic atom fraction |
|---|---|---|---|---|---|
| MST-NMR | 3.93 ± 1.72 [0, 15] | 1.05 ± 0.97 [0, 9] | 4.27 ± 2.59 [0, 26] | 0.352 ± 0.212 [0, 1] | 0.470 ± 0.206 [0, 1] |
| NMRexp | 2.86 ± 1.63 [0, 17] | 0.41 ± 0.64 [0, 11] | 4.21 ± 2.61 [0, 36] | 0.273 ± 0.198 [0, 1] | 0.526 ± 0.219 [0, 1] |
| NMRTrans / NMRSpec | 3.20 ± 1.67 [0, 12] | 0.54 ± 0.71 [0, 5] | 4.54 ± 2.83 [0, 26] | 0.313 ± 0.209 [0, 1] | 0.485 ± 0.222 [0, 1] |

The molecular-size distributions are close across sources and splits. The main
source signal is chemical composition: MST-NMR is more polar and has higher
HBA/HBD averages, while NMRexp and NMRTrans are more lipophilic and more
aromatic. Selected functional-group prevalences show the same pattern:

| Dataset / source | Amine | Amide | Ester | Halogenated | Heteroaromatic |
|---|---:|---:|---:|---:|---:|
| Train — MST-NMR | 38.7% | 36.2% | 27.4% | 45.7% | 55.6% |
| Train — NMRexp | 15.2% | 21.4% | 25.8% | 36.9% | 35.0% |
| Train — NMRTrans | 15.3% | 24.9% | 28.7% | 33.8% | 33.5% |
| Test — MST-NMR | 39.1% | 36.5% | 26.9% | 46.7% | 57.4% |
| Test — NMRexp | 15.2% | 21.2% | 25.9% | 37.5% | 35.6% |
| Test — NMRTrans | 15.8% | 25.8% | 31.0% | 34.4% | 33.4% |

![Train/validation molecular-property distributions](analytics/train_val/train_val_molecular_property_distributions.png)

![Benchmark-test molecular-property distributions](analytics/test_benchmark/test_benchmark_molecular_property_distributions.png)

### Peak composition by source

Train/validation peak counts:

| Source | `1H` peaks / record | `13C` peaks / record | `1H` peaks / heavy atom | `13C` peaks / heavy atom |
|---|---|---|---|---|
| MST-NMR | 9.23 ± 3.38 [3, 28] | 17.39 ± 8.37 [4, 60] | 0.421 ± 0.126 [0.086, 1.500] | 0.772 ± 0.289 [0.133, 5.600] |
| NMRexp | 7.36 ± 3.87 [0, 28] | 12.92 ± 7.17 [0, 55] | 0.336 ± 0.171 [0, 1.500] | 0.574 ± 0.263 [0, 1.000] |
| NMRTrans / NMRSpec | 9.04 ± 3.08 [1, 29] | 15.64 ± 5.47 [5, 60] | 0.409 ± 0.128 [0.029, 2.200] | 0.688 ± 0.144 [0.161, 6.000] |

| Source | `1H` shift (ppm) | `13C` shift (ppm) | J coupling (Hz) |
|---|---|---|---|
| MST-NMR | 5.01 ± 2.47 [−1.91, 10.00] | 107.68 ± 46.02 [−15.84, 229.91] | 5.76 ± 3.84 [0.46, 64.82] |
| NMRexp | 5.42 ± 2.50 [−2.94, 16.99] | 109.05 ± 45.04 [−35.54, 297.45] | 7.59 ± 6.29 [0.00, 738.00] |
| NMRTrans / NMRSpec | 5.23 ± 2.51 [−5.00, 20.00] | 106.91 ± 46.22 [0.01, 297.00] | 7.53 ± 4.20 [0.00, 50.00] |

Benchmark-test peak counts:

| Source | `1H` peaks / record | `13C` peaks / record | `1H` peaks / heavy atom | `13C` peaks / heavy atom |
|---|---|---|---|---|
| MST-NMR | 9.28 ± 3.38 [3, 26] | 17.62 ± 8.33 [4, 60] | 0.417 ± 0.123 [0.088, 1.333] | 0.773 ± 0.289 [0.182, 4.583] |
| NMRexp | 7.42 ± 3.90 [0, 26] | 13.04 ± 7.25 [0, 54] | 0.333 ± 0.170 [0, 1.333] | 0.571 ± 0.265 [0, 1.000] |
| NMRTrans / NMRSpec | 9.45 ± 3.13 [1, 34] | 16.35 ± 5.59 [5, 60] | 0.411 ± 0.129 [0.029, 1.467] | 0.692 ± 0.155 [0.240, 3.800] |

| Source | `1H` shift (ppm) | `13C` shift (ppm) | J coupling (Hz) |
|---|---|---|---|
| MST-NMR | 5.04 ± 2.47 [−0.99, 10.00] | 108.01 ± 45.91 [−12.25, 229.92] | 5.74 ± 3.83 [0.47, 58.43] |
| NMRexp | 5.43 ± 2.50 [−2.92, 16.95] | 109.07 ± 45.02 [−15.53, 260.40] | 7.59 ± 6.44 [0.00, 740.90] |
| NMRTrans / NMRSpec | 5.13 ± 2.51 [−0.37, 20.00] | 105.60 ± 46.96 [0.03, 296.00] | 7.60 ± 4.26 [0.00, 50.00] |

The train and benchmark peak distributions are also closely aligned. The
NMRexp maximum J values of 738.0 and 740.9 Hz are retained numerical source
annotations. The common cleaning rule constrains J-list length and numerical
validity while leaving the upper J magnitude available for downstream
inspection.

![Train/validation peak distributions](analytics/train_val/train_val_peak_distributions.png)

![Benchmark-test peak distributions](analytics/test_benchmark/test_benchmark_peak_distributions.png)

### SimNMR-PubChem composition

SimNMR-PubChem covers substantially larger and more structurally diverse
molecules than the rich literature-derived collection. Molecular properties
are not used as cleaning thresholds; the wide extrema below remain part of the
release and are visible in the complete machine-readable tables.

| Source | Heavy atoms | Exact molecular weight | RDKit logP | TPSA |
|---|---|---|---|---|
| SimNMR-PubChem | 25.99 ± 10.46 [1, 336] | 374.00 ± 143.61 [1.01, 15624.27] | 3.44 ± 2.50 [−59.98, 109.11] | 70.89 ± 42.00 [0, 1674.08] |

| Source | `1H` peaks / record | `13C` peaks / record | `1H` shift (ppm) | `13C` shift (ppm) |
|---|---|---|---|---|
| SimNMR-PubChem | 12.49 ± 5.57 [0, 60] | 17.08 ± 7.40 [0, 60] | 5.03 ± 2.66 [−4.52, 19.97] | 99.22 ± 49.86 [−34.88, 248.88] |

Integration is available from the supplied equivalence classes. Multiplicity,
J couplings, and peak ranges are unavailable by design, so they are null and
the SimNMR peak figure omits the J violin.

![SimNMR molecular-property distributions](analytics/simnmr/nmrsolver_molecular_property_distributions.png)

![SimNMR peak distributions](analytics/simnmr/nmrsolver_peak_distributions.png)

### Rich proton annotation completeness

All null counts for integration, canonical multiplicity, the three range
fields, and J lists are zero in both rich files. Empty J lists describe peaks
with no numerical coupling listed:

| Dataset | Source | Proton peaks | Empty J lists |
|---|---|---:|---:|
| Train/validation | MST-NMR | 6,457,766 | 2,740,153 (42.4%) |
| Train/validation | NMRexp | 7,108,512 | 3,985,995 (56.1%) |
| Train/validation | NMRTrans / NMRSpec | 1,552,069 | 868,481 (56.0%) |
| Benchmark test | MST-NMR | 672,056 | 284,859 (42.4%) |
| Benchmark test | NMRexp | 707,300 | 396,890 (56.1%) |
| Benchmark test | NMRTrans / NMRSpec | 116,744 | 65,197 (55.8%) |

### ADMET cohort composition

The three labelled cohorts contain smaller molecules than the general rich
collection and cover different chemical-property ranges:

| Endpoint | Records | Heavy atoms | Molecular weight | RDKit logP | TPSA |
|---|---:|---|---|---|---|
| Ames | 2,714 | 12.69 ± 4.79 [3, 38] | 180.59 ± 66.79 [41.03, 539.76] | 2.22 ± 1.40 [−3.62, 9.24] | 36.79 ± 25.99 [0, 213.28] |
| LD50 Zhu | 3,013 | 12.83 ± 5.21 [3, 38] | 185.41 ± 76.06 [41.03, 570.80] | 2.11 ± 1.36 [−3.33, 9.89] | 36.22 ± 23.07 [0, 199.73] |
| AqSolDB solubility | 4,095 | 13.12 ± 5.38 [3, 40] | 189.41 ± 77.60 [41.03, 570.80] | 2.21 ± 1.55 [−7.57, 11.08] | 38.20 ± 27.63 [0, 268.68] |

| Endpoint | `1H` peaks / record | `13C` peaks / record | `1H` shift (ppm) | `13C` shift (ppm) | J (Hz) |
|---|---|---|---|---|---|
| Ames | 5.08 ± 2.35 [0, 21] | 7.83 ± 4.34 [0, 45] | 5.58 ± 2.60 [0.09, 16.26] | 108.31 ± 46.68 [−5.10, 221.16] | 6.10 ± 3.72 [0.40, 43.60] |
| LD50 Zhu | 5.33 ± 2.49 [0, 18] | 8.01 ± 4.80 [0, 56] | 4.81 ± 2.65 [0.09, 15.70] | 99.07 ± 51.38 [−5.10, 221.16] | 6.15 ± 3.83 [0.00, 90.10] |
| AqSolDB solubility | 5.30 ± 2.61 [0, 20] | 8.27 ± 5.08 [0, 60] | 4.87 ± 2.75 [0.03, 16.90] | 98.33 ± 52.13 [−5.44, 221.16] | 6.20 ± 3.71 [0.00, 47.20] |

These cohort tables pool their official train/validation and test partitions
for descriptive analysis; their supervised split assignments remain unchanged.

![ADMET molecular-property distributions](analytics/admet/admet_molecular_property_distributions.png)

![ADMET peak distributions](analytics/admet/admet_peak_distributions.png)

## Complete generated analytics

The concise tables above are backed by machine-readable outputs:

| Scope | Inventory | Molecular properties | Functional groups | Peak statistics | Annotation completeness | Multiplicity |
|---|---|---|---|---|---|---|
| Train/validation | [CSV](analytics/train_val/train_val_source_inventory.csv) | [CSV](analytics/train_val/train_val_molecular_property_summary.csv) | [CSV](analytics/train_val/train_val_functional_group_prevalence.csv) | [CSV](analytics/train_val/train_val_peak_summary.csv) | [CSV](analytics/train_val/train_val_proton_annotation_completeness.csv) | [CSV](analytics/train_val/train_val_multiplicity_distribution.csv) |
| Benchmark test | [CSV](analytics/test_benchmark/test_benchmark_source_inventory.csv) | [CSV](analytics/test_benchmark/test_benchmark_molecular_property_summary.csv) | [CSV](analytics/test_benchmark/test_benchmark_functional_group_prevalence.csv) | [CSV](analytics/test_benchmark/test_benchmark_peak_summary.csv) | [CSV](analytics/test_benchmark/test_benchmark_proton_annotation_completeness.csv) | [CSV](analytics/test_benchmark/test_benchmark_multiplicity_distribution.csv) |
| SimNMR-PubChem | [CSV](analytics/simnmr/nmrsolver_source_inventory.csv) | [CSV](analytics/simnmr/nmrsolver_molecular_property_summary.csv) | [CSV](analytics/simnmr/nmrsolver_functional_group_prevalence.csv) | [CSV](analytics/simnmr/nmrsolver_peak_summary.csv) | [CSV](analytics/simnmr/nmrsolver_proton_annotation_completeness.csv) | [CSV](analytics/simnmr/nmrsolver_multiplicity_distribution.csv) |
| ADMET cohorts | [CSV](analytics/admet/admet_source_inventory.csv) | [CSV](analytics/admet/admet_molecular_property_summary.csv) | [CSV](analytics/admet/admet_functional_group_prevalence.csv) | [CSV](analytics/admet/admet_peak_summary.csv) | [CSV](analytics/admet/admet_proton_annotation_completeness.csv) | [CSV](analytics/admet/admet_multiplicity_distribution.csv) |

The combined [source inventory](analytics/collection/source_inventory.csv) and
[ADMET target summary](analytics/admet/admet_summary.csv) provide the aggregate
record and label counts.

## Use considerations

The primary learning unit is a structured resonance-level peak list linked to
a molecular structure. The rich collection supports models that consume
chemical shifts together with integration, multiplicity, ranges, and J
couplings. The SimNMR-PubChem component supports shift-set pretraining and
simulated-to-experimental curricula. The ADMET files support frozen-encoder
probes and supervised property-prediction studies with exact spectrum-to-label
alignment.

Literature-mined records vary in solvent, field strength, and reporting
practice. Source-aware evaluation is therefore informative alongside aggregate
metrics. Carbon `integral`, `intensity`, and `width` are MST-specific
simulated quantities. RDKit canonicalisation standardises the supplied
structure representation; salts, protonation states, tautomers, and missing
stereochemical information retain the distinctions present in the source
SMILES and identity rules described above.

## Reproducing the analytics

From the repository root:

```bash
PYTHONPATH=scripts python scripts/data/postprocess/analyze_cleaned_datasets.py
```

Use `--cleaned-root` for a different final collection directory and
`--sample-per-group` to change only the plotting sample size.
The separate SimNMR analysis uses:

```bash
PYTHONPATH=scripts python scripts/data/postprocess/analyze_cleaned_datasets.py \
  --nmrsolver-parquet datasets/cleaned/simnmr.parquet
```

Further implementation rationale and source audits are documented in
[`Datasets.md`](../../contex/Datasets.md),
[`Canonicalization_Implementation_Notes.md`](../../contex/Canonicalization_Implementation_Notes.md),
[`Multiplicity analysis.md`](../../contex/Multiplicity%20analysis.md),
[`Dataset_Filtering_and_Processing.md`](../../contex/Dataset_Filtering_and_Processing.md),
[`Dataset Analysis.md`](../../contex/Dataset%20Analysis.md), and
[`Properties Dataset.md`](../../contex/Properties%20Dataset.md).
