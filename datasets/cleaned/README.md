---
license: cc-by-4.0
pretty_name: Canonical NMR Dataset Collection
task_categories:
- feature-extraction
tags:
- chemistry
- nmr
- spectroscopy
- molecular-representation-learning
size_categories:
- 100M<n<1B
---

# Canonical NMR Dataset Collection — Data Card

- **Dataset release:** v4
- **Canonical schema:** v2
- **Spectral modalities:** `1H` and `13C` resonance-level peak lists

## Collection overview

This release brings several of the largest openly available processed NMR
corpora used by current deep-learning methods into one model-independent
schema. It combines simulated and literature-derived spectra while preserving
the provenance and annotation coverage of every source.

The collection has five functional components:

1. **`rich`**, the common representation-learning pool built from the
   MST-NMR, NMRexp, and NMRTrans/NMRSpec train and validation partitions;
2. **`test_benchmark`**, the union of their published test partitions after
   moving SimNMR-overlapping rich spectra into training and removing residual
   exact-canonical-SMILES overlap with the extended train pool and NMRGym;
3. **ADMET subsets**, exact molecule matches to four TDC endpoints plus
   Sangster logP; TDC assignments are preserved and Sangster uses a deterministic
   molecule-disjoint 80/20 split;
4. **SimNMR-PubChem / NMR-Solver**, a much larger simulated shift-only
   component kept separate because its proton peaks contain shifts and
   equivalence-derived integration, while multiplicity, J coupling, and
   reported peak ranges are unavailable;
5. **NMRGym**, a smaller experimental shift-only component with paired `1H`
   and `13C` resonance lists and no supplied integration, multiplicity, J
   coupling, or reported peak ranges.

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
| NMRGym | Experimental paired `1H` and `13C` shift lists released as the scaffold-split NMRGym benchmark used by UltraNMR. | [UltraNMR](https://arxiv.org/abs/2606.20756), [source repository](https://github.com/wuycM/UltraNMR) |
| ADMET labels | Ames, LD50 Zhu, AqSolDB solubility, and AstraZeneca lipophilicity use official Therapeutics Data Commons splits; Sangster logP uses the high-confidence `P` subset from Zenodo. | [Preparation implementation](https://github.com/nicogreeco/nmr_fomo/tree/main/scripts/data/postprocess) |

The source papers describe collection and upstream curation. This repository
starts from the released processed representations and records the additional
canonicalisation, split protection, and cleaning applied here.

## Final files

| Component | NMR records | Labels or derived features |
|---|---|---|
| Representation-learning pool | [`rich.parquet`](rich.parquet) | [`rich_mol_properties.parquet`](rich_mol_properties.parquet) |
| Molecule-disjoint benchmark | [`test_benchmark.parquet`](test_benchmark.parquet) | [`test_benchmark_mol_properties.parquet`](test_benchmark_mol_properties.parquet) |
| SimNMR-PubChem shift-only pool | [`simnmr.parquet`](simnmr.parquet) | [`simnmr_mol_properties.parquet`](simnmr_mol_properties.parquet) |
| NMRGym experimental shift-only pool | [`nmrgym.parquet`](nmrgym.parquet) | [`nmrgym_mol_properties.parquet`](nmrgym_mol_properties.parquet) |
| Ames | [`train_val.parquet`](admet/ames/train_val.parquet), [`test.parquet`](admet/ames/test.parquet) | Matched [train/validation](admet/ames/train_val.csv) and [test](admet/ames/test.csv) labels |
| LD50 Zhu | [`train_val.parquet`](admet/ld50_zhu/train_val.parquet), [`test.parquet`](admet/ld50_zhu/test.parquet) | Matched [train/validation](admet/ld50_zhu/train_val.csv) and [test](admet/ld50_zhu/test.csv) labels |
| AqSolDB solubility | [`train_val.parquet`](admet/solubility_aqsoldb/train_val.parquet), [`test.parquet`](admet/solubility_aqsoldb/test.parquet) | Matched [train/validation](admet/solubility_aqsoldb/train_val.csv) and [test](admet/solubility_aqsoldb/test.csv) labels |
| AstraZeneca lipophilicity | [`train_val.parquet`](admet/lipophilicity_astrazeneca/train_val.parquet), [`test.parquet`](admet/lipophilicity_astrazeneca/test.parquet) | Matched [train/validation](admet/lipophilicity_astrazeneca/train_val.csv) and [test](admet/lipophilicity_astrazeneca/test.csv) labels |
| Sangster logP | [`train_val.parquet`](admet/sangster_logp/train_val.parquet), [`test.parquet`](admet/sangster_logp/test.parquet) | Deterministic molecule-level [train/validation](admet/sangster_logp/train_val.csv) and [test](admet/sangster_logp/test.csv) labels |

The adjacent `*_report.json` files are compact processing reports generated by
the same DVC run; `admet/preparation_report.json` records cohort construction.

### Creating the FoMoNMR training splits

After cloning the project repository, download this release into
`datasets/cleaned/` and run the two final DVC stages:

```bash
hf download niccogreek/nmr-canonical-cleaned \
  --repo-type dataset \
  --local-dir datasets/cleaned
dvc repro --single-item split_foundation_datasets
dvc repro --single-item split_maccs_probe
```

This creates the molecule-safe train/validation files under
`datasets/train_splits/` and the fixed MACCS probe split. It does not rebuild
the raw, canonical, or intermediate datasets.

Each molecular-property Parquet has one row per final NMR `record_id`. It contains
RDKit descriptors, functional-group indicators, ECFP4 and MACCS fingerprints,
and an explicit RDKit processing status. Each ADMET CSV has the same unique
`record_id` set as its paired Parquet and contains the TDC molecule, label
`Y`, and source identifier.

### Molecular-property Parquet schema

The `*_mol_properties.parquet` files are derived after the final filtering and
disjoin steps. Their rows preserve the order of the paired Parquet and are
joined through `record_id`; the values are calculated from
`smiles_canonical`, not from the NMR peaks or a source property table. They are
provided for analysis and structure-based baselines and are not acceptance
criteria for the cleaned dataset.

| Column | Meaning |
|---|---|
| `record_id`, `smiles_canonical` | Spectrum identifier and exact structure string used by RDKit. |
| `rdkit_status` | `ok`, `missing_smiles`, `invalid_smiles`, or `calculation_error`. |
| `rdkit_error` | Empty for `ok`; otherwise the retained processing failure. |
| `exact_molecular_weight` | RDKit exact isotopic molecular weight, in daltons. |
| `calculated_logp` | RDKit Crippen MolLogP estimate; dimensionless. |
| `tpsa` | RDKit topological polar surface area, in Å². |
| `hba`, `hbd` | RDKit hydrogen-bond acceptor and donor counts. |
| `rotatable_bonds` | RDKit rotatable-bond count. |
| `fraction_csp3` | Fraction of carbon atoms that are sp3 hybridized. |
| `aromatic_atom_fraction` | Aromatic heavy atoms divided by all heavy atoms. |
| `has_amine` | `1` when the SMARTS finds a trivalent amine N excluding amide/sulfonamide-like and imine N; otherwise `0`. |
| `has_amide` | `1` when an N–C(=O) amide pattern is present. |
| `has_alcohol_or_phenol` | `1` when an alcohol or phenol O–H pattern is present. |
| `has_ester` | `1` when a C(=O)–O–C ester pattern is present. |
| `has_carboxylic_acid` | `1` when a C(=O)–OH carboxylic-acid pattern is present. |
| `has_aldehyde_or_ketone` | `1` when an aldehyde or carbon-substituted ketone carbonyl pattern is present. |
| `has_nitrile` | `1` when a C≡N pattern is present. |
| `has_halogenated_group` | `1` when a carbon–F/Cl/Br/I bond is present. |
| `has_heteroaromatic_ring` | `1` when an aromatic N, O, S, or P atom is present. |
| `morgan_ecfp4_2048` | Radius-2, 2,048-bit Morgan/ECFP4 vector stored as 256 raw RDKit fingerprint bytes in a fixed-size binary column. |
| `maccs_keys_166_bits` | The 166 usable RDKit MACCS positions as a `0`/`1` string; RDKit's unused position 0 is omitted. |

The nine `has_*` fields are binary, potentially overlapping SMARTS indicators,
not functional-group counts. For a non-`ok` row, all descriptor, indicator,
and fingerprint cells are blank; the row itself is retained so NMR/sidecar Parquet
alignment is not lost. The exact SMARTS definitions and fingerprint encodings
are implemented in
[`calculate_mol_properties.py`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/postprocess/calculate_mol_properties.py).

### Loading paired NMR and Morgan inputs

`PairedFoundationDataset` streams an NMR Parquet together with its
molecular-property sidecar. At construction it checks equal row counts,
row-group boundaries, schema metadata, and required columns. During iteration
it checks `record_id` row by row and requires `rdkit_status == "ok"` and a
256-byte Morgan value. It reads only `record_id`, `rdkit_status`, and
`morgan_ecfp4_2048` from the sidecar; the other descriptors are not loaded for
training.

The matching row-group layout lets PyTorch workers read disjoint row groups
without loading either file in memory. `FoundationNMRProcessor` expands the
compact bytes during collation and returns `fingerprints` as
`torch.float32[batch, 2048]`. Training shuffle is performed inside the
iterable dataset with randomized row groups and a bounded record buffer; use
`shuffle=False` for deterministic evaluation order. See
[`scripts/model/README.md`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/model/README.md)
for the complete loader example and current UniMol limitation.

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
[`CanonicalRecord`, `ProtonPeak`, and `CarbonPeak`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/schema.py).
[`CanonicalParquetDataset`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/dataset.py) streams one or more
Parquet files as validated `CanonicalRecord` objects, while
[`CanonicalNMRDataset`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/dataset.py) provides the in-memory
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
from NMRexp, NMRTrans, NMR-Solver, and NMRGym carry the common shift field.

### Missing values

- The two modality columns are always lists. `[]` means that the record has no
  usable peaks for that nucleus.
- In the rich MST-NMR, NMRexp, and NMRTrans files, `j_values` is always a
  list. `[]` means that no numerical coupling is listed for that peak.
- Shift-only sources use `null` for annotations outside their source
  representation, including multiplicity, J values, and reported ranges.
- A numerical `J=0` remains a supplied value and can be masked during model
  preparation.
- After common cleaning, all 16,733,130 proton peaks in the rich files retain
  populated positive integration, canonical multiplicity, and all three range
  fields.

Example streaming access:

```python
from data.dataset import CanonicalParquetDataset

records = CanonicalParquetDataset("datasets/cleaned/rich.parquet")

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
[`convert_mst_nmr.py`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/canonicalize/convert_mst_nmr.py).

### NMRexp through NMRPeak

NMRexp contains experimental spectra mined from literature and processed by
NMRPeak. Its proton mapping is the same as MST-NMR, including point intervals
for single reported shifts. Carbon records supply shifts only. Frequency and
solvent are retained where they occur in the processed release.

NMRexp is the source with partial modality coverage: an experimental report may
contain only `1H` or only `13C`. Those records remain useful and use an
empty list for the unavailable modality.

Converter:
[`convert_nmrexp.py`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/canonicalize/convert_nmrexp.py).

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
[`convert_nmrtrans.py`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/canonicalize/convert_nmrtrans.py).

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
[`convert_nmrsolver.py`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/canonicalize/convert_nmrsolver.py).

### NMRGym

NMRGym is an experimental shift-only benchmark released as train, validation,
and test pickle files. Conversion preserves its paired `1H` and `13C` shift
lists and concatenates the published partitions in their original order.
Unlike SimNMR-PubChem, it does not supply proton equivalence groups or
integration. Integration, multiplicity, J coupling, reported ranges, solvent,
and frequency therefore remain null rather than being inferred. An absent
modality would remain `[]`; after common filtering every released NMRGym row
has both modalities.

This component is distributed separately from the rich training pool so it can
support experimental shift-only pretraining, domain adaptation, or explicit
source-aware sampling.

Converter:
[`convert_nmrgym.py`](https://github.com/nicogreeco/nmr_fomo/blob/main/scripts/data/canonicalize/convert_nmrgym.py).

## Processing and split rationale

The release is produced by the explicit DVC graph in the project repository:

```text
public raw releases
    → source-specific schema-v2 canonicalisation
    → rich train/validation and source-test merges
    → move SimNMR-overlapping rich test spectra into train/validation
    → remove residual benchmark overlap with extended train and NMRGym
    → common filtering and exact-shift deduplication
    → ADMET matching and removal from rich pretraining
    → aligned molecular properties and final analytics
```

All NMR-to-NMR comparisons use exact `smiles_canonical` equality. This keeps
stereochemical distinctions represented by the canonical isomeric SMILES.
ADMET structures instead use full RDKit InChIKeys because they originate from
an external property collection.

### 1. Canonicalisation and rich merges

The five converters preserve source records in the common model-independent
schema, recalculate molecular metadata with RDKit, validate records, and retain
source provenance. The MST-NMR, NMRexp, and NMRTrans train/validation splits
produce 1,875,334 rich records; their source test splits produce 208,431.

### 2. SimNMR overlap transfer

The benchmark is compared with canonical SimNMR-PubChem. The 112,197 rich test
records representing 111,551 molecules also present in SimNMR are removed from
test and appended to rich train/validation. This protects evaluation from the
large shift-only pretraining pool without discarding the richer spectra. The
extended rich train/validation pool therefore contains 1,987,531 records, and
96,234 records remain in the benchmark candidate pool.

### 3. Residual benchmark disjoin

The residual benchmark is compared with the extended rich train/validation
pool and canonical NMRGym. This removes 7,901 records: 7,838 newly matched
through the extended train pool and 63 through NMRGym. The resulting 88,333
records enter common filtering. The comparison script accepts one benchmark
and any number of comparison datasets, while only removing rows from the
benchmark.

### 4. Common quality filtering and deduplication

The same filter is applied to rich train/validation, benchmark, SimNMR, and
NMRGym. A record is retained when at least one modality contains peaks and:

- shifts and J values are finite and within the documented physical ranges;
- each modality has at most 60 peaks and each proton peak at most six J values;
- supplied proton integrations are positive;
- the canonical structure is a single connected fragment.

Deduplication removes only identical canonical structure plus exact sorted
`1H` and `13C` shift lists; no rounding or tolerance is used. Different spectra
of the same molecule remain separate. Filtering removes 34,330 records from
the extended rich pool and 436 from the benchmark, leaving 1,953,201 and
87,897 records respectively before ADMET removal.

### 5. ADMET preparation

Ames, LD50 Zhu, AqSolDB solubility, and AstraZeneca lipophilicity
retain their official TDC split assignments. Sangster logP uses the 13,812
high-confidence `P` molecules and a deterministic full-InChIKey 80/20 split. Repeated property identities are consolidated; discordant labels
are excluded, and the single Ames identity occurring across official splits is
removed from both splits. Full RDKit InChIKeys match the external structures to
rich spectra. The union contains 8,169 matched molecules and removes 11,354 NMR
records from rich pretraining, leaving 1,941,847 final `rich` records.

ADMET spectra are selected from the filtered rich train/validation pool, not
from `test_benchmark.parquet`. This choice maximizes matches between NMR records
and experimental property labels as restricting matching to the smaller benchmark
test pool yielded very small ADMET cohorts. Using the larger rich pool retains
more labelled records for downstream evaluation.

The rich pool combines the released training and validation splits of NMRPeak
and NMRTrans, so the ADMET cohorts inherit records and molecules from those
encoders' training data. Prior spectral exposure is therefore expected by
construction.

The exclusion applies only to `rich.parquet`:
SimNMR and NMRGym are not explicitly decontaminated against ADMET, and the extent
of exposure across all external encoders has not been fully audited. Therefore
these cohorts do not guarantee unseen molecules across all pretraining sources,
including those used by FoMoNMR.

### 6. Molecular properties, reports, and analytics

Molecular-property Parquets are calculated only after the final split, filtering,
and ADMET operations. Every sidecar has the same row count, order, and
`record_id` sequence as its paired Parquet. Compact JSON processing reports
record stage inputs, outputs, counts, and useful reason breakdowns. The full
pipeline, parameters, dependencies, and output hashes are tracked by
`dvc.yaml`, `params.yaml`, and `dvc.lock` in the project repository.

## Dataset-specific composition

| Dataset | Source | Records | Unique canonical SMILES | With `1H` | With `13C` | With both |
|---|---|---:|---:|---:|---:|---:|
| Rich | MST-NMR | 771,255 | 771,199 | 771,255 | 771,255 | 771,255 |
| Rich | NMRexp | 993,568 | 993,568 | 868,591 | 847,898 | 722,921 |
| Rich | NMRTrans / NMRSpec | 177,024 | 177,024 | 177,024 | 177,024 | 177,024 |
| **Rich** | **All** | **1,941,847** | **1,869,511** | **1,816,870** | **1,796,177** | **1,671,200** |
| Benchmark test | MST-NMR | 3,717 | 3,717 | 3,717 | 3,717 | 3,717 |
| Benchmark test | NMRexp | 74,209 | 74,209 | 64,866 | 63,446 | 54,103 |
| Benchmark test | NMRTrans / NMRSpec | 9,971 | 9,971 | 9,971 | 9,971 | 9,971 |
| **Benchmark test** | **All** | **87,897** | **87,631** | **78,554** | **77,134** | **67,791** |

NMRexp accounts for the single-modality rich records. MST-NMR and NMRTrans
remain paired in both files.

### ADMET property subsets

| Endpoint | Split | Records | Unique property SMILES |
|---|---|---:|---:|
| Ames | Train/validation | 2,527 | 1,748 |
| Ames | Test | 455 | 334 |
| LD50 Zhu | Train/validation | 2,748 | 1,964 |
| LD50 Zhu | Test | 574 | 427 |
| AqSolDB solubility | Train/validation | 3,777 | 2,671 |
| AqSolDB solubility | Test | 770 | 567 |
| AstraZeneca lipophilicity | Train/validation | 865 | 678 |
| AstraZeneca lipophilicity | Test | 182 | 153 |
| Sangster logP | Train/validation | 4,565 | 3,091 |
| Sangster logP | Test | 1,153 | 773 |

A molecule can have several distinct NMR records, so spectrum-record counts can
exceed unique property structures. Target definitions and units follow the TDC
releases or, for Sangster, the pinned Zenodo workbook.

### SimNMR-PubChem shift-only component

The canonical source contains 105,764,812 simulated records. Common filtering
removes 255,196, leaving 105,509,616 records and approximately 97,936,772
unique canonical SMILES. Both modalities are non-empty in 105,476,128 records.

### NMRGym experimental shift-only component

The canonical source contains 269,999 records. Common filtering removes 4,904,
leaving 265,095 records and exactly 265,095 unique canonical SMILES. Every
retained record has both modalities.

## Analytics

Detailed descriptive statistics, source comparisons, figures, annotation
coverage, NMRGym and SimNMR shift-only analyses, and links to every generated
CSV are collected in the
[analytics report](ANALYTICS.md). The report links every generated table and
figure and documents how to reproduce the analysis from the project repository.

## Use considerations

The primary learning unit is a structured resonance-level peak list linked to
a molecular structure. The rich collection supports models that consume
chemical shifts together with integration, multiplicity, ranges, and J
couplings. SimNMR-PubChem supports simulated shift-set pretraining, while
NMRGym supplies experimental shift-only examples for domain support and
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

The repository documents the reusable data API, canonical converters, and
postprocessing commands in
[`scripts/data`](https://github.com/nicogreeco/nmr_fomo/tree/main/scripts/data).
Generated source, schema, filtering, and property summaries are indexed in
[`ANALYTICS.md`](ANALYTICS.md).
