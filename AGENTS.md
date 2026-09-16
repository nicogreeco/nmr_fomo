# Repository Agent Instructions

## Project Purpose

- The project builds canonical NMR datasets and trains **FoMoNMR**, a foundation
  encoder for structured, combined `1H + 13C` resonance lists.
- Published-model embedding extraction is a separate benchmarking path, not the
  architecture of FoMoNMR.
- Keep the core separation explicit:

  ```text
  canonical dataset -> model-specific processor/collator -> native batch -> encoder/embedding
  ```

## Current Data and Training Invariants

- Canonical schema v2 is model independent. It must not gain tokenization,
  padding, checkpoint-specific limits, inferred NMR attributes, or training-only
  fields.
- Preserve padding, missing source data, and deliberately masked training data
  as distinct states. Preserve `multiplicity_raw` alongside normalized
  `multiplicity`; `null` is unavailable data and `<unk>` is an observed but
  unrepresentable annotation.
- Canonicalization is conservative: one deterministic `record_id` per source
  record, source SMILES retained, RDKit-derived canonical SMILES/formula/atoms,
  and no invented splits, coordinates, or equivalence classes. SimNMR atom
  shifts are grouped only by its supplied equivalence classes, never shift
  proximity.
- NMR-to-NMR split and overlap operations use exact `smiles_canonical`.
  External property matching uses full RDKit InChIKey, retaining stereochemical
  and molecular-form distinctions.
- Derived filtering, exact-shift deduplication, analytics, molecule-property
  sidecars, and physical train/validation files must not alter source canonical
  Parquets. Sidecars must remain same-order and `record_id`-aligned with their
  paired NMR Parquet.
- `dvc.lock` and materialized stage reports are authoritative for current
  hashes and counts; documentation numbers are explanatory snapshots.

## FoMoNMR Boundary

- FoMoNMR consumes canonical peak sets, not FIDs or dense traces. One token is
  one resonance/multiplet. It uses separate H/C adapters, a shared
  permutation-equivariant Transformer, and masked mean pooling.
- Its foundation-training implementation belongs in `scripts/model/`; reusable
  schema and canonical loading remain in `scripts/data/`.
- Rich proton attributes are optional, gated features. Do not replace missing
  fields with zero-valued measurements or bypass their availability masks.
- The maintained curriculum is shift-only-heavy pretraining followed by
  rich-focused continued pretraining with explicit SimNMR/NMRGym replay.
  `max_steps`, not epochs, is the authoritative streaming-training budget.
- Molecular fingerprints and UniMol2 teacher vectors are training targets or
  comparisons only when the relevant recipe documents them. Do not present a
  structure-aware or combined input as spectrum-only.

## Published-Model Benchmark Boundary

- `scripts/model_benchmarks/` owns adapters, collators, checkpoint wrappers,
  pooling, and extraction for published encoders. Keep imports lazy and run one
  model family per process in its required environment.
- Current NMR baselines are NMRPeak-R, NMRTrans, UltraNMR, and the fixed
  NMR-Solver control. UniMol2 and Morgan are structure-only comparisons;
  DiffNMR is documented but not implemented in this extraction path.
- Preserve each checkpoint's native input and pooling rule. In particular, use
  masked local-state means for NMRTrans rather than its padding-sensitive PMA
  output. The benchmark requires non-empty paired `1H + 13C` inputs for all
  four NMR paths.
- Across a comparison, hold molecules, labels, splits, downstream heads, and
  metrics fixed. Report molecular, NMR-only, and combined representations as
  distinct conditions.

## Documentation to Read Before Changes

- **DVC/data DAG:** `README.md`, `datasets/README.md`, `scripts/data/README.md`,
  `dvc.yaml`, and `params.yaml`; regenerate stage definitions with
  `scripts/data/generate_dvc_pipeline.sh`.
- **Canonical schema/converters:** `contex/Datasets.md`,
  `contex/Canonicalization_Implementation_Notes.md`,
  `contex/Multiplicity analysis.md`, and the relevant converter README.
- **Filtering, analysis, or ADMET:**
  `contex/Dataset_Filtering_and_Processing.md`, `contex/Dataset Analysis.md`,
  `contex/Properties Dataset.md`, and `contex/Property Prediction.md`.
- **FoMoNMR architecture or training:** `contex/FoMoNMR_Model_Architecture.md`,
  `contex/FoMoNMR Ablation Studies.md`, `scripts/model/README.md`, and the
  relevant training configuration.
- **Published-model extraction:** `contex/Embedding_Pipeline_Architecture.md`,
  `contex/Baseline NMR Encoders.md`, the relevant note in
  `contex/NMR/DL Methods/`, and the selected upstream model README under
  `models/`.

## Implementation Boundaries

- Put reusable schema, validation, and dataset code in `scripts/data/`.
- Put source-specific canonical conversion code in `scripts/data/canonicalize/`.
- Put derived-data and benchmark-preparation utilities in
  `scripts/data/postprocess/`.
- Put FoMoNMR datasets, processor, model, objectives, and training code in
  `scripts/model/`.
- Put published-model processors and embedders in
  `scripts/model_benchmarks/`. A processor must never parse a raw source.
- Reuse official preprocessing and checkpoint loading where practical. Report
  assumptions, incompatibilities, and unsupported cases rather than silently
  clipping, coercing, filtering, or applying lossy mappings.
- Do not process, rewrite, move, merge, filter, or convert real datasets unless
  explicitly requested. Do not download data, modify large checkpoints, or edit
  cloned repositories under `models/` unless explicitly required.

## Code Style

- Write clear beginner-to-intermediate research Python: small functions,
  explicit control flow, descriptive names, limited type hints, and concise
  comments for model-specific details.
- Prefer readability to abstractions. Avoid deep inheritance, metaprogramming,
  broad registries/config layers, and speculative frameworks.

## Tests

- Run tests from the repository root with `scripts` on `PYTHONPATH` in the
  appropriate environment (`nmr-env main` for data and foundation-model work;
  model-specific environments only when required):

  ```bash
  PYTHONPATH=scripts python -m unittest discover -s scripts/data/canonicalize/tests -v
  PYTHONPATH=scripts python -m unittest discover -s scripts/data/postprocess/tests -v
  PYTHONPATH=scripts python -m unittest discover -s scripts/model_benchmarks/tests -v
  PYTHONPATH=scripts python -m unittest scripts.model_benchmarks.tests.test_nmrtrans_mask -v
  ```

- Keep tests small and fixture-based. They must not scan or rewrite real
  datasets. Dependency- or checkpoint-specific tests may skip when the needed
  local environment or asset is unavailable.
