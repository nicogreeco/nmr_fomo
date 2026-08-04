# Repository Agent Instructions

## Project Purpose

- The main project builds canonical NMR datasets and develops an NMR foundation model using combined `1H + 13C` spectra.
- Extracting embeddings from published models is a secondary benchmarking task, not the architecture of the new foundation model.

## Documentation Guide

- `README.md` gives the repository overview and published-model boundaries.
- `contex/Internship – NMR Project Plan.md` defines project goals, phases, and deliverables.
- `contex/NMR foundations and AI.md` covers NMR fundamentals and the broader AI research context.
- `contex/Datasets.md` explains the dataset landscape, schema rationale,
  multiplicity harmonization, canonical pipeline, and training strategy.
- `contex/Canonicalization_Implementation_Notes.md` is the authority for the
  implemented schema, conversions, ranges, missing values, provenance, and IDs.
- `contex/Multiplicity analysis.md` records vocabulary evidence, lossless aliases,
  and measured coverage.
- `contex/Dataset_Filtering_and_Processing.md` distinguishes source curation,
  benchmark selection, model preparation, and canonical conversion.
- `contex/Dataset Analysis.md` records measured dataset properties and processor
  compatibility.
- `contex/Embedding_Pipeline_Architecture.md` defines component responsibilities,
  embedding flow, extraction behavior, and current limitations.
- `contex/Baseline NMR Encoders.md` explains benchmark models, inputs, pooling,
  and comparison choices.
- `contex/Properties Dataset.md` covers downstream property benchmarks and
  leakage-safe evaluation.
- `contex/NMR/DL Methods/` contains model- and dataset-specific notes, paper summaries, implementation observations, and links to relevant work.

## Repository Navigation

- Before changing canonical schema or shared data code, read `contex/Datasets.md`,
  `contex/Canonicalization_Implementation_Notes.md`, and
  `contex/Multiplicity analysis.md`.
- Before converting a dataset, also read
  `contex/Dataset_Filtering_and_Processing.md`, `contex/Dataset Analysis.md`, and
  `scripts/canonicalize/README.md`.
- Before changing model benchmarks, read `contex/Embedding_Pipeline_Architecture.md`,
  `contex/Baseline NMR Encoders.md`, and the relevant note under
  `contex/NMR/DL Methods/`.
- Before changing embedding extraction, also read the selected model's upstream
  README under `models/`.
- Put reusable schema, validation, and dataset code in `scripts/data/`.
- Put source-specific conversion code in `scripts/canonicalize/`.
- Put published-model processors and embedding wrappers in `scripts/model_benchmarks/`.

## Architecture and Boundaries

- Canonical data code must remain model-independent and reusable for future foundation-model training.
- Keep the intended flow:

  `canonical dataset -> model-specific processor/collator -> model-specific batch -> lightweight embedder or trainable model`

- Do not place reusable canonical dataset logic inside the benchmarking package.
- Processors own validation, model-native conversion, padding, and collation.
  Embedders own checkpoint loading, encoder execution, and pooling.
- Reuse official preprocessing, checkpoint-loading, and model code where practical.
- Keep model-family imports lazy. Different families may require separate Python environments and should run one family per process.
- The current published-model benchmark scope requires combined `1H + 13C` input.
- Follow the canonicalization notes rather than restating or guessing schema rules.
  Preserve missing values versus empty collections and raw versus normalized
  annotations. Do not silently invent missing NMR information.
- Report assumptions, incompatible records, and unsupported cases clearly. Do not
  hide them with implicit clipping, filtering, coercion, or lossy mappings.
- Do not process, rewrite, move, merge, filter, or convert real datasets unless
  explicitly requested.
- Do not download data, modify large checkpoints, or edit cloned repositories under `models/` unless explicitly required.

## Code Style

- Write clear beginner-to-intermediate research Python that a strong university student can read and modify.
- Prefer simple functions, small classes, explicit control flow, descriptive names,
  limited type hints, and short comments for non-obvious model-specific logic.
- A little duplication is acceptable when it makes model paths easier to follow.
- Keep useful checks and clear errors with the minimum required complexity.
- Avoid deep inheritance, advanced generics or protocols, decorators,
  metaprogramming, excessive factories or registries, configuration layers,
  one-use abstractions, and clever compact code.
- Do not design a general framework for hypothetical future requirements.

## Tests

- Run tests from the repository root with `scripts` on `PYTHONPATH` and use the
  environment appropriate to the model family.

```bash
PYTHONPATH=scripts python -m unittest discover -s scripts/canonicalize/tests -v
PYTHONPATH=scripts python -m unittest discover -s scripts/model_benchmarks/tests -v
PYTHONPATH=scripts python -m unittest scripts.model_benchmarks.tests.test_nmrtrans_mask -v
```

- Keep testing lightweight and focused on canonical dataset loading, multiplicity
  normalization, model processors, checkpoint loading and embedding extraction,
  and the NMRTrans padding/PMA diagnostic.
- Tests must use small fixtures and must not scan or rewrite real datasets.
- Model dependency or checkpoint tests may skip when the required local environment
  or asset is unavailable. Do not require exhaustive tests for small helpers unless
  the task specifically needs them.
