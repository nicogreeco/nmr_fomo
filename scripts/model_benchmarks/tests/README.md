# Smoke tests

These tests provide quick checks for the code paths most likely to break during
research changes:

- canonical Parquet loading;
- one small processor batch for each model;
- UniMol2 structure preprocessing, model-size selection, and CLI defaults;
- embedding shape and record-ID order;
- real checkpoint loading where a local checkpoint is available;
- the observed NMRTrans padding and PMA behavior.

They do not require `dvc pull` or a materialized `datasets/cleaned/` directory.

Only explicitly marked checkpoint tests may require local model assets.

Fixtures are small hand-written canonical records. The tests do not scan or
rewrite the real datasets.

Run shared tests from the project root:

```bash
PYTHONPATH=scripts python -m unittest discover -s scripts/model_benchmarks/tests -v
```

Checkpoint tests must be run in the corresponding model environment. A test is
skipped when its model dependency or local checkpoint is unavailable.

The UniMol2 checkpoint smoke test is opt-in and never downloads weights:

```bash
NMR_BENCHMARK_RUN_UNIMOL2_SMOKE=1 \
NMR_BENCHMARK_UNIMOL2_CHECKPOINT=/path/to/checkpoint.pt \
PYTHONPATH=scripts python -m unittest \
  scripts.model_benchmarks.tests.test_embedders -v
```
