# Smoke tests

These tests provide quick checks for the code paths most likely to break during
research changes:

- canonical Parquet loading;
- one small processor batch for each model;
- embedding shape and record-ID order;
- real checkpoint loading where a local checkpoint is available;
- the observed NMRTrans padding and PMA behavior.

Fixtures are small hand-written canonical records. The tests do not scan or
rewrite the real datasets.

Run shared tests from the project root:

```bash
PYTHONPATH=scripts python -m unittest discover -s scripts/model_benchmarks/tests -v
```

Checkpoint tests must be run in the corresponding model environment. A test is
skipped when its model dependency or local checkpoint is unavailable.
