# Processors

Each file contains one callable processor used as a PyTorch `collate_fn`.
A processor keeps the complete model-specific preparation path together:

```text
canonical records
    -> required-field validation
    -> model-native values
    -> token or tensor preparation
    -> padding and batch collation
```

Processors return record IDs in batch order and raise a clear compatibility
error when required information is absent. The extraction command can catch
these errors and skip incompatible records while writing a rejection report.

`mode="canonical"` uses normalized multiplicity. `mode="native"` uses the raw
annotation where useful for NMRPeak and NMRTrans. The setting is a documented
no-op for UltraNMR and NMR-Solver.

Edit the named model file when its required fields or official preprocessing
path changes. Do not put source-dataset parsing, checkpoint loading, or pooling
logic here.

Processors read canonical records produced by the DVC data pipeline but are not
stages in that pipeline; they cannot alter split or filtering decisions.
