# Embedders

Each file is a thin interface to one official model implementation. An
embedder loads the relevant weights, runs the spectrum encoder, applies the
documented pooling step, and returns embeddings with ordered record IDs.

The representations are fixed for this project:

| Model | Size | Representation |
|---|---:|---|
| NMRPeak-R | 768 | Final BART encoder BOS state, L2-normalized |
| NMRTrans | 1024 | Concatenated validity-masked means of final H and C local states |
| UltraNMR | 768 | Official embedding/Transformer path with validity-masked mean pooling |
| NMR-Solver | 256 | Official combined H+C Gaussian `set2vec` fixed featurizer |

NMR-Solver is not a learned encoder. NMRTrans PMA is diagnostic only because
the released pooled result changes with padding; the model's mask behavior is
not corrected in this project.

Keep repository imports inside the model-specific path so the shared packages
remain importable in every model environment.

Embedders consume model-native batches only. DVC dataset restoration and
raw-to-cleaned processing remain responsibilities of `scripts/data/`.
