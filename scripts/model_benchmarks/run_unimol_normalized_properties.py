"""Compare train-centered, L2-normalized UniMol2 with existing linear probes.

Reuses the benchmark cohorts, parameter grids and grouped folds. Centering is
fitted inside each CV fold; exported embeddings use the complete benchmark
training cohort's center. The held-out test never contributes to that center.
"""

import argparse
import hashlib
from pathlib import Path

from model_benchmarks import run_property_prediction as baseline

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline


COHORT_MODELS = ["morgan", "unimol2", "nmrpeak", "nmrtrans", "ultranmr", "nmrsolver"]
DATASETS = ["solubility_aqsoldb", "ld50_zhu", "lipophilicity_astrazeneca", "sangster_logp", "ames"]


class CenterL2(TransformerMixin, BaseEstimator):
    def fit(self, features, targets=None):
        self.center_ = np.asarray(features, dtype=np.float64).mean(axis=0)
        return self

    def transform(self, features):
        centered = np.asarray(features, dtype=np.float64) - self.center_
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        if not np.isfinite(centered).all() or (norms == 0).any():
            raise ValueError("Cannot normalize zero or non-finite centered embeddings")
        return centered / norms


def export_embeddings(dataset, root, transform, common_train):
    output = root / dataset / "unimol2_normalized"
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "train_center.npy", transform.center_)
    for split in ("train", "test"):
        source = root / dataset / "unimol2" / f"{split}.parquet"
        table = pq.read_table(source)
        values = np.asarray(table.column("embedding").to_pylist(), dtype=np.float64)
        normalized = transform.transform(values).astype(np.float32)
        column = pa.FixedSizeListArray.from_arrays(pa.array(normalized.ravel()), normalized.shape[1])
        table = table.set_column(table.schema.get_field_index("embedding"), "embedding", column)
        metadata = dict(table.schema.metadata or {})
        metadata.update({
            b"representation": b"unimol2_normalized",
            b"transform": b"benchmark_train_center_then_l2",
            b"center_sha256": hashlib.sha256(transform.center_.tobytes()).hexdigest().encode(),
            b"center_record_ids_sha256": hashlib.sha256("\n".join(sorted(common_train)).encode()).hexdigest().encode(),
            b"center_record_count": str(len(common_train)).encode(),
        })
        table = table.replace_schema_metadata(metadata)
        partial = output / f"{split}.partial"
        pq.write_table(table, partial, compression="zstd")
        partial.replace(output / f"{split}.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings/admet"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets/cleaned/admet"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/property_prediction"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/property_prediction_unimol2_normalized"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for dataset in baseline.parse_names(args.datasets):
        common = {split: baseline.common_record_ids(dataset, COHORT_MODELS, split, args.embeddings_dir)
                  for split in ("train", "test")}
        train_x, train_y, groups = baseline.load_split(dataset, "unimol2", "train", common["train"], args.embeddings_dir, args.datasets_dir)
        test_x, test_y, test_groups = baseline.load_split(dataset, "unimol2", "test", common["test"], args.embeddings_dir, args.datasets_dir)
        old = pl.read_csv(args.baseline_dir / f"{dataset}.csv").filter(pl.col("probe") == "linear")
        if any(old["n_train"] != len(train_y)) or any(old["n_test"] != len(test_y)):
            raise ValueError(f"{dataset}: cohort counts differ from existing benchmarks")
        classification = dataset in baseline.CLASSIFICATION_DATASETS
        if classification:
            estimator = baseline.LogisticRegression(max_iter=8000)
            parameter = "logisticregression__C"
            values = baseline.LOGISTIC_CS
            cv = baseline.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "roc_auc"
        else:
            estimator = baseline.Ridge()
            parameter = "ridge__alpha"
            values = baseline.RIDGE_ALPHAS
            cv = baseline.GroupKFold(n_splits=5)
            scoring = "r2"
        search = baseline.GridSearchCV(
            make_pipeline(CenterL2(), baseline.VarianceThreshold(1e-6), baseline.StandardScaler(), estimator),
            {parameter: values}, scoring=scoring, cv=cv, n_jobs=1,
        )
        print(f"{dataset}: fitting normalized linear probe ({len(train_y)}/{len(test_y)} train/test)", flush=True)
        search.fit(train_x, train_y, groups=groups)
        predictions = search.predict(test_x)
        if classification:
            scores = search.predict_proba(test_x)[:, 1]
            metrics = baseline.classification_metrics(test_y, predictions, scores)
        else:
            scores = predictions
            metrics = baseline.regression_metrics(test_y, predictions)
        transform = search.best_estimator_.steps[0][1]
        export_embeddings(dataset, args.embeddings_dir, transform, common["train"])
        # Check that the exported representation gives the same final predictions.
        saved_x, saved_y, _ = baseline.load_split(dataset, "unimol2_normalized", "test", common["test"], args.embeddings_dir, args.datasets_dir)
        tail = Pipeline(search.best_estimator_.steps[1:])
        saved_scores = tail.predict_proba(saved_x)[:, 1] if classification else tail.predict(saved_x)
        np.testing.assert_array_equal(saved_y, test_y)
        np.testing.assert_allclose(saved_scores, scores, atol=1e-4, rtol=1e-4)
        value = float(search.best_params_[parameter])
        row = {
            "dataset": dataset, "task": "classification" if classification else "regression",
            "representation": "unimol2_normalized", "probe": "linear",
            "n_train": len(train_y), "n_test": len(test_y),
            "n_train_molecules": len(np.unique(groups)), "n_test_molecules": len(np.unique(test_groups)),
            "alpha": None if classification else value, "C": value if classification else None,
            "cv_score": float(search.best_score_), **metrics,
        }
        pl.DataFrame([row]).write_csv(args.output_dir / f"{dataset}.csv")
        # Save predictions for future paired comparisons, without retraining baselines.
        pl.DataFrame({"molecule_group": test_groups, "target": test_y, "prediction": predictions, "score": scores}).write_csv(args.output_dir / f"{dataset}_predictions.csv")
        comparisons.extend(old.to_dicts())
        comparisons.append(row)
        pl.DataFrame(comparisons, strict=False).write_csv(args.output_dir / "comparison.csv")
        original = old.filter(pl.col("representation") == "unimol2").row(0, named=True)
        print(f"{dataset}: {scoring} original={original[scoring]:.6f}, normalized={metrics[scoring]:.6f}, delta={metrics[scoring]-original[scoring]:+.6f}", flush=True)


if __name__ == "__main__":
    main()
