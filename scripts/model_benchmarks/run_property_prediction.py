#!/usr/bin/env python3
"""Run the notebook property-prediction probes on saved embeddings."""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rdkit import Chem
from torch import nn

# cuml.accel must be enabled before importing scikit-learn estimators.
try:
    import cuml
except ModuleNotFoundError:
    CUML_ACCEL_ENABLED = False
    N_JOBS = -1
else:
    cuml.accel.install()
    CUML_ACCEL_ENABLED = True
    N_JOBS = 1

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHAS = np.logspace(-1, 5, 20)
MLP_REGRESSION_ALPHAS = np.logspace(-3, 5, 25)
LOGISTIC_CS = np.logspace(-4, 4, 12)
MLP_CLASSIFICATION_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1]
CLASSIFICATION_DATASETS = {"ames"}
CANONICAL_FILES = {"train": "train_val.parquet", "test": "test.parquet"}
TARGET_FILES = {"train": "train_val.csv", "test": "test.csv"}
MOLECULE_GROUP_CACHE = {}


def split_mlp_train_validation(targets, groups, classification, fraction, seed):
    """Keep every spectrum of a molecule on one side of the internal holdout."""

    if groups is None:
        raise ValueError("Molecular groups are required for MLP validation")
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(targets):
        raise ValueError("Provide one molecular group per MLP training record")
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("MLP validation requires at least two molecular groups")

    if classification:
        # As in fine-tuning, one stratified group fold approximates the fraction.
        splitter = StratifiedGroupKFold(
            n_splits=min(max(2, round(1 / fraction)), n_groups),
            shuffle=True,
            random_state=seed,
        )
    else:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=fraction, random_state=seed,
        )
    return next(splitter.split(np.zeros(len(targets)), targets, groups))


class TorchMLP(BaseEstimator):
    """Small sklearn-compatible MLP trained with PyTorch on GPU."""

    classification = False

    def __init__(
        self,
        alpha=1e-4,
        hidden_size=256,
        batch_size=64,
        learning_rate=1e-3,
        max_iter=300,
        validation_fraction=0.1,
        random_state=42,
        device="auto",
    ):
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device

    def fit(self, features, targets, groups=None):
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        train_indices, validation_indices = split_mlp_train_validation(
            targets, groups, self.classification,
            self.validation_fraction, self.random_state,
        )
        train_x, validation_x = features[train_indices], features[validation_indices]
        train_y, validation_y = targets[train_indices], targets[validation_indices]

        use_cuda = self.device != "cpu" and torch.cuda.is_available()
        if self.device == "cuda" and not use_cuda:
            raise RuntimeError("CUDA requested for the MLP, but it is unavailable")
        self.device_ = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(self.random_state)
        self.model_ = nn.Sequential(
            nn.Linear(features.shape[1], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        ).to(self.device_)
        train_x = torch.from_numpy(train_x).to(self.device_)
        train_y = torch.from_numpy(train_y).to(self.device_)
        validation_x = torch.from_numpy(validation_x).to(self.device_)
        validation_y = torch.from_numpy(validation_y).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_function = nn.BCEWithLogitsLoss() if self.classification else nn.MSELoss()
        generator = torch.Generator(device=self.device_).manual_seed(self.random_state)
        best_score = -np.inf
        best_state = None
        epochs_without_improvement = 0

        for iteration in range(self.max_iter):
            self.model_.train()
            indices = torch.randperm(len(train_y), generator=generator, device=self.device_)
            for start in range(0, len(train_y), self.batch_size):
                batch = indices[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_x[batch]).squeeze(1)
                penalty = sum(
                    parameter.square().sum()
                    for parameter in self.model_.parameters()
                    if parameter.ndim > 1
                )
                loss = loss_function(outputs, train_y[batch])
                loss = loss + self.alpha * penalty / (2 * len(train_y))
                loss.backward()
                optimizer.step()

            predictions = self._predict_tensor(validation_x)
            if self.classification:
                score = accuracy_score(validation_y.cpu(), predictions >= 0.5)
            else:
                score = r2_score(validation_y.cpu(), predictions)
            if score > best_score + 1e-4:
                best_score = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model_.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    break

        self.model_.load_state_dict(best_state)
        self.n_iter_ = iteration + 1
        return self

    def _predict_tensor(self, features):
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(features).squeeze(1)
            if self.classification:
                outputs = outputs.sigmoid()
        return outputs.cpu().numpy()

    def predict(self, features):
        features = torch.as_tensor(
            np.asarray(features, dtype=np.float32), device=self.device_
        )
        scores = self._predict_tensor(features)
        return (scores >= 0.5).astype(np.int64) if self.classification else scores


class TorchMLPRegressor(RegressorMixin, TorchMLP):
    pass


class TorchMLPClassifier(ClassifierMixin, TorchMLP):
    classification = True

    def fit(self, features, targets, groups=None):
        self.classes_ = np.asarray([0, 1])
        return super().fit(features, targets, groups=groups)

    def predict_proba(self, features):
        features = torch.as_tensor(
            np.asarray(features, dtype=np.float32), device=self.device_
        )
        scores = self._predict_tensor(features)
        return np.column_stack((1 - scores, scores))


def parse_names(values):
    """Accept names separated by spaces, commas, or both."""

    return [
        name.strip()
        for value in values
        for name in value.split(",")
        if name.strip()
    ]


def molecule_group(smiles):
    """Use the same full-InChIKey identity as ADMET preparation."""

    if smiles not in MOLECULE_GROUP_CACHE:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid canonical SMILES: {smiles!r}")
        MOLECULE_GROUP_CACHE[smiles] = Chem.MolToInchiKey(molecule)
    return MOLECULE_GROUP_CACHE[smiles]


def common_record_ids(dataset, representations, split, embeddings_root):
    """Keep every representation on the same records, as in the notebook."""

    models = sorted({model for name in representations for model in name.split("+")})
    record_sets = []
    for model in models:
        path = embeddings_root / dataset / model / f"{split}.parquet"
        record_sets.append(set(pl.read_parquet(path, columns=["record_id"])["record_id"]))
    return set.intersection(*record_sets)


def load_split(
    dataset,
    representation,
    split,
    common_records,
    embeddings_root,
    datasets_root,
):
    canonical_path = datasets_root / dataset / CANONICAL_FILES[split]
    target_path = datasets_root / dataset / TARGET_FILES[split]
    table = pl.read_parquet(
        canonical_path,
        columns=["record_id", "smiles_canonical"],
    ).join(pl.read_csv(target_path), on="record_id", how="inner")

    embedding_columns = []
    for model in representation.split("+"):
        column = f"{model}_embedding"
        embedding_columns.append(column)
        embeddings = pl.read_parquet(
            embeddings_root / dataset / model / f"{split}.parquet"
        ).select("record_id", pl.col("embedding").alias(column))
        table = table.join(embeddings, on="record_id", how="inner")

    table = table.filter(pl.col("record_id").is_in(common_records)).sort("record_id")
    if table.is_empty():
        raise ValueError(f"No common {split} records for {dataset}/{representation}")

    matrices = [
        np.stack(table[column].to_list()).astype(np.float64)
        for column in embedding_columns
    ]
    features = np.concatenate(matrices, axis=1)
    targets = np.asarray(table["Y"].to_numpy(), dtype=np.float64)
    groups = np.asarray([molecule_group(smiles) for smiles in table["smiles_canonical"]])
    return features, targets, groups


def regression_metrics(targets, predictions):
    return {
        "mae": mean_absolute_error(targets, predictions),
        "rmse": np.sqrt(mean_squared_error(targets, predictions)),
        "r2": r2_score(targets, predictions),
    }


def classification_metrics(targets, predictions, scores):
    return {
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "roc_auc": roc_auc_score(targets, scores),
        "average_precision": average_precision_score(targets, scores),
    }


def run_regression(
    kind, train_x, train_y, test_x, test_y, groups, folds, max_iter, device="auto"
):
    cv = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    if kind == "linear":
        estimator = Ridge()
        parameter = "ridge__alpha"
        values = RIDGE_ALPHAS
    else:
        estimator = TorchMLPRegressor(
            hidden_size=256,
            batch_size=64,
            learning_rate=1e-3,
            validation_fraction=0.1,
            max_iter=max_iter,
            random_state=42,
            device=device,
        )
        parameter = "torchmlpregressor__alpha"
        values = MLP_REGRESSION_ALPHAS

    search = GridSearchCV(
        make_pipeline(VarianceThreshold(1e-6), StandardScaler(), estimator),
        {parameter: values},
        scoring="r2",
        cv=cv,
        refit=True,
        n_jobs=N_JOBS,
    )
    fit_params = {}
    if kind != "linear":
        # GridSearchCV slices this per-record parameter for each training fold.
        step_name = parameter.split("__")[0]
        fit_params[f"{step_name}__groups"] = np.asarray(groups)
    search.fit(train_x, train_y, groups=groups, **fit_params)
    metrics = regression_metrics(test_y, search.predict(test_x))
    return search, parameter, metrics


def run_classification(
    kind,
    train_x,
    train_y,
    test_x,
    test_y,
    groups,
    folds,
    max_iter,
    device="auto",
):
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)
    cv = StratifiedGroupKFold(
        n_splits=min(folds, len(np.unique(groups))),
        shuffle=True,
        random_state=42,
    )
    if kind == "linear":
        estimator = LogisticRegression(max_iter=8000)
        parameter = "logisticregression__C"
        values = LOGISTIC_CS
    else:
        estimator = TorchMLPClassifier(
            hidden_size=256,
            batch_size=64,
            learning_rate=1e-3,
            validation_fraction=0.1,
            max_iter=max_iter,
            random_state=42,
            device=device,
        )
        parameter = "torchmlpclassifier__alpha"
        values = MLP_CLASSIFICATION_ALPHAS

    search = GridSearchCV(
        make_pipeline(VarianceThreshold(1e-6), StandardScaler(), estimator),
        {parameter: values},
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=N_JOBS,
    )
    fit_params = {}
    if kind != "linear":
        # GridSearchCV slices this per-record parameter for each training fold.
        step_name = parameter.split("__")[0]
        fit_params[f"{step_name}__groups"] = np.asarray(groups)
    search.fit(train_x, train_y, groups=groups, **fit_params)
    predictions = search.predict(test_x)
    scores = search.predict_proba(test_x)[:, 1]
    return search, parameter, classification_metrics(test_y, predictions, scores)


def main(argv=None, cohort_representations=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings/admet"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets/cleaned/admet"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/property_prediction")
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300, help="Maximum MLP iterations")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not CUML_ACCEL_ENABLED:
        raise RuntimeError("--device cuda requested, but cuml.accel is not installed")
    if args.device == "cpu" and CUML_ACCEL_ENABLED:
        raise RuntimeError("Use an environment without cuML for --device cpu")

    representations = parse_names(args.models)
    if cohort_representations is None:
        cohort_representations = representations
    backend = "cuml.accel" if CUML_ACCEL_ENABLED else "scikit-learn CPU"
    print(f"Backend: {backend}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in parse_names(args.datasets):
        common = {
            split: common_record_ids(
                dataset, cohort_representations, split, args.embeddings_dir
            )
            for split in ("train", "test")
        }
        print(
            f"{dataset}: {len(common['train'])} common train records, "
            f"{len(common['test'])} common test records",
            flush=True,
        )
        task = "classification" if dataset in CLASSIFICATION_DATASETS else "regression"
        rows = []

        for representation in representations:
            train_x, train_y, train_groups = load_split(
                dataset,
                representation,
                "train",
                common["train"],
                args.embeddings_dir,
                args.datasets_dir,
            )
            test_x, test_y, test_groups = load_split(
                dataset,
                representation,
                "test",
                common["test"],
                args.embeddings_dir,
                args.datasets_dir,
            )

            for kind in ("linear", "mlp"):
                print(f"{dataset}: {representation} ({kind})", flush=True)
                run = run_classification if task == "classification" else run_regression
                search, parameter, metrics = run(
                    kind,
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    train_groups,
                    args.cv_folds,
                    args.epochs,
                    args.device,
                )
                value = float(search.best_params_[parameter])
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "representation": representation,
                        "probe": kind,
                        "n_train": len(train_y),
                        "n_test": len(test_y),
                        "n_train_molecules": len(np.unique(train_groups)),
                        "n_test_molecules": len(np.unique(test_groups)),
                        "alpha": value if "alpha" in parameter else None,
                        "C": value if parameter.endswith("__C") else None,
                        "cv_score": float(search.best_score_),
                        **metrics,
                    }
                )

        results = pl.DataFrame(rows, strict=False)
        output_path = args.output_dir / f"{dataset}.csv"
        results.write_csv(output_path)
        print(results)
        print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
