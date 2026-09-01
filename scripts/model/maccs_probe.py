"""Frozen MACCS linear-probe diagnostic for FoMoNMR."""

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.functional.classification import multilabel_auroc


class MaccsLinearProbe(L.Callback):
    """Train a fresh linear MACCS probe after each validation run."""

    def __init__(
        self,
        train_loader,
        eval_loader,
        *,
        epochs=20,
        lr=1e-3,
        batch_size=1024,
        seed=42,
        every_n_validations=1,
    ):
        if every_n_validations < 1:
            raise ValueError("every_n_validations must be at least 1")
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.every_n_validations = every_n_validations
        self.validation_count = 0

    @staticmethod
    def _collect_embeddings(model, loader):
        embeddings = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                batch = model.transfer_batch_to_device(batch, model.device, 0)
                batch["h"]["availability"].zero_()
                batch["h"]["j_mask"].zero_()
                pooled, _, _ = model(batch)
                embeddings.append(pooled.float().cpu())
                targets.append(batch["maccs_fingerprints"].float().cpu())
        return torch.cat(embeddings), torch.cat(targets)

    def run(self, model):
        cuda_devices = (
            [model.device.index or 0] if model.device.type == "cuda" else []
        )
        # Lightning validation uses inference mode; disable it only for the probe.
        with torch.inference_mode(False):
            model.eval()
            train_embeddings, train_targets = self._collect_embeddings(
                model, self.train_loader
            )
            eval_embeddings, eval_targets = self._collect_embeddings(
                model, self.eval_loader
            )

            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(self.seed)
                order = torch.randperm(len(train_embeddings))
                probe = nn.Linear(
                    model.config.d_model,
                    train_targets.shape[1],
                ).to(model.device)

            train_embeddings = train_embeddings[order].to(model.device)
            train_targets = train_targets[order].to(model.device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)
            criterion = nn.BCEWithLogitsLoss()

            for _ in range(self.epochs):
                for start in range(0, len(train_embeddings), self.batch_size):
                    stop = start + self.batch_size
                    loss = criterion(
                        probe(train_embeddings[start:stop]),
                        train_targets[start:stop],
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            probe.eval()
            with torch.no_grad():
                predictions = torch.sigmoid(
                    probe(eval_embeddings.to(model.device))
                ).cpu()
            positives = eval_targets.sum(dim=0)
            valid = (positives > 0) & (positives < len(eval_targets))
            return multilabel_auroc(
                predictions[:, valid],
                eval_targets[:, valid].int(),
                num_labels=int(valid.sum()),
                average="macro",
            )

    def on_validation_epoch_end(self, trainer, model):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        self.validation_count += 1
        if self.validation_count % self.every_n_validations != 0:
            return

        model.log(
            "probe/maccs_macro_auroc",
            self.run(model),
            on_step=False,
            on_epoch=True,
        )


__all__ = ["MaccsLinearProbe"]
