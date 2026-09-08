"""Frozen MACCS linear-probe diagnostic for FoMoNMR."""

import lightning as L
import torch
import torch.distributed as dist
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

    @staticmethod
    def _gather_rows_on_rank_zero(model, *local_tensors):
        if not dist.is_available() or not dist.is_initialized():
            return local_tensors

        widths = [tensor.shape[1] for tensor in local_tensors]
        local_rows = local_tensors[0].shape[0]
        if any(tensor.shape[0] != local_rows for tensor in local_tensors):
            raise ValueError("distributed probe tensors have different row counts")

        collective_device = (
            model.device if dist.get_backend() == "nccl" else torch.device("cpu")
        )
        payload = torch.cat(local_tensors, dim=1).to(collective_device)
        size = torch.tensor([local_rows], device=collective_device)
        sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, size)
        row_counts = [int(rank_size.item()) for rank_size in sizes]
        max_rows = max(row_counts)

        padded = torch.zeros(
            (max_rows, payload.shape[1]),
            device=collective_device,
            dtype=payload.dtype,
        )
        padded[:local_rows] = payload
        gathered = (
            [torch.empty_like(padded) for _ in row_counts]
            if dist.get_rank() == 0
            else None
        )
        dist.gather(padded, gather_list=gathered, dst=0)

        if dist.get_rank() != 0:
            return None

        combined = torch.cat(
            [rank_tensor[:rows] for rank_tensor, rows in zip(gathered, row_counts)]
        ).cpu()
        return combined.split(widths, dim=1)

    def _fit_probe(
        self,
        model,
        train_embeddings,
        train_targets,
        eval_embeddings,
        eval_targets,
    ):
        cuda_devices = (
            [model.device.index or 0] if model.device.type == "cuda" else []
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

    def run(self, model):
        # Lightning validation uses inference mode; disable it only for the probe.
        with torch.inference_mode(False):
            model.eval()
            local_train = self._collect_embeddings(model, self.train_loader)
            local_eval = self._collect_embeddings(model, self.eval_loader)
            gathered_train = self._gather_rows_on_rank_zero(model, *local_train)
            gathered_eval = self._gather_rows_on_rank_zero(model, *local_eval)

            score = torch.zeros((), device=model.device)
            if gathered_train is not None and gathered_eval is not None:
                score = self._fit_probe(
                    model,
                    *gathered_train,
                    *gathered_eval,
                ).to(model.device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(score, src=0)
            return score

    def on_validation_epoch_end(self, trainer, model):
        if trainer.sanity_checking:
            return

        self.validation_count += 1
        if self.validation_count % self.every_n_validations != 0:
            return

        model.log(
            "probe/maccs_macro_auroc",
            self.run(model),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


__all__ = ["MaccsLinearProbe"]
