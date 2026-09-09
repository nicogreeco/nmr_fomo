"""Extract pooled embeddings from a FoMoNMR checkpoint."""

from pathlib import Path

from . import EmbeddingResult


class FoMoNMREmbedder:
    """Load FoMoNMR from a local checkpoint or an MLflow run."""

    model_name = "fomonmr"
    modality = "1H+13C"
    pooling = "mean of valid 1H+13C transformer peak states"

    def __init__(self, checkpoint_path=None, run_id=None, device="cpu", tracking_uri=None):
        import torch
        from model.FoMoNMR import FoMoNMR

        if (checkpoint_path is None) == (run_id is None):
            raise ValueError("give exactly one of checkpoint_path or run_id")

        self._torch = torch
        self.device = torch.device(device)
        if run_id is not None:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self.checkpoint_path = f"runs:/{run_id}/model"
            self.model = mlflow.pytorch.load_model(
                self.checkpoint_path, map_location="cpu"
            )
        else:
            self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
            self.model = FoMoNMR.load_from_checkpoint(
                self.checkpoint_path, map_location="cpu"
            )

        self.model.eval().to(self.device)
        self.dimension = int(self.model.config.d_model)

    def encode(self, batch):
        record_ids = list(batch["record_ids"])
        batch = self.model.transfer_batch_to_device(batch, self.device, 0)

        if self.model.config.stage == "pretrain" or not self.model.config.use_rich_input:
            batch["h"]["availability"].zero_()
            batch["h"]["j_mask"].zero_()

        with self._torch.inference_mode():
            embeddings, _, _ = self.model(batch)

        return EmbeddingResult(
            embeddings=embeddings.detach().to(device="cpu", dtype=self._torch.float32),
            record_ids=record_ids,
            model_name=self.model_name,
            checkpoint=str(self.checkpoint_path),
            dimension=self.dimension,
            pooling=self.pooling,
            metadata={
                "modality": self.modality,
                "stage": self.model.config.stage,
                "use_rich_input": bool(self.model.config.use_rich_input),
            },
        )


__all__ = ["FoMoNMREmbedder"]
