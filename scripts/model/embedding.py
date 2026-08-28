from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .processor import MAX_J_VALUES, MULTIPLICITY_TO_ID
from .utils.fourier_features import FourierFeatures, RBFExpansion


INTEGRATION_DIM = 32
MULTIPLICITY_DIM = 32
J_COUPLING_DIM = 64
WIDTH_DIM = 32
NUM_AVAILABILITY_FEATURES = 4


class ShiftEmbedder(nn.Module):
    """Embed scalar chemical shifts using Fourier features followed by an MLP."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        resolution: float,
        d_model: int,
        fourier_strategy: str,
        num_fourier_freqs: Optional[int],
    ) -> None:
        super().__init__()

        self.fourier = FourierFeatures(
            strategy=fourier_strategy,
            x_min=x_min,
            x_max=x_max,
            resolution=resolution,
            num_freqs=num_fourier_freqs,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.fourier.num_features(), d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shift: Chemical shifts with shape [B, N].

        Returns:
            Peak embeddings with shape [B, N, d_model].
        """
        return self.mlp(self.fourier(shift.unsqueeze(-1)))


class IntegrationEmbedder(nn.Module):
    """Embed absolute and spectrum-relative proton integration values."""

    def __init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, INTEGRATION_DIM),
            nn.GELU(),
        )

    def forward(self, integration: torch.Tensor) -> torch.Tensor:
        """
        Args:
            integration: Proton integrations with shape [B, N].

        Returns:
            Integration embeddings with shape [B, N, INTEGRATION_DIM].
        """
        total = integration.sum(dim=1, keepdim=True).clamp_min(1)
        relative = integration / total
        log_absolute = torch.log1p(integration)

        features = torch.stack(
            (log_absolute, relative),
            dim=-1,
        )

        return self.mlp(features)


class MultiplicityEmbedder(nn.Module):
    """Embed categorical proton multiplicities."""

    def __init__(self, num_multiplicities: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_multiplicities,
            embedding_dim=MULTIPLICITY_DIM,
            padding_idx=0,
        )

    def forward(self, multiplicity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multiplicity: Multiplicity IDs with shape [B, N].

        Returns:
            Multiplicity embeddings with shape [B, N, MULTIPLICITY_DIM].
        """
        return self.embedding(multiplicity)


class JCouplingEmbedder(nn.Module):
    """Embed a variable-size set of J-coupling constants for each proton peak."""

    def __init__(
        self,
        j_min: float,
        j_max: float,
        num_rbf_centers: int,
        sigma: float,
    ) -> None:
        super().__init__()

        self.rbf_embedder = RBFExpansion(
            n=num_rbf_centers,
            x_min=j_min,
            x_max=j_max,
            sigma=sigma,
        )

        self.j_value_mlp = nn.Sequential(
            nn.Linear(num_rbf_centers + 1, J_COUPLING_DIM),
            nn.GELU(),
            nn.Linear(J_COUPLING_DIM, J_COUPLING_DIM),
        )

        self.j_pool_projection = nn.Sequential(
            nn.Linear(J_COUPLING_DIM + 1, J_COUPLING_DIM),
            nn.GELU(),
        )

    def forward(
        self,
        j_values: torch.Tensor,
        j_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            j_values:
                J values with shape [B, N, K].
            j_mask:
                Boolean valid-value mask with shape [B, N, K].

        Returns:
            Per-peak J embeddings with shape [B, N, J_COUPLING_DIM].
        """
        rbf_features = self.rbf_embedder(j_values)
        log_features = torch.log1p(j_values).unsqueeze(-1)

        j_features = torch.cat(
            (rbf_features, log_features),
            dim=-1,
        )

        j_embeddings = self.j_value_mlp(j_features)

        mask = j_mask.unsqueeze(-1).to(j_embeddings.dtype)

        pooled = (j_embeddings * mask).sum(dim=2)

        normalized_count = (
            j_mask.sum(dim=-1, dtype=j_embeddings.dtype) / MAX_J_VALUES
        )

        pooled = torch.cat(
            (pooled, normalized_count.unsqueeze(-1)),
            dim=-1,
        )

        return self.j_pool_projection(pooled)


class WidthEmbedder(nn.Module):
    """Embed the half-span of a reported proton chemical-shift range."""

    def __init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, WIDTH_DIM),
            nn.GELU(),
        )

    def forward(self, width: torch.Tensor) -> torch.Tensor:
        """
        Args:
            width: Range half-span with shape [B, N].

        Returns:
            Width embeddings with shape [B, N, WIDTH_DIM].
        """
        return self.mlp(width.unsqueeze(-1))


class HPeakEmbedder(nn.Module):
    """
    Build proton peak embeddings from chemical shift and optional rich annotations.

    Rich annotations are fused into a residual vector that is added to the
    shift-only representation.
    """

    def __init__(
        self,
        d_model: int,
        h_range: tuple[float, float],
        h_resolution: float,
        fourier_strategy: str,
        num_fourier_freqs: Optional[int],
        num_multiplicities: int,
        j_min: float,
        j_max: float,
        num_rbf_centers: int,
        rbf_sigma: float,
    ) -> None:
        super().__init__()

        self.shift_embedder = ShiftEmbedder(
            *h_range,
            resolution=h_resolution,
            d_model=d_model,
            fourier_strategy=fourier_strategy,
            num_fourier_freqs=num_fourier_freqs,
        )

        self.integration_embedder = IntegrationEmbedder()
        self.multiplicity_embedder = MultiplicityEmbedder(num_multiplicities)
        self.j_embedder = JCouplingEmbedder(
            j_min=j_min,
            j_max=j_max,
            num_rbf_centers=num_rbf_centers,
            sigma=rbf_sigma,
        )
        self.width_embedder = WidthEmbedder()

        rich_input_dim = (
            INTEGRATION_DIM
            + MULTIPLICITY_DIM
            + J_COUPLING_DIM
            + WIDTH_DIM
            + NUM_AVAILABILITY_FEATURES
        )

        self.rich_mlp = nn.Sequential(
            nn.Linear(rich_input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        nn.init.normal_(self.rich_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.rich_mlp[-1].bias)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        shift: torch.Tensor,
        integration: torch.Tensor,
        multiplicity: torch.Tensor,
        j_values: torch.Tensor,
        width: torch.Tensor,
        j_mask: torch.Tensor,
        availability: torch.Tensor,
        shift_prediction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            shift: [B, N]
            integration: [B, N]
            multiplicity: [B, N]
            j_values: [B, N, K]
            width: [B, N]
            j_mask: [B, N, K]
            availability: Boolean feature mask [B, N, 4].

        Returns:
            Proton peak embeddings with shape [B, N, d_model].
        """
        availability_features = availability.to(dtype=shift.dtype)
        a_i, a_m, a_j, a_w = torch.unbind(availability_features, dim=-1)

        shift_embedding = self.shift_embedder(shift)
        if shift_prediction_mask is not None:
            shift_embedding = shift_embedding.masked_fill(
                shift_prediction_mask.unsqueeze(-1),
                0.0,
            )

        integration_embedding = (
            a_i.unsqueeze(-1) * self.integration_embedder(integration)
        )
        multiplicity_embedding = (
            a_m.unsqueeze(-1) * self.multiplicity_embedder(multiplicity)
        )
        j_embedding = (
            a_j.unsqueeze(-1) * self.j_embedder(j_values, j_mask)
        )
        width_embedding = (
            a_w.unsqueeze(-1) * self.width_embedder(width)
        )

        rich_input = torch.cat(
            (
                integration_embedding,
                multiplicity_embedding,
                j_embedding,
                width_embedding,
                availability_features,
            ),
            dim=-1,
        )

        rich_embedding = self.rich_mlp(rich_input)

        has_rich_features = availability.any(dim=-1, keepdim=True)
        rich_embedding = (
            rich_embedding
            * has_rich_features.to(rich_embedding.dtype)
        )

        return self.layer_norm(shift_embedding + rich_embedding)


class CPeakEmbedder(nn.Module):
    """Build carbon peak embeddings from chemical shifts."""

    def __init__(
        self,
        d_model: int,
        c_range: tuple[float, float],
        c_resolution: float,
        fourier_strategy: str,
        num_fourier_freqs: Optional[int],
    ) -> None:
        super().__init__()

        self.shift_embedder = ShiftEmbedder(
            *c_range,
            resolution=c_resolution,
            d_model=d_model,
            fourier_strategy=fourier_strategy,
            num_fourier_freqs=num_fourier_freqs,
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        shift: torch.Tensor,
        shift_prediction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            shift: Carbon shifts with shape [B, N].

        Returns:
            Carbon peak embeddings with shape [B, N, d_model].
        """
        shift_embedding = self.shift_embedder(shift)
        if shift_prediction_mask is not None:
            shift_embedding = shift_embedding.masked_fill(
                shift_prediction_mask.unsqueeze(-1),
                0.0,
            )
        return self.layer_norm(shift_embedding)


class NmrEmbedder(nn.Module):
    """
    Convert canonical batched 1H and 13C peak data into Transformer-ready tokens.

    Output peaks are concatenated along the set dimension:
        [H peaks, C peaks].
    """

    def __init__(
        self,
        d_model: int = 512,
        fourier_strategy: str = "log_spaced",
        h_x_min: float = -5.5,
        h_x_max: float = 20.0,
        h_resolution: float = 0.02,
        c_x_min: float = -40.0,
        c_x_max: float = 300.0,
        c_resolution: float = 0.2,
        num_fourier_freqs: Optional[int] = 256,
        num_multiplicities: Optional[int] = None,
        j_min: float = 0.0,
        j_max: float = 20.0,
        num_rbf_centers: int = 32,
        rbf_sigma: float = 2.0,
    ) -> None:
        super().__init__()

        if num_multiplicities is None:
            num_multiplicities = len(MULTIPLICITY_TO_ID)

        self.d_model = d_model

        self.h_type_embedding = nn.Parameter(torch.empty(d_model))
        self.c_type_embedding = nn.Parameter(torch.empty(d_model))

        nn.init.normal_(self.h_type_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.c_type_embedding, mean=0.0, std=0.02)

        self.h_embedder = HPeakEmbedder(
            d_model=d_model,
            h_range=(h_x_min, h_x_max),
            h_resolution=h_resolution,
            fourier_strategy=fourier_strategy,
            num_fourier_freqs=num_fourier_freqs,
            num_multiplicities=num_multiplicities,
            j_min=j_min,
            j_max=j_max,
            num_rbf_centers=num_rbf_centers,
            rbf_sigma=rbf_sigma,
        )

        self.c_embedder = CPeakEmbedder(
            d_model=d_model,
            c_range=(c_x_min, c_x_max),
            c_resolution=c_resolution,
            fourier_strategy=fourier_strategy,
            num_fourier_freqs=num_fourier_freqs,
        )

    def forward(
        self,
        batch: dict,
        *,
        h_shift_prediction_mask: Optional[torch.Tensor] = None,
        c_shift_prediction_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Canonical collated NMR batch.

        Returns:
            nmr_peak_tokens:
                Tensor with shape [B, N_H + N_C, d_model].

            nmr_peak_valid_mask:
                Boolean tensor with shape [B, N_H + N_C].
                True indicates a real peak, False indicates padding.
        """
        h_peaks = batch["h"]
        c_peaks = batch["c"]

        h_tokens = self.h_embedder(
            shift=h_peaks["shift"],
            integration=h_peaks["integration"],
            multiplicity=h_peaks["multiplicity"],
            j_values=h_peaks["j_values"],
            width=h_peaks["range_half_span"],
            j_mask=h_peaks["j_mask"],
            availability=h_peaks["availability"],
            shift_prediction_mask=h_shift_prediction_mask,
        )

        c_tokens = self.c_embedder(
            shift=c_peaks["shift"],
            shift_prediction_mask=c_shift_prediction_mask,
        )

        h_valid = h_peaks["peak_mask"].unsqueeze(-1)
        c_valid = c_peaks["peak_mask"].unsqueeze(-1)

        h_tokens = (
            h_tokens + self.h_type_embedding.view(1, 1, -1)
        ) * h_valid.to(h_tokens.dtype)

        c_tokens = (
            c_tokens + self.c_type_embedding.view(1, 1, -1)
        ) * c_valid.to(c_tokens.dtype)

        nmr_peak_tokens = torch.cat(
            (h_tokens, c_tokens),
            dim=1,
        )

        nmr_peak_valid_mask = torch.cat(
            (h_peaks["peak_mask"], c_peaks["peak_mask"]),
            dim=1,
        )

        return nmr_peak_tokens, nmr_peak_valid_mask
