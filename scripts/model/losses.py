import torch
import torch.nn.functional as F


def gaussian_soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Distance-aware shift classification adapted from UltraNMR."""
    distances = targets.unsqueeze(1) - bin_centers.unsqueeze(0)
    soft_targets = torch.exp(-0.5 * (distances / sigma) ** 2)
    soft_targets /= soft_targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Focal classification loss following UltraNMR's implementation."""
    cross_entropy = F.cross_entropy(logits, targets, reduction="none")
    target_probability = torch.exp(-cross_entropy)
    return (((1.0 - target_probability) ** gamma) * cross_entropy).mean()


def symmetric_fingerprint_pairs(
    pooled: torch.Tensor,
    fingerprints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all unique symmetric representation pairs and Tanimoto targets."""

    batch_size = pooled.shape[0]
    pair_indices = torch.triu_indices(
        batch_size,
        batch_size,
        offset=1,
        device=pooled.device,
    )
    first = pooled[pair_indices[0]]
    second = pooled[pair_indices[1]]
    features = torch.cat(
        (torch.abs(first - second), first * second),
        dim=-1,
    )

    intersections = fingerprints @ fingerprints.T
    bit_counts = fingerprints.sum(dim=1, keepdim=True)
    unions = bit_counts + bit_counts.T - intersections
    tanimoto = intersections / unions.clamp_min(1e-8)
    similarities = tanimoto[pair_indices[0], pair_indices[1]]
    return features, similarities


def tanimoto_to_bins(
    similarities: torch.Tensor,
    bin_size: float,
    num_bins: int,
) -> torch.Tensor:
    """Apply UltraNMR's floor-and-clamp convention, including exact 1.0."""

    return (similarities / bin_size).long().clamp(0, num_bins - 1)
