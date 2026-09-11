import torch
import torch.nn.functional as F
from typing import List


def capped_similarity_indices(similarities, bin_size, max_pairs_per_bin):
    """Keep at most N random pairs per bin for similarities scaled to [0, 1]."""
    num_bins = int(1.0 / bin_size) + 1
    bins = tanimoto_to_bins(similarities, bin_size, num_bins)
    selected = []
    for bin_index in range(num_bins):
        indices = torch.where(bins == bin_index)[0]
        if len(indices) > max_pairs_per_bin:
            order = torch.randperm(len(indices), device=indices.device)
            indices = indices[order[:max_pairs_per_bin]]
        selected.append(indices)
    return torch.cat(selected)


def relational_cosine_loss(
    pooled,
    teacher,
    bin_size=0.05,
    max_pairs_per_bin=None,
    return_pairs=False,
):
    """Match off-diagonal cosine geometry; teacher is train-centered upstream.

    No learned pair head: the loss constrains the reusable NMR embedding itself.
    Calculate in float32 even under autocast; the teacher never gets gradients.
    Binning maps [-1, 1] to [0, 1] only for sampling, not for the MSE target.
    """
    if len(pooled) < 2:
        zero = pooled.sum() * 0.0
        if return_pairs:
            empty = pooled.new_empty(0, dtype=torch.float32)
            return zero, zero, empty, empty
        return zero, zero
    with torch.autocast(device_type=pooled.device.type, enabled=False):
        student = F.normalize(pooled.float(), dim=1)
        teacher = F.normalize(teacher.detach().float(), dim=1)
        pairs = torch.triu_indices(len(student), len(student), offset=1, device=student.device)
        targets = (teacher @ teacher.T)[pairs[0], pairs[1]].clamp(-1, 1)
        if max_pairs_per_bin is not None:
            selected = capped_similarity_indices(
                (targets + 1) / 2, bin_size, max_pairs_per_bin
            )
            pairs = pairs[:, selected]
            targets = targets[selected]
        predictions = (student @ student.T)[pairs[0], pairs[1]].clamp(-1, 1)
        loss = F.mse_loss(predictions, targets)
        mae = (predictions - targets).abs().mean()
        if return_pairs:
            return loss, mae, predictions, targets
        return loss, mae


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
    weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Focal classification loss following UltraNMR's implementation."""
    cross_entropy = F.cross_entropy(logits, targets, reduction="none")
    target_probability = torch.exp(-cross_entropy)
    loss = ((1.0 - target_probability) ** gamma) * cross_entropy
    if weights is not None:
        return (loss * weights).sum() / weights.sum()
    return loss.mean()

def symmetric_fingerprint_pairs(
    pooled: torch.Tensor,
    fingerprints: torch.Tensor,
    bin_size: float = 0.05,
    max_pairs_per_bin: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return symmetric pairs, optionally capped within each Tanimoto bin."""

    batch_size = pooled.shape[0]
    pair_indices = torch.triu_indices(
        batch_size,
        batch_size,
        offset=1,
        device=pooled.device,
    )
    intersections = fingerprints @ fingerprints.T
    bit_counts = fingerprints.sum(dim=1, keepdim=True)
    unions = bit_counts + bit_counts.T - intersections
    tanimoto = intersections / unions.clamp_min(1e-8)
    similarities = tanimoto[pair_indices[0], pair_indices[1]]

    if max_pairs_per_bin is not None:
        selected = capped_similarity_indices(similarities, bin_size, max_pairs_per_bin)
        pair_indices = pair_indices[:, selected]
        similarities = similarities[selected]

    first = pooled[pair_indices[0]]
    second = pooled[pair_indices[1]]
    features = torch.cat(
        (torch.abs(first - second), first * second),
        dim=-1,
    )
    return features, similarities


def tanimoto_to_bins(
    similarities: torch.Tensor,
    bin_size: float,
    num_bins: int,
) -> torch.Tensor:
    """Apply UltraNMR's floor-and-clamp convention, including exact 1.0."""

    return (similarities / bin_size).long().clamp(0, num_bins - 1)
