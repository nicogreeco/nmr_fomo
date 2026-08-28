import torch

from ..processor import MULTIPLICITY_TO_ID


def apply_global_shift(
    shifts,
    peak_mask,
    minimum,
    maximum,
    max_offset,
    probability,
):
    lowest_shift = shifts.masked_fill(~peak_mask, float("inf")).amin(dim=1)
    highest_shift = shifts.masked_fill(~peak_mask, float("-inf")).amax(dim=1)

    offset_min = torch.maximum(
        shifts.new_full((shifts.shape[0],), -max_offset),
        minimum - lowest_shift,
    )
    offset_max = torch.minimum(
        shifts.new_full((shifts.shape[0],), max_offset),
        maximum - highest_shift,
    )

    offsets = offset_min + torch.rand_like(offset_min) * (offset_max - offset_min)
    use_offset = (
        torch.rand_like(offsets) < probability
    ) & peak_mask.any(dim=1)
    offsets *= use_offset
    return shifts + offsets.unsqueeze(1) * peak_mask


def select_masked_peak(peak_mask):
    scores = torch.rand(peak_mask.shape, device=peak_mask.device)
    scores[~peak_mask] = -1
    selected_peak = scores.argmax(dim=1, keepdim=True)

    prediction_mask = torch.zeros_like(peak_mask)
    prediction_mask.scatter_(1, selected_peak, True)
    return prediction_mask & peak_mask


def apply_local_jitter(
    shifts,
    peak_mask,
    prediction_mask,
    probability,
    sigma,
    minimum,
    maximum,
):
    jitter_spectrum = (
        torch.rand(shifts.shape[0], 1, device=shifts.device) < probability
    )
    jitter_mask = peak_mask & ~prediction_mask & jitter_spectrum

    jittered_shifts = shifts.clone()
    jittered_shifts[jitter_mask] += (
        torch.randn_like(shifts)[jitter_mask] * sigma
    )
    return jittered_shifts.clamp_(minimum, maximum)


def mask_annotations(h, probability):
    available = h["availability"]
    valid_peaks = h["peak_mask"]
    annotation_mask = torch.zeros_like(available)

    mask_integration = (
        torch.rand(valid_peaks.shape[0], 1, device=valid_peaks.device)
        < probability
    )
    annotation_mask[:, :, 0] = (
        available[:, :, 0] & valid_peaks & mask_integration
    )

    random_masks = torch.rand(
        *valid_peaks.shape,
        3,
        device=valid_peaks.device,
    ) < probability
    annotation_mask[:, :, 1] = (
        available[:, :, 1]
        & valid_peaks
        & random_masks[:, :, 0]
        & (h["multiplicity"] != MULTIPLICITY_TO_ID["<unk>"])
    )
    annotation_mask[:, :, 2] = (
        available[:, :, 2] & valid_peaks & random_masks[:, :, 1]
    )
    annotation_mask[:, :, 3] = (
        available[:, :, 3] & valid_peaks & random_masks[:, :, 2]
    )

    h["availability"] &= ~annotation_mask
    h["j_mask"][annotation_mask[:, :, 2]] = False
    return annotation_mask
