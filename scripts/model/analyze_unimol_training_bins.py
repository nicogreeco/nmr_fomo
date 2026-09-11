"""Measure UniMol teacher-cosine bin occupancy on sampled training batches."""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.FoMoNMR import ModelConfig
from model.losses import capped_similarity_indices, tanimoto_to_bins
from model.train import make_dataloaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="scripts/model/configs/ablation/"
        "final_posttrain_unimol_relational_lambda025_trial.yaml",
    )
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs-per-bin", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("results/unimol_training_bins.csv")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batches < 1:
        raise ValueError("--batches must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    with open(args.config) as file:
        values = yaml.safe_load(file)
    config = ModelConfig(**values)
    if config.molecular_target != "unimol":
        raise ValueError("Expected a UniMol configuration")

    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA requested, but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    loader, _ = make_dataloaders(
        "posttrain", args.batch_size, args.num_workers, args.seed, config=config
    )
    num_bins = int(1.0 / config.fp_sim_bin_size) + 1
    candidate_totals = torch.zeros(num_bins, dtype=torch.long, device=device)
    selected_totals = torch.zeros_like(candidate_totals)
    nonempty_batches = torch.zeros_like(candidate_totals)
    rich_counts = []

    for batch_index, batch in zip(range(args.batches), loader):
        mask = batch["unimol_mask"]
        teacher = batch["unimol_embeddings"][mask].to(device, non_blocking=True)
        rich_counts.append(len(teacher))
        teacher = F.normalize(teacher.float(), dim=1)
        pairs = torch.triu_indices(len(teacher), len(teacher), 1, device=device)
        targets = (teacher @ teacher.T)[pairs[0], pairs[1]].clamp(-1, 1)
        scaled = (targets + 1.0) / 2.0
        indices = tanimoto_to_bins(scaled, config.fp_sim_bin_size, num_bins)
        counts = torch.bincount(indices, minlength=num_bins)
        selected = capped_similarity_indices(
            scaled, config.fp_sim_bin_size, args.max_pairs_per_bin
        )
        selected_counts = torch.bincount(indices[selected], minlength=num_bins)
        candidate_totals += counts
        selected_totals += selected_counts
        nonempty_batches += counts.gt(0)
        print(
            f"batch {batch_index + 1}/{args.batches}: "
            f"rich={len(teacher)}, pairs={len(targets):,}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates = candidate_totals.cpu().tolist()
    selected = selected_totals.cpu().tolist()
    nonempty = nonempty_batches.cpu().tolist()
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((
            "bin",
            "cosine_range",
            "candidate_pairs",
            "mean_candidate_pairs_per_batch",
            "selected_pairs",
            "mean_selected_pairs_per_batch",
            "batches_nonempty",
            "nonempty_batch_fraction",
        ))
        for index, (raw, kept, present) in enumerate(
            zip(candidates, selected, nonempty)
        ):
            lower = -1.0 + 2.0 * index * config.fp_sim_bin_size
            if index == num_bins - 1:
                label = "1.0 (endpoint)"
            else:
                upper = min(1.0, lower + 2.0 * config.fp_sim_bin_size)
                label = f"[{lower:.1f}, {upper:.1f})"
            writer.writerow((
                index,
                label,
                raw,
                raw / args.batches,
                kept,
                kept / args.batches,
                present,
                present / args.batches,
            ))
    print(
        f"Saved {args.output}; mean Rich records/batch="
        f"{sum(rich_counts) / len(rich_counts):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
