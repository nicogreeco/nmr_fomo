"""Post-hoc Rich validation cosine diagnostics; does not update the model."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.FoMoNMR import FoMoNMR
from model.processor import FoundationNMRProcessor
from model.train import paired_dataset


def range_statistics(target, prediction):
    """Twenty teacher-cosine bins of width 0.1, including both endpoints."""
    index = ((target + 1) / 0.1).long().clamp(0, 19)
    target, prediction = target.double(), prediction.double()
    error = prediction - target
    values = [torch.ones_like(target), target, prediction, target.square(),
              prediction.square(), target * prediction, error.abs(), error.square()]
    return torch.stack([torch.bincount(index, weights=v, minlength=20) for v in values], dim=1)


def metric_row(label, sums):
    n, target, prediction, target2, prediction2, product, absolute, squared = sums.tolist()
    if not n:
        return dict(range=label, count=0)
    target_var = max(0., target2 / n - (target / n)**2)
    prediction_var = max(0., prediction2 / n - (prediction / n)**2)
    denominator = (target_var * prediction_var)**0.5
    return dict(range=label, count=int(n), mean_target_similarity=target/n,
                mean_predicted_similarity=prediction/n, bias=(prediction-target)/n,
                mae=absolute/n, mse=squared/n, predicted_std=prediction_var**0.5,
                pearson=(product/n-target*prediction/n**2)/denominator if denominator else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'cosine_ranges.csv'
    if output.exists():
        raise FileExistsError(output)
    torch.set_float32_matmul_precision('high')
    model = FoMoNMR.load_from_checkpoint(args.checkpoint, map_location='cpu').eval().cuda()
    if model.config.molecular_target != 'unimol':
        raise ValueError('Expected a UniMol relational checkpoint')
    dataset = paired_dataset('rich', 'val', False, False, model.config.validation_seed,
                             shard_across_ranks=False,
                             unimol_sidecar_dir=model.config.unimol_sidecar_dir)
    loader = DataLoader(dataset, batch_size=2048, num_workers=0, drop_last=False,
                        collate_fn=FoundationNMRProcessor(molecular_target='unimol'))
    totals = torch.zeros(20, 8, dtype=torch.float64, device='cuda')
    records = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = model.transfer_batch_to_device(batch, model.device, 0)
            with torch.random.fork_rng():
                torch.manual_seed(model.config.validation_seed + batch_idx)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    corrupted, corruption = model.corrupt_batch(batch)
                    pooled, _, _ = model(corrupted,
                        h_shift_prediction_mask=corruption['h_shift_mask'],
                        c_shift_prediction_mask=corruption['c_shift_mask'])
            mask = batch['unimol_mask']
            assert mask.all(), 'Rich validation must have a teacher for every record'
            student = F.normalize(pooled[mask].float(), dim=1)
            teacher = F.normalize(batch['unimol_embeddings'][mask].float(), dim=1)
            i, j = torch.triu_indices(len(student), len(student), 1, device='cuda')
            target = (teacher @ teacher.T)[i, j].clamp(-1, 1)
            prediction = (student @ student.T)[i, j].clamp(-1, 1)
            totals += range_statistics(target, prediction)
            records += len(student)
            print(f'batch {batch_idx+1}: {records} records', flush=True)
    rows = [metric_row(f'[{(-1+k*.1):.1f}, {(-.9+k*.1):.1f}{"]" if k==19 else ")"}', sums)
            for k, sums in enumerate(totals.cpu())]
    overall = metric_row('overall', totals.sum(0).cpu())
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=overall.keys())
        writer.writeheader()
        writer.writerows(rows + [overall])
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), records=records,
                    batch_size=2048, num_workers=0, seed=model.config.validation_seed,
                    precision='bf16-mixed', input='deterministically corrupted validation',
                    pairs='all within-batch unordered non-self pairs; no balancing',
                    teacher='Rich train-centered, L2-normalized UniMol2', overall=overall)
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
