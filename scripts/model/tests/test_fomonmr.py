"""Focused tests for FoMoNMR pretraining and continued pretraining."""

import copy
import unittest

import torch

from data import CanonicalRecord, CarbonPeak, ProtonPeak
from model import FoundationNMRProcessor, MAX_H_PEAKS, MULTIPLICITY_TO_ID
from model.FoMoNMR import (
    FoMoNMR,
    ModelConfig,
)
from model.losses import (
    gaussian_soft_cross_entropy,
    symmetric_fingerprint_pairs,
    tanimoto_to_bins,
)


def small_config(stage="pretrain", **overrides):
    options = {
        "d_model": 16,
        "num_fourier_freqs": 8,
        "num_rbf_centers": 4,
        "stage": stage,
        "nhead": 4,
        "num_encoder_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
    }
    options.update(overrides)
    return ModelConfig(**options)


def rich_peak(
    shift,
    *,
    integration=None,
    multiplicity=None,
    j_values=None,
    width=None,
):
    return ProtonPeak(
        shift=shift,
        integration=integration,
        multiplicity=multiplicity,
        j_values=j_values,
        range_half_span=width,
    )


def example_batch():
    records = [
        CanonicalRecord(
            record_id="paired",
            h_nmr_peaks=(
                rich_peak(
                    -1.0,
                    integration=3,
                    multiplicity="t",
                    j_values=(7.0, 1.0),
                    width=0.02,
                ),
                rich_peak(
                    2.0,
                    integration=1,
                    multiplicity="<unk>",
                    j_values=(),
                    width=0.0,
                ),
            ),
            c_nmr_peaks=(CarbonPeak(-10.0), CarbonPeak(100.0)),
        ),
        CanonicalRecord(
            record_id="h-only-missing",
            h_nmr_peaks=(rich_peak(4.0),),
        ),
        CanonicalRecord(
            record_id="c-only",
            c_nmr_peaks=(CarbonPeak(50.0),),
        ),
    ]
    batch = FoundationNMRProcessor()(records)
    fingerprints = torch.zeros(3, 2048)
    fingerprints[0, :2] = 1
    fingerprints[1, 0] = 1
    fingerprints[2, 2] = 1
    batch["fingerprints"] = fingerprints
    return batch


class FoMoNMRTests(unittest.TestCase):
    def test_negative_ranges_bins_and_soft_labels(self):
        model = FoMoNMR(small_config())
        h_bin_centers = torch.linspace(-5.5, 20.0, 1276)
        c_bin_centers = torch.linspace(-40.0, 300.0, 1701)

        self.assertEqual(model.h_classification_head.out_features, 1276)
        self.assertEqual(model.c_classification_head.out_features, 1701)
        self.assertAlmostEqual(h_bin_centers[0].item(), -5.5)
        self.assertAlmostEqual(h_bin_centers[-1].item(), 20.0)
        self.assertAlmostEqual(c_bin_centers[0].item(), -40.0)
        self.assertAlmostEqual(c_bin_centers[-1].item(), 300.0)

        endpoint_batch = FoundationNMRProcessor()(
            [
                CanonicalRecord(
                    record_id="endpoints",
                    h_nmr_peaks=(
                        ProtonPeak(-5.5),
                        ProtonPeak(20.0),
                    ),
                    c_nmr_peaks=(
                        CarbonPeak(-40.0),
                        CarbonPeak(300.0),
                    ),
                )
            ]
        )
        pooled, peaks, valid = model(endpoint_batch)
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.isfinite(peaks).all())
        self.assertEqual(valid.sum().item(), 4)

        logits = torch.zeros(1, h_bin_centers.numel())
        loss = gaussian_soft_cross_entropy(
            logits,
            torch.tensor([-5.5]),
            h_bin_centers,
            sigma=0.05,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_masked_shift_zeroes_only_shift_contribution(self):
        model = FoMoNMR(small_config())
        batch = example_batch()
        batch["h"]["availability"].fill_(False)
        batch["h"]["j_mask"].fill_(False)

        h_shift_mask = torch.zeros_like(batch["h"]["peak_mask"])
        h_shift_mask[0, 0] = True
        c_shift_mask = torch.zeros_like(batch["c"]["peak_mask"])
        c_shift_mask[0, 0] = True
        tokens_a, valid_a = model.embedding(
            batch,
            h_shift_prediction_mask=h_shift_mask,
            c_shift_prediction_mask=c_shift_mask,
        )

        changed_batch = copy.deepcopy(batch)
        changed_batch["h"]["shift"][0, 0] = 12.0
        changed_batch["c"]["shift"][0, 0] = 250.0
        tokens_b, valid_b = model.embedding(
            changed_batch,
            h_shift_prediction_mask=h_shift_mask,
            c_shift_prediction_mask=c_shift_mask,
        )
        unmasked_tokens, _ = model.embedding(changed_batch)

        self.assertTrue(valid_a[0, 0].item())
        self.assertTrue(valid_b[0, 0].item())
        self.assertTrue(torch.allclose(tokens_a[0, 0], tokens_b[0, 0]))
        self.assertFalse(torch.allclose(tokens_a[0, 0], unmasked_tokens[0, 0]))
        c_token_index = MAX_H_PEAKS
        self.assertTrue(
            torch.allclose(
                tokens_a[0, c_token_index],
                tokens_b[0, c_token_index],
            )
        )
        self.assertFalse(
            torch.allclose(
                tokens_a[0, c_token_index],
                unmasked_tokens[0, c_token_index],
            )
        )

    def test_pretrain_masks_valid_shifts_and_removes_rich_inputs(self):
        model = FoMoNMR(
            small_config(
                global_shift_probability=0.0,
                local_jitter_probability=0.0,
                modality_dropout_probability=0.0,
            )
        )
        clean = example_batch()
        original_availability = clean["h"]["availability"].clone()

        torch.manual_seed(4)
        corrupted, info = model.corrupt_batch(clean)

        self.assertTrue((info["h_shift_mask"].sum(dim=1) <= 1).all())
        self.assertTrue((info["c_shift_mask"].sum(dim=1) <= 1).all())
        self.assertFalse(
            (info["h_shift_mask"] & ~clean["h"]["peak_mask"]).any().item()
        )
        self.assertFalse(
            (info["c_shift_mask"] & ~clean["c"]["peak_mask"]).any().item()
        )
        self.assertTrue(
            corrupted["h"]["peak_mask"][info["h_shift_mask"]].all().item()
        )
        self.assertFalse(corrupted["h"]["availability"].any().item())
        self.assertFalse(corrupted["h"]["j_mask"].any().item())
        self.assertTrue(torch.equal(clean["h"]["availability"], original_availability))

        rich_parameters = model.embedding.h_embedder.rich_mlp.parameters()
        self.assertTrue(
            all(not parameter.requires_grad for parameter in rich_parameters)
        )
        posttrain = FoMoNMR(small_config(stage="posttrain"))
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in posttrain.embedding.h_embedder.rich_mlp.parameters()
            )
        )

    def test_modality_dropout_preserves_nonempty_and_single_modal_spectra(self):
        model = FoMoNMR(
            small_config(
                global_shift_probability=0.0,
                local_jitter_probability=0.0,
                modality_dropout_probability=1.0,
            )
        )
        clean = example_batch()
        torch.manual_seed(8)
        corrupted, _ = model.corrupt_batch(clean)

        h_counts = corrupted["h"]["peak_mask"].sum(dim=1)
        c_counts = corrupted["c"]["peak_mask"].sum(dim=1)
        self.assertTrue(((h_counts + c_counts) > 0).all())
        self.assertTrue((h_counts[0] == 0) ^ (c_counts[0] == 0))
        self.assertEqual(h_counts[1].item(), 1)
        self.assertEqual(c_counts[1].item(), 0)
        self.assertEqual(h_counts[2].item(), 0)
        self.assertEqual(c_counts[2].item(), 1)

    def test_global_shift_is_target_but_local_jitter_is_not(self):
        clean = example_batch()
        global_model = FoMoNMR(
            small_config(
                global_shift_probability=1.0,
                local_jitter_probability=0.0,
                modality_dropout_probability=0.0,
            )
        )
        torch.manual_seed(3)
        global_batch, global_info = global_model.corrupt_batch(clean)
        self.assertTrue(
            torch.allclose(
                global_batch["h"]["shift"],
                global_info["h_shift_targets"],
            )
        )
        self.assertTrue(
            torch.allclose(
                global_batch["c"]["shift"],
                global_info["c_shift_targets"],
            )
        )

        boundary_batch = FoundationNMRProcessor()(
            [
                CanonicalRecord(
                    record_id="boundaries",
                    h_nmr_peaks=(ProtonPeak(-5.5), ProtonPeak(20.0)),
                    c_nmr_peaks=(CarbonPeak(-40.0), CarbonPeak(300.0)),
                )
            ]
        )
        torch.manual_seed(3)
        boundary_corrupted, boundary_info = global_model.corrupt_batch(
            boundary_batch
        )
        self.assertTrue(
            torch.equal(
                boundary_corrupted["h"]["shift"],
                boundary_info["h_shift_targets"],
            )
        )
        self.assertTrue(
            torch.equal(
                boundary_corrupted["c"]["shift"],
                boundary_info["c_shift_targets"],
            )
        )
        self.assertTrue(
            (
                (boundary_info["h_shift_targets"] >= -5.5)
                & (boundary_info["h_shift_targets"] <= 20.0)
            ).all()
        )
        self.assertTrue(
            (
                (boundary_info["c_shift_targets"] >= -40.0)
                & (boundary_info["c_shift_targets"] <= 300.0)
            ).all()
        )

        jitter_model = FoMoNMR(
            small_config(
                global_shift_probability=0.0,
                local_jitter_probability=1.0,
                modality_dropout_probability=0.0,
            )
        )
        torch.manual_seed(3)
        jittered, jitter_info = jitter_model.corrupt_batch(clean)
        self.assertTrue(
            torch.equal(jitter_info["h_shift_targets"], clean["h"]["shift"])
        )
        self.assertTrue(
            torch.equal(jitter_info["c_shift_targets"], clean["c"]["shift"])
        )
        self.assertTrue(
            torch.equal(
                jittered["h"]["shift"][jitter_info["h_shift_mask"]],
                clean["h"]["shift"][jitter_info["h_shift_mask"]],
            )
        )
        h_visible = (
            clean["h"]["peak_mask"]
            & ~jitter_info["h_shift_mask"]
        )
        self.assertFalse(
            torch.equal(
                jittered["h"]["shift"][h_visible],
                clean["h"]["shift"][h_visible],
            )
        )

    def test_posttrain_annotation_masks_follow_availability_and_can_overlap(self):
        model = FoMoNMR(
            small_config(
                stage="posttrain",
                global_shift_probability=0.0,
                local_jitter_probability=0.0,
                modality_dropout_probability=0.0,
                annotation_mask_probability=1.0,
            )
        )
        clean = example_batch()
        torch.manual_seed(2)
        corrupted, info = model.corrupt_batch(clean)
        annotation_mask = info["annotation_mask"]

        self.assertTrue(annotation_mask[0, :2, 0].all())
        self.assertTrue(annotation_mask[0, 0, 1].item())
        self.assertFalse(annotation_mask[0, 1, 1].item())
        self.assertTrue(annotation_mask[0, :2, 2].all())
        self.assertTrue(annotation_mask[0, :2, 3].all())
        self.assertFalse(annotation_mask[1].any().item())
        self.assertFalse(
            corrupted["h"]["availability"][annotation_mask].any().item()
        )
        self.assertFalse(
            corrupted["h"]["j_mask"][annotation_mask[:, :, 2]].any().item()
        )
        overlap = annotation_mask.any(dim=2) & info["h_shift_mask"]
        self.assertTrue(overlap.any().item())

    def test_fingerprint_pairs_are_complete_symmetric_and_include_identity_bin(self):
        first = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
        fingerprints = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        features, similarities = symmetric_fingerprint_pairs(first, fingerprints)
        self.assertEqual(features.shape, (3, 4))
        self.assertEqual(similarities.shape, (3,))

        swapped_features, swapped_similarity = symmetric_fingerprint_pairs(
            first[:2].flip(0),
            fingerprints[:2].flip(0),
        )
        original_features, original_similarity = symmetric_fingerprint_pairs(
            first[:2],
            fingerprints[:2],
        )
        self.assertTrue(torch.equal(swapped_features, original_features))
        self.assertTrue(torch.equal(swapped_similarity, original_similarity))

        bins = tanimoto_to_bins(torch.tensor([0.0, 0.999, 1.0]), 0.05, 21)
        self.assertEqual(bins.tolist(), [0, 19, 20])
        model = FoMoNMR(small_config())
        self.assertEqual(model.fp_sim_num_bins, 21)
        self.assertEqual(model.fp_sim_classifier[-1].out_features, 21)

    def test_j_loss_sorts_valid_values_and_ignores_padding(self):
        model = FoMoNMR(small_config(stage="posttrain", lambda_fp=0.0))
        clean = example_batch()
        annotation_mask = torch.zeros_like(clean["h"]["availability"])
        annotation_mask[0, 0, 2] = True

        pooled = torch.zeros(3, model.config.d_model, requires_grad=True)
        peak_states = torch.zeros(
            3,
            MAX_H_PEAKS * 2,
            model.config.d_model,
            requires_grad=True,
        )
        valid_peaks = torch.cat(
            (clean["h"]["peak_mask"], clean["c"]["peak_mask"]),
            dim=1,
        )
        corruption = {
            "h_shift_mask": torch.zeros_like(clean["h"]["peak_mask"]),
            "c_shift_mask": torch.zeros_like(clean["c"]["peak_mask"]),
            "h_shift_targets": clean["h"]["shift"].clone(),
            "c_shift_targets": clean["c"]["shift"].clone(),
            "annotation_mask": annotation_mask,
        }

        with torch.no_grad():
            model.j_value_head.weight.zero_()
            model.j_value_head.bias.copy_(
                torch.tensor([1.0, 7.0, 0.0, 0.0, 0.0, 0.0])
            )

        first_losses = model.compute_losses(
            (pooled, peak_states, valid_peaks),
            clean,
            corruption,
        )
        changed_padding = copy.deepcopy(clean)
        changed_padding["h"]["j_values"][0, 0, 2:] = 999.0
        second_losses = model.compute_losses(
            (pooled, peak_states, valid_peaks),
            changed_padding,
            corruption,
        )

        self.assertEqual(first_losses["j_value"].item(), 0.0)
        self.assertTrue(
            torch.allclose(first_losses["j"], second_losses["j"])
        )

    def test_training_and_validation_steps_return_finite_losses(self):
        for stage in ("pretrain", "posttrain"):
            model = FoMoNMR(
                small_config(
                    stage=stage,
                    annotation_mask_probability=1.0,
                )
            )
            batch = example_batch()
            calls = []
            handle = model.register_forward_hook(
                lambda module, inputs, output: calls.append(1)
            )
            loss = model.training_step(batch, 0)
            handle.remove()

            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(len(calls), 1)

            model.eval()
            with torch.no_grad():
                validation_losses = model.validation_step(batch, 4)
            for value in validation_losses.values():
                self.assertTrue(torch.isfinite(value))

    def test_losses_handle_batch_without_component_targets(self):
        model = FoMoNMR(small_config(lambda_fp=0.0))
        batch = example_batch()
        pooled = torch.zeros(3, model.config.d_model, requires_grad=True)
        peaks = torch.zeros(
            3,
            MAX_H_PEAKS * 2,
            model.config.d_model,
            requires_grad=True,
        )
        valid = torch.cat(
            (batch["h"]["peak_mask"], batch["c"]["peak_mask"]),
            dim=1,
        )
        info = {
            "h_shift_mask": torch.zeros_like(batch["h"]["peak_mask"]),
            "c_shift_mask": torch.zeros_like(batch["c"]["peak_mask"]),
            "h_shift_targets": batch["h"]["shift"].clone(),
            "c_shift_targets": batch["c"]["shift"].clone(),
            "annotation_mask": torch.zeros_like(batch["h"]["availability"]),
        }
        losses = model.compute_losses((pooled, peaks, valid), batch, info)
        self.assertEqual(losses["shift"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))


if __name__ == "__main__":
    unittest.main()
