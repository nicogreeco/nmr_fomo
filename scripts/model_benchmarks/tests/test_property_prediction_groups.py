"""Small fixtures for molecule-disjoint MLP validation and CV routing."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from model_benchmarks import run_property_prediction as probes


class PropertyPredictionGroupTests(unittest.TestCase):
    def test_internal_holdout_keeps_replicates_together_and_is_repeatable(self):
        groups = np.repeat(np.arange(40), 2)
        for classification in (False, True):
            with self.subTest(classification=classification):
                targets = groups % 2 if classification else groups.astype(float)
                first = probes.split_mlp_train_validation(
                    targets, groups, classification, 0.2, 42,
                )
                second = probes.split_mlp_train_validation(
                    targets, groups, classification, 0.2, 42,
                )
                train, validation = first
                self.assertFalse(set(groups[train]) & set(groups[validation]))
                np.testing.assert_array_equal(
                    np.sort(np.concatenate(first)), np.arange(len(groups)),
                )
                for left, right in zip(first, second):
                    np.testing.assert_array_equal(left, right)
                if classification:
                    self.assertEqual(set(targets[train]), {0, 1})
                    self.assertEqual(set(targets[validation]), {0, 1})

    def test_groups_are_required_and_must_allow_a_holdout(self):
        for groups in (None, [0], [0, 0, 0, 0]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                probes.split_mlp_train_validation(np.arange(4), groups, False, 0.2, 42)

    def test_grid_search_passes_matching_groups_to_each_fold_and_final_refit(self):
        groups = np.repeat(np.arange(40), 2)
        features = np.random.default_rng(7).normal(size=(80, 4))
        split = probes.split_mlp_train_validation
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)

        for classification in (False, True):
            with self.subTest(classification=classification):
                targets = groups % 2 if classification else groups.astype(float)
                seen = []

                def checked_split(y, fold_groups, is_classification, fraction, seed):
                    self.assertEqual(is_classification, classification)
                    expected = fold_groups % 2 if classification else fold_groups
                    np.testing.assert_array_equal(y, expected)
                    train, validation = split(y, fold_groups, classification, fraction, seed)
                    self.assertFalse(set(fold_groups[train]) & set(fold_groups[validation]))
                    seen.append(len(fold_groups))
                    return train, validation

                run = probes.run_classification if classification else probes.run_regression
                with patch.multiple(
                    probes, N_JOBS=1,
                    MLP_REGRESSION_ALPHAS=[0.01],
                    MLP_CLASSIFICATION_ALPHAS=[0.01],
                ), patch.object(probes, 'split_mlp_train_validation', side_effect=checked_split):
                    search, _, metrics = run(
                        'mlp', features, targets, features, targets,
                        groups, folds=2, max_iter=1, device='cpu',
                    )
                self.assertEqual(sorted(seen), [40, 40, 80])
                self.assertTrue(np.isfinite(search.best_score_))
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == '__main__':
    unittest.main()
