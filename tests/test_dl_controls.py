"""Behavioural checks for condition labels, participant folds and explanations."""
import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class EEGNetControlTests(unittest.TestCase):
    def test_distractor_is_positive_and_target_is_rejected(self):
        from src.dl.report_controls import task_labels
        events = np.array([[0, 0, 5], [301, 0, 7], [602, 0, 5]])
        np.testing.assert_array_equal(task_labels(events, "distractor"), [0, 1, 0])
        with self.assertRaises(ValueError):
            task_labels(events, "target")

    def test_changed_trial_counts_preserve_original_test_participants(self):
        from src.dl.report_controls import matched_splits
        from src.dl.train_eegnet import get_splits
        reference_groups = np.repeat(np.arange(10), np.arange(3, 13))
        labels = np.arange(len(reference_groups)) % 2
        changed_groups = np.repeat(np.arange(10), np.arange(14, 4, -1))
        original, _ = get_splits(labels, reference_groups)
        changed = matched_splits(changed_groups, labels, reference_groups)
        visits = np.zeros(len(changed_groups), dtype=int)
        for (_, original_test), (train, test) in zip(original, changed):
            np.testing.assert_array_equal(np.unique(reference_groups[original_test]),
                                          np.unique(changed_groups[test]))
            self.assertEqual(np.intersect1d(changed_groups[train], changed_groups[test]).size, 0)
            visits[test] += 1
        np.testing.assert_array_equal(visits, np.ones(len(visits), dtype=int))
        with self.assertRaises(ValueError):
            matched_splits(changed_groups[changed_groups != 9], labels, reference_groups)

    def test_small_validation_batches_support_prediction_and_saliency(self):
        from src.dl.train_eegnet import train_eegnet, predict_proba, saliency_map
        import torch
        X = np.random.default_rng(42).normal(size=(12, 3, 301)).astype(np.float32)
        y = np.tile([0, 1], 6)
        model, history, scale = train_eegnet(
            X[:8], y[:8], X[8:], y[8:], epochs=1, batch_size=4,
            eval_batch_size=2, device=torch.device("cpu"))
        probs = predict_proba(model, X[8:], scale, batch_size=2)
        sal = saliency_map(model, X[8:][y[8:] == 1], scale, batch_size=1)
        self.assertEqual(len(history["val_loss"]), 1)
        self.assertEqual(probs.shape, (4,))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())
        self.assertEqual(sal.shape, (3, 301))
        self.assertTrue(np.isfinite(sal).all())
        self.assertGreater(sal.sum(), 0)


if __name__ == "__main__":
    unittest.main()
