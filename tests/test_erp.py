"""Behavioural regression tests; run with python -m unittest discover -s tests."""
import unittest
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from config import EVENT_ID
from src.erp.plot_erp import collect_group_erps, verify_p300


def make_epochs(peak=0.4, amplitude=5.0):
    times = -0.2 + np.arange(301) / 250
    pulse = np.exp(-0.5 * ((times - peak) / 0.04) ** 2)
    data = np.zeros((3, 3, 301))
    data[1] = np.array([0.2, 0.6, 1.0])[:, None] * amplitude * pulse * 1e-6
    data[2] = np.array([3.0, 2.0, 1.0])[:, None] * pulse * 1e-6
    return mne.EpochsArray(
        data, mne.create_info(["Fz", "Cz", "Pz"], 250, "eeg"),
        events=np.array([[0, 0, 5], [400, 0, 6], [800, 0, 7]]),
        event_id=EVENT_ID, tmin=-0.2, baseline=None, verbose="error")


class P300VerificationTests(unittest.TestCase):
    def test_late_peak_is_not_clipped_to_acceptance_window(self):
        report = verify_p300(make_epochs(peak=0.7), "test")
        self.assertAlmostEqual(report["metrics"]["target_pz_latency_ms"], 700)
        self.assertFalse(report["polich_rules"]["rule3_target_pz_latency_valid"])
        self.assertFalse(report["polich_rules"]["p300_detected"])

    def test_amplitude_and_latency_are_independent(self):
        report = verify_p300(make_epochs(), "test")
        self.assertTrue(report["polich_rules"]["p300_detected"])
        self.assertGreater(report["metrics"]["target_pz_peak_uv"],
                           report["metrics"]["target_pz_mean_uv"])

    def test_missing_condition_is_an_error(self):
        with self.assertRaises(ValueError):
            verify_p300(make_epochs()[[0, 1]], "test")

    def test_search_must_cover_acceptance_window(self):
        with self.assertRaises(ValueError):
            verify_p300(make_epochs(), "test", peak_search_window=(0.3, 0.5))

    def test_threshold_is_evaluated_before_display_rounding(self):
        base = verify_p300(make_epochs(), "test")["polich_rules"]["rule2_diff_uv"]
        report = verify_p300(make_epochs(amplitude=5 * 1.499 / base), "test")
        self.assertFalse(report["polich_rules"]["rule2_oddball_effect_target_gt_std"])

    def test_600ms_analysis_does_not_change_the_peak_search(self):
        epochs = make_epochs(peak=0.56)
        strict = verify_p300(epochs, "test")
        broad = verify_p300(epochs, "test", window=(0.3, 0.6))
        self.assertFalse(strict["polich_rules"]["rule3_target_pz_latency_valid"])
        self.assertTrue(broad["polich_rules"]["rule3_target_pz_latency_valid"])
        self.assertEqual(strict["polich_rules"]["rule3_latency_ms"],
                         broad["polich_rules"]["rule3_latency_ms"])

    def test_missing_subject_is_not_silently_skipped(self):
        with patch("src.erp.plot_erp.load_subject_epochs", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                collect_group_erps(subjects=["sub-01"])

    def test_table_uses_latency_and_keeps_exclusion_separate(self):
        qc = pd.DataFrame({
            "subject": ["sub-01", "sub-02"], "task_misunderstood": [False, True],
            "target_rt_ms": [500, 500], "false_alarm_distractor_pct": [0, 90],
            "target_omission_pct": [0, 0],
        })
        with patch("src.erp.plot_erp.pd.read_csv", return_value=qc):
            with patch("src.erp.plot_erp.load_subject_epochs", side_effect=[make_epochs(0.7), make_epochs()]):
                table, evokeds = collect_group_erps(subjects=["sub-01", "sub-02"])
        self.assertFalse(table.iloc[0]["p300_detected"])
        self.assertFalse(table.iloc[0]["p300_verified"])
        self.assertTrue(table.iloc[1]["p300_detected"])
        self.assertFalse(table.iloc[1]["p300_verified"])
        self.assertEqual(len(evokeds["target"]), 1)


if __name__ == "__main__":
    unittest.main()
