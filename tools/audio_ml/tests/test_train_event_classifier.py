"""Regression tests for classifier labels and temporal window boundaries."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from train_event_classifier import (
    CONTEXT_MS, FRAMES, HOP, SR, WINDOW_SAMPLES, features, positive_int, segment_window_range, split_by_recording,
)


class EventClassifierWindowTests(unittest.TestCase):
    def test_windows_never_cross_segment_boundaries(self):
        feature_count = 100
        start, end = 0.25, 0.9
        first, last = segment_window_range(start, end, feature_count)
        self.assertGreaterEqual(first * HOP, int(start * SR))
        for position in range(first, last, 1):
            window_start = position * HOP
            window_end = window_start + WINDOW_SAMPLES
            self.assertGreaterEqual(window_start, int(start * SR))
            self.assertLessEqual(window_end, int(end * SR))

    def test_short_segment_produces_no_partial_context_window(self):
        first, last = segment_window_range(0.0, (WINDOW_SAMPLES - 1) / SR, 100)
        self.assertEqual(first, last)

    def test_exactly_one_full_context_window_is_included(self):
        duration = WINDOW_SAMPLES / SR
        first, last = segment_window_range(0.0, duration, FRAMES)
        self.assertEqual((first, last), (0, 1))

    def test_fractional_boundaries_round_inward(self):
        start = 0.001
        end = (WINDOW_SAMPLES + 1) / SR
        first, last = segment_window_range(start, end, 100)
        self.assertGreaterEqual(first * HOP, start * SR)
        self.assertLessEqual((last - 1) * HOP + WINDOW_SAMPLES, end * SR)

    def test_features_do_not_depend_on_future_audio(self):
        rng = np.random.default_rng(7)
        prefix = rng.standard_normal(SR).astype(np.float32) * 0.02
        extended = np.concatenate((prefix, np.full(SR, 0.25, np.float32)))
        with tempfile.TemporaryDirectory() as directory:
            short_path = Path(directory) / "short.wav"
            long_path = Path(directory) / "long.wav"
            sf.write(short_path, prefix, SR)
            sf.write(long_path, extended, SR)
            short_features, _ = features(short_path)
            long_features, _ = features(long_path)
        np.testing.assert_allclose(short_features, long_features[:len(short_features)],
                                   rtol=1e-5, atol=1e-5)

    def test_epochs_must_be_positive_integers(self):
        self.assertEqual(positive_int("1"), 1)
        for value in ("0", "-2", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(Exception):
                    positive_int(value)

    def test_context_duration_is_reported_as_342_ms(self):
        self.assertAlmostEqual(CONTEXT_MS, 342.0, places=5)

    def test_recording_split_is_disjoint_and_deterministic(self):
        groups = np.repeat(np.asarray([f"rec_{i}" for i in range(8)]), 2)
        targets = np.tile(np.eye(4, dtype=np.float32), (4, 1))
        train_a, valid_a = split_by_recording(groups, targets, 123)
        train_b, valid_b = split_by_recording(groups, targets, 123)
        train_groups = set(groups[train_a])
        valid_groups = set(groups[valid_a])
        self.assertTrue(train_groups.isdisjoint(valid_groups))
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(valid_a, valid_b)
        self.assertEqual(set(np.flatnonzero(targets[train_a].sum(axis=0) > 0)), set(range(4)))
        self.assertEqual(set(np.flatnonzero(targets[valid_a].sum(axis=0) > 0)), set(range(4)))

    def test_invalid_bounds_are_rejected(self):
        for start, end in ((float("nan"), 1.0), (0.0, float("inf")),
                           (-0.1, 1.0), (1.0, 1.0), (2.0, 1.0)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    segment_window_range(start, end, 100)


if __name__ == "__main__":
    unittest.main()
