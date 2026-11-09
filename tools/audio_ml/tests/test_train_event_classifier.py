"""Regression tests for classifier labels and temporal window boundaries."""
import unittest

from train_event_classifier import FRAMES, HOP, SR, WINDOW_SAMPLES, segment_window_range


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

    def test_invalid_bounds_are_rejected(self):
        for start, end in ((float("nan"), 1.0), (0.0, float("inf")),
                           (-0.1, 1.0), (1.0, 1.0), (2.0, 1.0)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    segment_window_range(start, end, 100)


if __name__ == "__main__":
    unittest.main()
