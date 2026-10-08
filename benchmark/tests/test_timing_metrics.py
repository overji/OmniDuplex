import unittest

from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import match_windows


class TimingMetricsTests(unittest.TestCase):
    def test_boundaries_are_inclusive_and_onsets_are_not_reused(self):
        sample = {"responses": [{"window_start": "00:01", "window_end": "00:02"},
                                {"window_start": "00:02", "window_end": "00:03"},
                                {"role": "tool", "trigger_time": "00:05"}]}
        result = match_windows(sample, [{"onset_seconds": 2}, {"onset_seconds": 3},
                                        {"onset_seconds": 4}])
        self.assertEqual((result["windows"], result["hits"], result["extra_predictions"]), (2, 2, 1))
        self.assertEqual(match_windows(sample, [{"onset_seconds": 2}])["hits"], 1)

    def test_earliest_deadline_preserves_matches_for_overlapping_windows(self):
        sample = {"responses": [{"window_start": "00:01", "window_end": "00:10"},
                                {"window_start": "00:01", "window_end": "00:02"}]}
        self.assertEqual(match_windows(sample, [{"onset_seconds": 1}, {"onset_seconds": 3}])["hits"], 2)
        self.assertEqual(seconds("01:02:03"), 3723)


if __name__ == "__main__":
    unittest.main()
