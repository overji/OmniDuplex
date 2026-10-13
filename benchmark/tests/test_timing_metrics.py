import unittest

from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import match_windows, reference_windows, summarize


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


    def test_agent_scores_completion_windows_but_not_tool_inputs(self):
        def response(role, start, call=False):
            result = {"role": role, "window_start": f"00:{start:02d}",
                      "window_end": f"00:{start + 1:02d}"}
            if call:
                result["tool_calls"] = [{"function": {"name": "agent_call"}}]
            return result
        sample = {"mode": "agent", "task": "delegation", "ability": "trigger", "responses": [
            response("assistant", 1, True), response("tool", 3),
            response("assistant", 5), response("assistant", 7, True),
            {**response("assistant", 9), "tool_calls": [{"function": {"name": "other_tool"}}]},
        ]}
        events = [{"onset_seconds": t} for t in [1, 3, 5, 7, 9]]
        score = match_windows(sample, events)
        self.assertEqual([i for i, _ in reference_windows(sample)], [0, 2, 3, 4])
        self.assertEqual([m["response_index"] for m in score["matches"]], [0, 2, 3, 4])
        self.assertEqual((score["windows"], score["hits"], score["predictions"],
                          score["extra_predictions"]), (4, 4, 5, 1))
        metrics = summarize([sample], [{"sample_index": 0, "events": events}])
        self.assertEqual(metrics["excluded_agent_non_trigger_responses"], 1)
        self.assertEqual(metrics["groups"]["overall"]["timing_precision"], 0.8)
        # Collection reports must use the same filtered windows as matching.
        import json
        import tempfile
        from pathlib import Path
        from benchmark.evaluation.collection import collect
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            prediction = {"sample_index": 0, "events": [dict(e, text="prediction") for e in events]}
            (run / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
            report_sample = dict(sample, question="Q", question_time="00:00", video="clip.mp4")
            collect([report_sample], [run], root / "collected")
            report = (root / "collected" / "report.md").read_text()
            self.assertIn("00:01–00:02", report)
            self.assertIn("00:07–00:08", report)
            self.assertNotIn("00:03–00:04", report)
            self.assertIn("00:05–00:06", report)

    def test_completion_requires_strictly_later_onset_and_existing_window(self):
        sample = {"mode": "agent", "responses": [
            {"role": "tool", "trigger_time": "00:03", "response": "result"},
            {"role": "assistant", "trigger_time": "00:03", "window_start": "00:03",
             "window_end": "00:05", "response": "report result"},
            {"role": "assistant", "trigger_time": "00:06", "response": "no window"},
        ]}
        self.assertEqual(match_windows(sample, [{"onset_seconds": 3}])["hits"], 0)
        self.assertEqual(match_windows(sample, [{"onset_seconds": 4}])["hits"], 1)
        self.assertEqual(match_windows(sample, [{"onset_seconds": 6}])["hits"], 0)
        self.assertEqual(len(reference_windows(sample)), 1)


    def test_agent_negative_refusals_and_other_modes_remain_scored(self):
        response = {"window_start": "00:01", "window_end": "00:02"}
        for sample in [
            {"mode": "agent", "sample_type": "missing_tool"},
            {"mode": "agent", "generation_context": {"sample_type": "no_context"}},
            {"mode": "rtd"}, {"mode": "pr"},
        ]:
            with self.subTest(sample=sample):
                sample["responses"] = [response]
                self.assertEqual(match_windows(sample, [{"onset_seconds": 1}])["hits"], 1)


if __name__ == "__main__":
    unittest.main()
