from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_generation.src.tasks import route


class RouteTests(unittest.TestCase):
    def test_prompt_requires_video_and_reference_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.json"
            objects = root / "objects.json"
            events.write_text(json.dumps({"audio_events": [], "visual_events": []}))
            objects.write_text(json.dumps({"items": []}))
            request = route.build_request(events, Path("data_generation/videos/test.mp4"), objects_path=objects)
            self.assertIn("fallible navigation references", request.system_prompt)
            self.assertIn("<task_name>", request.system_prompt)
            self.assertIn("may be deliberately proposed", request.system_prompt)
            self.assertIn("not need to be a spontaneous", request.system_prompt)
            self.assertIn("Judge the task as a combination", request.system_prompt)
            self.assertIn("does not need to contain an explicit request", request.system_prompt)

    def test_route_requires_full_partition_and_consecutive_ranks(self) -> None:
        ranked = [
            {"task": name, "rank": index, "rationale": "supported by the video"}
            for index, name in enumerate(route.TASK_CATALOG, 1)
        ]
        result = route.validate_result(
            {"status": "ok", "unsuitable_tasks": [], "ranked_tasks": ranked},
            route.GenerationContext(120),
        )
        self.assertEqual(result["ranked_tasks"][0]["rank"], 1)

    def test_route_rejects_duplicate_or_missing_tasks(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover every"):
            route.validate_result(
                {"status": "ok", "unsuitable_tasks": [], "ranked_tasks": [
                    {"task": "event_alert", "rank": 1, "rationale": "clear"}
                ]},
                route.GenerationContext(120),
            )

    def test_skip_is_valid(self) -> None:
        self.assertEqual(
            route.validate_result(
                {"status": "skip", "reason": "No task has sufficient evidence."},
                route.GenerationContext(120),
            )["status"],
            "skip",
        )
