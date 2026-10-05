from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from data_generation.tests.fixtures import caption

from data_generation.src.domain.tasks import route
from data_generation.src.application import requests


class RouteTests(unittest.TestCase):
    def test_prompt_requires_video_and_caption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caption.json"
            path.write_text(json.dumps(caption(120)))
            with patch.object(requests, "probe_video_duration", return_value=120):
                request = requests.build_task_request("route", path, Path("video.mp4"))
            self.assertEqual(json.loads(request.user_prompt), {"caption": caption(120)})
            self.assertIn("fallible navigation references", request.system_prompt)
            self.assertEqual(len(route.ABILITIES), 12)
            for name in route.ABILITIES:
                self.assertIn(f"- {name}: ", request.system_prompt)
                for mode in ("rtd", "pr", "agent"):
                    self.assertNotIn(f"- {name}_{mode}:", request.system_prompt)
            self.assertIn("at least one valid task", request.system_prompt)
            self.assertNotIn("{{", request.system_prompt)

    def test_route_requires_full_partition_and_consecutive_ranks(self) -> None:
        ranked = [
            {"task": name, "rank": index, "rationale": "supported by the video"}
            for index, name in enumerate(route.ABILITIES, 1)
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
                    {"task": "event_understanding", "rank": 1, "rationale": "clear"}
                ]},
                route.GenerationContext(120),
            )

    def test_mixed_partition(self) -> None:
        result = self.partition()
        self.assertEqual(route.validate_result(result, route.GenerationContext(120)), result)

    def partition(self):
        return {"status": "ok",
                "ranked_tasks": [{"task": "event_understanding", "rank": 1, "rationale": "clear event"}],
                "unsuitable_tasks": [{"task": name, "reason": "no evidence"}
                                     for name in route.ABILITIES if name != "event_understanding"]}

    def test_rejects_mode_specific_ids_in_either_array(self) -> None:
        for field in ("ranked_tasks", "unsuitable_tasks"):
            with self.subTest(field=field):
                result = self.partition()
                result[field][0]["task"] += "_pr"
                with self.assertRaisesRegex(ValueError, "invalid"):
                    route.validate_result(result, route.GenerationContext(120))

    def test_rejects_duplicate_across_arrays(self) -> None:
        result = self.partition()
        result["unsuitable_tasks"][0]["task"] = "event_understanding"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            route.validate_result(result, route.GenerationContext(120))

    def test_rejects_nonconsecutive_rank(self) -> None:
        result = self.partition()
        result["ranked_tasks"][0]["rank"] = 2
        with self.assertRaisesRegex(ValueError, "consecutive"):
            route.validate_result(result, route.GenerationContext(120))

    def test_skip_is_valid(self) -> None:
        self.assertEqual(
            route.validate_result(
                {"status": "skip", "reason": "No task has sufficient evidence."},
                route.GenerationContext(120),
            )["status"],
            "skip",
        )
