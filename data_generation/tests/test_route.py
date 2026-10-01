from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from data_generation.tests.fixtures import caption

from data_generation.src.tasks import route


class RouteTests(unittest.TestCase):
    def test_prompt_requires_video_and_caption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caption.json"
            path.write_text(json.dumps(caption(120)))
            with patch.object(route, "probe_video_duration", return_value=120):
                request = route.build_request(path, Path("video.mp4"))
            self.assertEqual(json.loads(request.user_prompt), {"caption": caption(120)})
            self.assertIn("fallible navigation references", request.system_prompt)
            self.assertEqual(len(route.TASK_CATALOG), 12)
            self.assertFalse(any(name.startswith("triggered_") for name in route.TASK_CATALOG))

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
