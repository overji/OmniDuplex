import json
import tempfile
import unittest
from pathlib import Path

from src.merge_qa import build_bundle


class MergeQaTests(unittest.TestCase):
    def test_merges_and_adapts_agent_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            videos_dir = root / "videos"
            folder = data_dir / "test_video"
            folder.mkdir(parents=True)
            videos_dir.mkdir()
            (videos_dir / "test.mp4").touch()
            ordinary = {
                "status": "ok", "question": "What happened?",
                "responses": [{"trigger_time": "00:02", "response": "A car arrived."}],
            }
            agent = {
                "status": "ok", "question": "Look this up.",
                "responses": [{
                    "trigger_time": "00:20", "trigger_type": "visual",
                    "event_description": "Object appears.",
                    "messages": [
                        {"timestamp": "00:10", "role": "assistant", "content": "Checking.",
                         "tool_calls": [{"function": {"name": "agent_call"}}]},
                        {"timestamp": "00:15", "role": "tool", "name": "agent_call",
                         "content": '{"status":"success"}'},
                        {"timestamp": "00:15", "role": "assistant", "content": "Found it."},
                    ],
                }],
            }
            (folder / "a_agent.json").write_text(json.dumps(agent), encoding="utf-8")
            (folder / "b_ordinary.json").write_text(json.dumps(ordinary), encoding="utf-8")
            (folder / "router.json").write_text(
                json.dumps({"status": "ok", "ranked_tasks": []}), encoding="utf-8"
            )
            (folder / "skipped.json").write_text(
                json.dumps({"status": "skip", "reason": "not applicable"}), encoding="utf-8"
            )

            bundle = build_bundle(data_dir, videos_dir)

            self.assertEqual([item["task"] for item in bundle], ["a_agent", "b_ordinary"])
            self.assertEqual(bundle[0]["video"], "test.mp4")
            self.assertEqual(bundle[0]["video_folder"], "test_video")
            responses = bundle[0]["responses"]
            self.assertEqual([item["trigger_time"] for item in responses],
                             ["00:10", "00:15", "00:15"])
            self.assertEqual([item["role"] for item in responses],
                             ["assistant", "tool", "assistant"])
            self.assertEqual(responses[0]["source_trigger_time"], "00:20")
            self.assertEqual(responses[0]["event_description"], "Object appears.")
            self.assertEqual(responses[0]["trigger_type"], "visual")
            self.assertEqual(responses[0]["tool_calls"],
                             [{"function": {"name": "agent_call"}}])
            self.assertEqual(responses[1]["name"], "agent_call")
            self.assertEqual(responses[1]["response"], '{"status":"success"}')
            self.assertNotIn("messages", responses[0])
            self.assertNotIn("event_description", responses[1])
            self.assertEqual(bundle[1]["responses"], ordinary["responses"])
            self.assertEqual(json.loads((folder / "a_agent.json").read_text()), agent)

    def test_missing_video_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "data" / "video_550"
            folder.mkdir(parents=True)
            (root / "videos").mkdir()
            (folder / "event_alert.json").write_text(json.dumps({
                "status": "ok", "question": "What happened?", "responses": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no video found"):
                build_bundle(root / "data", root / "videos")


if __name__ == "__main__":
    unittest.main()
