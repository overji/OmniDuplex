"""Regression checks for observed failures in the first full suite."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from benchmark.adapters.gander import StreamingThinker


class RuntimeRepairTests(unittest.TestCase):
    def test_json_artifacts_work_under_ascii_locale(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            code = """import sys
from pathlib import Path
from benchmark.data.files import write_json,append_jsonl
p=Path(sys.argv[1]);write_json(p/'data.json',{'text':'中文—'});append_jsonl(p/'rows.jsonl',{'text':'中文—'})
assert '中文—' in (p/'data.json').read_text(encoding='utf-8')
assert '中文—' in (p/'rows.jsonl').read_text(encoding='utf-8')
"""
            import os
            env = {**os.environ, 'LC_ALL': 'C', 'LANG': 'C', 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'}
            subprocess.run([sys.executable, '-c', code.encode('ascii', 'backslashreplace').decode(), directory], cwd=root, env=env, check=True)

    def test_gander_reuses_runner_and_resets_each_session(self):
        native = SimpleNamespace(prepare=Mock(), duplex=SimpleNamespace(force_listen_count=None))
        factory = Mock(return_value=native)
        wrapper = StreamingThinker(object(), object(), factory)
        wrapper.force_listen_count = 2
        wrapper.tools = [{'name': 'first'}]
        wrapper.prepare('first prompt')
        wrapper.force_listen_count = 7
        wrapper.tools = []
        wrapper.prepare('second prompt')
        factory.assert_called_once()
        self.assertEqual(native.prepare.call_count, 2)
        native.prepare.assert_called_with(system_prompt='second prompt', tools=[])
        self.assertEqual(native.duplex.force_listen_count, 7)


if __name__ == '__main__':
    unittest.main()
