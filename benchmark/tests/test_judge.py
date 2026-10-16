import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark.cli.main import build_parser
from benchmark.data.agent_inputs import AGENT_INPUT_PROTOCOL, input_protocol
from benchmark.evaluation.agent_calls import parse_event
from benchmark.evaluation.judge import prepare_rows, evaluate, summarize, validate_answer, execute
from benchmark.runtime.runner import SampleContext


def sample(mode='pr', kind=None):
    s = {'task': 'test', 'video': 'v.mp4', 'question': 'Notify me', 'question_time': '00:00',
         'mode': mode, 'ability': 'test', 'responses': [
             {'response': 'Here', 'window_start': '00:02', 'window_end': '00:04'}]}
    if mode == 'agent':
        s['sample_type'] = kind or 'positive'
        s['responses'][0].update(role='assistant', tool_calls=[
            {'function': {'name': 'agent_call', 'arguments': '{"instruction":"do it"}'}}])
    return s


def prediction(s, text='Here', onset=3):
    return {**{k: s[k] for k in ('task', 'video', 'question')}, 'sample_index': 0,
            'status': 'complete', 'events': [{'onset_seconds': onset, 'text': text}]}


def answer(call=None):
    return json.dumps(dict(response_score=1.5, call_score=call, response_reason='correct',
                           call_reason='ok', error_types=[]))


class JudgeTests(unittest.TestCase):
    def test_five_scores_and_invalid_outputs(self):
        for score in (0, .5, 1, 1.5, 2):
            a = json.loads(answer()); a['response_score'] = score
            self.assertEqual(validate_answer(json.dumps(a), False)['response_score'], score)
        for score in (True, -1, 1.2, 3, '2', None):
            a = json.loads(answer()); a['response_score'] = score
            with self.assertRaises(ValueError): validate_answer(json.dumps(a), False)
        with self.assertRaises(ValueError): validate_answer(answer(2), False)
        with self.assertRaises(ValueError): validate_answer(answer(), True)

    def test_null_reason_only_allowed_when_call_not_applicable(self):
        obj = json.loads(answer())
        obj['call_reason'] = None
        self.assertEqual(validate_answer(json.dumps(obj), False)['call_reason'], 'Not applicable')
        obj['call_score'] = 2
        with self.assertRaises(ValueError):
            validate_answer(json.dumps(obj), True)

    def test_agent_subgroups_exclude_positive_pr(self):
        s = sample(); s['sample_type'] = 'positive'
        rows, negatives = prepare_rows([s], [prediction(s)])
        result = evaluate(rows[0], {'retries': 0}, '', '', lambda *a: (answer(), {}, True))
        self.assertNotIn('agent:positive', summarize([result], negatives, 1, 1)['groups'])

    def test_calls_not_inferred(self):
        self.assertEqual(parse_event({'text': 'I will delegate this'})['calls'], [])
        text = json.dumps({'response': 'Doing it', 'tool_calls': [
            {'function': {'name': 'agent_call', 'arguments': '{"instruction":"do it"}'}}]})
        p = parse_event({'text': '```json\n' + text + '\n```'})
        self.assertEqual(p['response'], 'Doing it')
        self.assertEqual(p['calls'][0]['function']['arguments'], {'instruction': 'do it'})
        self.assertEqual(parse_event({'text': '{"tool_calls":'})['status'], 'malformed')

    def test_matching_exclusions_and_no_future_results(self):
        s = sample('agent')
        s['responses'].append({'role': 'tool', 'response': 'FUTURE', 'window_start': '00:05', 'window_end': '00:07'})
        rows, _ = prepare_rows([s], [prediction(s)])
        self.assertEqual(len(rows), 1)
        self.assertNotIn('FUTURE', json.dumps(rows))
        self.assertFalse(rows[0]['payload']['judge_call'])
        result = evaluate(rows[0], {'retries': 0}, '', '', lambda *a: (answer(), {}, True))
        self.assertEqual((result['response_score'], result['call_score']), (1.5, 0))
        rows, _ = prepare_rows([s], [prediction(s, onset=5)])
        with patch('benchmark.evaluation.judge.completion', side_effect=AssertionError):
            result = evaluate(rows[0], {'retries': 0}, '', '')
        self.assertEqual(result['response_score'], 0)

    def test_negative_checks_outside_window_and_partial_exclusion(self):
        s = sample('agent', 'missing_tool')
        text = json.dumps({'response': 'Oops', 'tool_calls': [{'function': {
            'name': 'agent_call', 'arguments': {}}}]})
        rows, negatives = prepare_rows([s], [prediction(s, text, 10)])
        self.assertTrue(negatives[0]['erroneous_call'])
        result = evaluate(rows[0], {'retries': 0}, '', '')
        self.assertIsNone(result['call_score'])
        p = prediction(s); p['status'] = 'partial'
        self.assertEqual(prepare_rows([s], [p]), ([], []))

    def test_failure_is_not_zero_and_usage_is_kept(self):
        s = sample(); rows, neg = prepare_rows([s], [prediction(s)])
        with patch('benchmark.evaluation.judge.time.sleep'):
            result = evaluate(rows[0], {'retries': 1}, '', '', lambda *a: ('bad', {'total_tokens': 10}, True))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['usage_attempts']), 2)
        metrics = summarize([result], neg, 1, 1)
        self.assertEqual(metrics['status'], 'incomplete')
        self.assertIsNone(metrics['groups']['overall']['response_mean_all_windows'])

    def test_resume_and_identity(self):
        with TemporaryDirectory() as d:
            root = Path(d); run = root / 'model'; run.mkdir()
            s = sample(); (root / 'data.json').write_text(json.dumps([s]))
            (run / 'predictions.jsonl').write_text(json.dumps(prediction(s)) + '\n')
            args = build_parser().parse_args(['judge', str(run), '--data', str(root / 'data.json'),
                '--output-dir', str(root / 'judge'), '--provider', 'openai', '--model', 'test'])
            with patch('benchmark.evaluation.judge.resolve_config', return_value=('secret', 'https://example.test')):
                with patch('benchmark.evaluation.judge.evaluate', wraps=lambda row, *a: evaluate(
                        row, {'retries': 0}, '', '', lambda *a: (answer(), {'total_tokens': 10}, True))) as mock:
                    execute(args); execute(args)
                    self.assertEqual(mock.call_count, 1)
                args.model = 'changed'
                with self.assertRaises(ValueError): execute(args)
            self.assertNotIn('secret', (root / 'judge/model/judge_config.json').read_text())

    def test_actual_call_scores_and_two_denominators(self):
        s = sample('agent')
        s['responses'].append({**s['responses'][0], 'window_start': '00:08', 'window_end': '00:09'})
        text = json.dumps({'response': 'Here', 'tool_calls': [{'function': {
            'name': 'agent_call', 'arguments': {'instruction': 'do it'}}}]})
        rows, negatives = prepare_rows([s], [prediction(s, text)])
        results = [evaluate(row, {'retries': 0}, '', '', lambda *a: (answer(2), {}, True)) for row in rows]
        self.assertEqual(results[0]['call_score'], 2)
        m = summarize(results, negatives, 2, 1)['groups']['overall']
        self.assertEqual(m['response_mean_matched'], 1.5)
        self.assertEqual(m['response_mean_all_windows'], .75)
        self.assertEqual(m['call_mean_all_windows'], 1)

    def test_retry_success_keeps_attempts(self):
        s = sample(); rows, _ = prepare_rows([s], [prediction(s)])
        request = unittest.mock.Mock(side_effect=[('invalid', {'total_tokens': 3}, True),
                                                (answer(), {'total_tokens': 5}, True)])
        with patch('benchmark.evaluation.judge.time.sleep'):
            result = evaluate(rows[0], {'retries': 1}, '', '', request)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['usage_attempts']), 2)

    def test_comparison_includes_content_without_changing_timing(self):
        from benchmark.evaluation.comparison import execute as compare
        from benchmark.evaluation.timing import summarize as timing
        with TemporaryDirectory() as d:
            root = Path(d); run = root / 'model'; run.mkdir()
            judged = root / 'judge/model'; judged.mkdir(parents=True)
            s = sample(); pred = prediction(s)
            (run / 'metrics.json').write_text(json.dumps(timing([s], [pred])))
            rows, negatives = prepare_rows([s], [pred])
            result = evaluate(rows[0], {'retries': 0}, '', '', lambda *a: (answer(), {}, True))
            (judged / 'judge_metrics.json').write_text(json.dumps(summarize([result], negatives, 1, 1)))
            compare(SimpleNamespace(runs=[run], output=root / 'comparison', judge_dir=root / 'judge'))
            self.assertIn('1.500', (root / 'comparison.md').read_text())
            self.assertIn('100.00%', (root / 'comparison.md').read_text())

    def test_structured_protocol_is_opt_in(self):
        self.assertEqual(input_protocol(SimpleNamespace()), AGENT_INPUT_PROTOCOL)
        self.assertNotEqual(input_protocol(SimpleNamespace(agent_output_protocol='structured')), AGENT_INPUT_PROTOCOL)
        with TemporaryDirectory() as d:
            context = SampleContext(0, sample('agent'), 'base', Path(d))
            original = context.build_prompt()
            context.agent_output_protocol = 'structured'
            structured = context.build_prompt()
            self.assertTrue(structured.startswith(original))
            self.assertIn('one complete JSON object', structured)


if __name__ == '__main__':
    unittest.main()
