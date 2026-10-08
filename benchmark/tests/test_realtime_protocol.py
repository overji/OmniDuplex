import copy
import unittest

from benchmark.data.prompts import build_system_prompt, user_request


class RealtimeProtocolTests(unittest.TestCase):
    def test_question_input_does_not_depend_on_reference_answers_or_windows(self):
        sample = {
            'question': 'Tell me when the cup moves.', 'question_time': '00:12',
            'frontend_context': {'system_prompt': 'Available application context.'},
            'available_tools': ['note_write'],
            'responses': [{'response': 'secret reference', 'window_start': '00:14',
                           'window_end': '00:18', 'evidence': [{'fact': 'secret evidence'}]}],
            'generation_context': {'hidden_reference': 'not user context'},
        }
        changed = copy.deepcopy(sample)
        changed['responses'] = [{'response': 'different answer', 'window_start': '03:00'}]
        changed['generation_context'] = {}
        self.assertEqual(user_request(sample), user_request(changed))
        self.assertEqual(build_system_prompt('Base', sample), build_system_prompt('Base', changed))
        self.assertIn(sample['question'], user_request(sample))
        self.assertNotIn('note_write', user_request(sample))
        self.assertIn('note_write', build_system_prompt('Base', sample))
        self.assertNotIn('secret', build_system_prompt('Base', sample))

    def test_only_available_tools_and_agent_call_schema_are_in_system_prompt(self):
        sample = {'available_tools': ['note_write'], 'frontend_context': {'tools': [
            {'type': 'function', 'function': {'name': 'agent_call', 'parameters': {}}},
            {'type': 'function', 'function': {'name': 'unavailable_tool'}},
        ]}}
        prompt = build_system_prompt('Base', sample)
        self.assertIn('note_write', prompt)
        self.assertIn('agent_call', prompt)
        self.assertNotIn('unavailable_tool', prompt)
        sample['available_tools'] = []
        self.assertEqual(build_system_prompt('Base', sample), 'Base')


if __name__ == '__main__':
    unittest.main()
