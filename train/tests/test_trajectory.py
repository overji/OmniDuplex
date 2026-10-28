import copy
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from train.src.domain.records import qa_record
from train.src.application.prepare import prepare
from train.src.domain.agent import RECORD_FORMAT
from train.src.domain.trajectory import compile_trajectory, second_target, TrajectoryError
from train.src.models.streaming import replay, SequenceTooLong
from train.src.models.graph import TrainGraph
from train.tests.helpers import ByteTokenizer
from train.tests.helpers import TinyMedia, TinyDuplex


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'v.mp4').touch()
        self.tokenizer = ByteTokenizer()
        self.config = dict(videos_dir=str(self.root), media_cache=str(self.root/'cache'), max_prefix_seconds=300, max_sequence_tokens=32768,
            max_speak_tokens_per_chunk=8, supervise_silence=True, agent_output_protocol='task_start_token_sft_v1',
            seed=42, decision_weight=1., content_weight=1., call_weight=1., loss_projection_chunk=8)
        self.sample = dict(mode='rtd', video='v.mp4', question_time='00:01', responses=[
            dict(response='abcdefgh', window_start='00:03', window_end='00:05', trigger_time='00:03'),
            dict(response='XYZ', window_start='00:09', window_end='00:11', trigger_time='00:10')])

    def record(self):
        row, reason = qa_record(self.sample, 7, self.config)
        self.assertIsNone(reason)
        return row

    def test_all_responses_chunked_with_causal_order_and_no_source_mutation(self):
        before = copy.deepcopy(self.sample)
        schedule = compile_trajectory(self.record(), self.tokenizer, self.config)
        self.assertEqual(self.sample, before)
        self.assertEqual(schedule['response_count'], 2)
        self.assertEqual([(t['onset_seconds'], t['end_seconds']) for t in schedule['targets']], [(3, 4), (10, 10)])
        text = bytes(t for clock in sorted(schedule['chunks']) for t,g in zip(*schedule['chunks'][clock]) if g==2).decode()
        self.assertEqual(text, 'abcdefghXYZ')
        self.assertEqual(sum(ids.count(303) for ids,_ in schedule['chunks'].values()), 2)

    def test_listen_between_replies_and_mask_entire_reference_window(self):
        schedule = compile_trajectory(self.record(), self.tokenizer, self.config)
        for second, sent, expected in [(1,False,0),(2,True,1),(5,True,0),(6,True,1),(9,True,0),(11,True,0)]:
            ids, groups = second_target(schedule, second, self.tokenizer, self.config, sent)
            self.assertEqual((ids,groups),([302],[expected]))
        self.assertEqual(second_target(schedule,6,self.tokenizer,dict(self.config,supervise_silence=False),True)[1],[0])

    def test_optional_tail_starts_after_closed_windows_and_is_bounded(self):
        config = dict(self.config, post_response_silence_seconds=10)
        record = self.record()
        original = compile_trajectory(record, self.tokenizer, self.config)
        self.assertEqual(original['horizon'], 10)
        schedule = compile_trajectory(record, self.tokenizer, config)
        self.assertEqual(schedule['targets'], original['targets'])
        self.assertEqual(schedule['horizon'], 21)
        self.assertEqual(schedule['tail_silence_seconds'], 10)
        self.assertEqual(second_target(schedule, 11, self.tokenizer, config, True)[1], [0])
        self.assertEqual(second_target(schedule, 12, self.tokenizer, config, True), ([302], [1]))
        short = compile_trajectory(record, self.tokenizer, config, duration=15.9)
        self.assertEqual((short['horizon'], short['tail_silence_seconds']), (15, 4))
        capped = compile_trajectory(record, self.tokenizer, dict(config, max_prefix_seconds=12))
        self.assertEqual((capped['horizon'], capped['tail_silence_seconds']), (12, 1))
        no_room = compile_trajectory(record, self.tokenizer, config, duration=10.9)
        self.assertEqual((no_room['horizon'], no_room['tail_silence_seconds']), (10, 0))
        with self.assertRaisesRegex(TrajectoryError, 'target_exceeds_video_duration'):
            compile_trajectory(record, self.tokenizer, config, duration=9.9)

    @patch('train.src.models.streaming.user_request', return_value='question')
    @patch('train.src.models.streaming.system_prompt', return_value='system')
    def test_tail_replay_preserves_all_content_and_trims_only_optional_token_overflow(self, *_):
        model = TinyMedia(); duplex = TinyDuplex(model)
        record = self.record()
        def clip(end):
            return SimpleNamespace(chunks=[{}] * end, question_time=1, horizon=end, duration=15,
                                   preprocessing_cache=lambda p: nullcontext())
        baseline = replay(model, duplex, record, clip(10), self.config)
        config = dict(self.config, post_response_silence_seconds=10)
        complete = replay(model, duplex, record, clip(15), config)
        self.assertEqual(complete['tail_silence_seconds'], 4)
        self.assertEqual(complete['labels'][complete['groups'] == 2].tolist(), list(b'abcdefghXYZ'))
        limited = replay(model, duplex, record, clip(15),
                         dict(config, max_sequence_tokens=len(baseline['labels']) + 12))
        self.assertEqual(len(limited['labels']), len(baseline['labels']) + 12)
        self.assertEqual(limited['tail_silence_seconds'], 2)
        self.assertEqual(limited['labels'][limited['groups'] == 2].tolist(), list(b'abcdefghXYZ'))
        self.assertEqual(limited['labels'][-2:].tolist(), [302, -100])
        with self.assertRaises(SequenceTooLong):
            replay(model, duplex, record, clip(15), dict(config, max_sequence_tokens=35))

    def test_conflict_moves_onset_only_inside_unchanged_window(self):
        self.sample['responses'][1].update(window_start='00:03',window_end='00:06',trigger_time='00:03')
        schedule = compile_trajectory(self.record(),self.tokenizer,self.config)
        self.assertEqual(schedule['targets'][1]['onset_seconds'],5)
        self.sample['responses'][1]['window_end']='00:04'
        with self.assertRaisesRegex(TrajectoryError,'response_schedule_conflict:response_index=1'):
            compile_trajectory(self.record(),self.tokenizer,self.config)

    def test_rejects_partial_duplicate_and_old_records_and_video_overrun(self):
        record=self.record()
        with self.assertRaisesRegex(TrajectoryError,'target_exceeds_video_duration:response_index=1'):
            compile_trajectory(record,self.tokenizer,self.config,duration=9.9)
        with self.assertRaisesRegex(ValueError,'every scored response'):
            compile_trajectory(dict(record,targets=record['targets'][:1]),self.tokenizer,self.config)
        with self.assertRaisesRegex(ValueError,'every scored response'):
            compile_trajectory(dict(record,targets=record['targets']*2),self.tokenizer,self.config)
        with self.assertRaisesRegex(ValueError,'prepare again'):
            compile_trajectory({'sample':self.sample},self.tokenizer,self.config)
        model=TinyMedia();duplex=TinyDuplex(model)
        with self.assertRaisesRegex(TrajectoryError,'clip_does_not_cover_complete_trajectory'):
            replay(model,duplex,record,SimpleNamespace(chunks=[{}]*9,duration=15),self.config)

    def test_prepare_counts_all_responses_and_keeps_videos_disjoint(self):
        samples=[]
        for i in range(10):
            sample=copy.deepcopy(self.sample);sample['video']=f'{i}.mp4'
            (self.root/sample['video']).touch();samples.append(sample)
        source=self.root/'qa.json';source.write_text(json.dumps(samples))
        output=self.root/'prepared';prepare(source,output,self.config,8,.2)
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['record_format'],RECORD_FORMAT)
        self.assertEqual(manifest['response_counts'],{'train':16,'validation':4})
        self.assertFalse(set(manifest['videos']['train']) & set(manifest['videos']['validation']))
        self.assertTrue(all(len(r['targets'])==2 for r in json.loads((output/'train.json').read_text())))

    def test_later_noncausal_or_overlong_response_rejects_entire_qa(self):
        self.sample['responses'][1]['evidence']=[{'available_time':'00:12'}]
        self.assertEqual(qa_record(self.sample,0,self.config)[1],'no_causal_integer_onset:response_index=1')
        self.sample['responses'][1].pop('evidence')
        self.assertEqual(qa_record(self.sample,0,dict(self.config,max_prefix_seconds=5))[1],'prefix_too_long:response_index=1')

    def test_agent_targets_exclude_tool_return_and_completion(self):
        call=dict(type='function',function=dict(name='agent_call',arguments={'instruction':'find a cap'}))
        self.sample['mode']='agent'
        for ref in self.sample['responses']:
            ref.update(role='assistant',tool_calls=[copy.deepcopy(call)],response='Checking.')
            ref['window_end']='00:30'
        self.sample['responses'].insert(1,dict(role='tool',response='SECRET TOOL RESULT'))
        self.sample['responses'].append(dict(role='assistant',response='SECRET COMPLETION',window_start='00:25',window_end='00:30'))
        schedule=compile_trajectory(self.record(),self.tokenizer,self.config)
        self.assertEqual([t['response_index'] for t in schedule['targets']],[0,2])
        text=self.tokenizer.decode([t for ids,gs in schedule['chunks'].values() for t,g in zip(ids,gs) if g and (t<256 or t==300)],skip_special_tokens=False)
        self.assertEqual(text.count('<|task_start|>'),2)
        self.assertNotIn('SECRET',text)
        self.sample['sample_type']='missing_tool'
        self.assertEqual(qa_record(self.sample,0,self.config)[1],'agent_negative')

    @patch('train.src.models.streaming.user_request',return_value='question')
    @patch('train.src.models.streaming.system_prompt',return_value='system')
    def test_real_replay_is_causal_and_has_all_labels_and_gradients(self,*_):
        model=TinyMedia();duplex=TinyDuplex(model)
        original_decoder=duplex.decoder
        record=self.record();schedule=compile_trajectory(record,self.tokenizer,self.config)
        clip=SimpleNamespace(chunks=[{} for _ in range(schedule['horizon'])],question_time=1,
                             horizon=schedule['horizon'],duration=15,preprocessing_cache=lambda p:nullcontext())
        encoded=replay(model,duplex,record,clip,self.config)
        self.assertIs(duplex.decoder,original_decoder)
        self.assertEqual(encoded['response_count'],2)
        self.assertEqual(encoded['labels'][encoded['groups']==2].tolist(),list(b'abcdefghXYZ'))
        # ByteTokenizer encodes one input position per token. Infer the exact
        # media/text order from embeddings, including masked history tokens.
        expected=list(b'<|im_start|>system\nsystem<|im_end|>');labels=[-100]*len(expected)
        for second in range(1,schedule['horizon']+1):
            if second==2:expected+=list(b'question');labels+=[-100]*8
            expected += [None,None];labels += [-100,-100]  # this second's vision/audio first
            ids,gs=second_target(schedule,second,self.tokenizer,self.config,second>=2)
            expected+=ids+[305];labels += [token if g else -100 for token,g in zip(ids,gs)]+[-100]
        self.assertEqual(encoded['labels'].tolist(),labels)
        for i,token in enumerate(expected):
            if token is not None:
                torch.testing.assert_close(encoded['inputs_embeds'][i],model.llm.get_input_embeddings().weight[token])
        graph=TrainGraph(model,duplex,self.config)
        graph(record,clip).backward()
        self.assertEqual(graph.last_stats['response_count'],2)
        for module in ('llm','vpm','apm','resampler','audio_projection_layer'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,module).parameters()))

    @patch('train.src.models.streaming.user_request',return_value='question')
    @patch('train.src.models.streaming.system_prompt',return_value='system')
    def test_token_overflow_does_not_return_partial_replies(self,*_):
        model=TinyMedia();duplex=TinyDuplex(model);decoder=duplex.decoder
        record=self.record()
        clip=SimpleNamespace(chunks=[{}]*10,question_time=1,horizon=10,duration=15,
                             preprocessing_cache=lambda p:nullcontext())
        with self.assertRaises(SequenceTooLong):
            replay(model,duplex,record,clip,dict(self.config,max_sequence_tokens=35))
        self.assertIs(duplex.decoder,decoder)



if __name__=='__main__':
    unittest.main()
