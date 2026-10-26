import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.data.agent_token import AgentStream, TOKEN
from benchmark.evaluation.agent_calls import parse_event


class ByteTokenizer:
    agent_id = 300
    all_special_ids = [300, 301, 302, 303, 304, 305]
    def get_vocab(self):
        return {TOKEN: 300}
    def encode(self, text, **kwargs):
        return [300] if text == TOKEN else list(text.encode())
    def convert_tokens_to_ids(self, token):
        return {TOKEN: 300, '<|speak|>': 301, '<|listen|>': 302,
                '<|turn_eos|>': 303, '<|chunk_eos|>': 304, '</unit>': 305}.get(token, -1)
    def decode(self, ids, **kwargs):
        return bytes(ids).decode()


class StreamTests(unittest.TestCase):
    def test_separator_first_token_of_next_chunk_and_complete_call(self):
        t = ByteTokenizer(); s = AgentStream(t)
        self.assertIsNone(s.feed([301] + t.encode('I will check.') + [304, 305], 4))
        self.assertIsNone(s.feed([300] + t.encode('{"name":"task_start",') + [304, 305], 5))
        event = s.feed([301] + t.encode('"arguments":{"instruction":"search"}}') + [303, 304, 305], 6, ended=True)
        self.assertEqual(event['response_text'], 'I will check.')
        self.assertEqual(event['onset_seconds'], 4)
        self.assertEqual(event['agent_call_onset_seconds'], 5)
        self.assertEqual(event['agent_call_complete_seconds'], 6)
        self.assertEqual(event['tool_calls'][0]['function']['name'], 'task_start')
        self.assertIn(300, event['raw_generated_ids'])

    def test_call_completion_is_turn_end_not_last_json_token(self):
        t = ByteTokenizer(); s = AgentStream(t)
        s.feed([300] + t.encode('{"name":"task_start","arguments":{"instruction":"search"}}'), 2)
        event = s.feed([303], 3, ended=True)
        self.assertEqual(event['agent_call_complete_seconds'], 3)

    def test_truncated_duplicate_invalid_and_negative(self):
        t = ByteTokenizer()
        for ids, status in [([300] + t.encode('{}'), 'malformed'),
                            ([300, 300] + t.encode('{}'), 'duplicate_separator')]:
            event = AgentStream(t).feed(ids, 1, ended=True)
            self.assertEqual(event['agent_parse_status'], status)
            self.assertTrue(parse_event(event)['attempted'])
            self.assertEqual(event['tool_calls'], [])
        stream = AgentStream(t)
        stream.feed([300] + t.encode('{"name":"task_start","arguments":{"instruction":"search"}}'), 1)
        event = stream.finish(complete=False)
        self.assertEqual(event['agent_parse_status'], 'truncated')
        self.assertEqual(event['tool_calls'], [])
        self.assertIsNone(stream.feed([302, 305], 2, interrupted=True))
        event = stream.feed(t.encode('Need more context.'), 3, ended=True)
        self.assertEqual(event['agent_parse_status'], 'absent')
        self.assertFalse(parse_event(event)['attempted'])


class TargetTests(unittest.TestCase):
    def test_token_protocol_targets_one_separator_and_no_negative_call(self):
        from train.src.data import target_chunks
        tokenizer = ByteTokenizer()
        record = {'sample': {'mode': 'agent'}, 'reference': {'response': 'I will check.',
            'tool_calls': [{'function': {'name': 'agent_call', 'arguments': '{"instruction":"search"}'}}]}}
        chunks = target_chunks(record, tokenizer, 8, protocol='task_text_sft_v1')
        self.assertEqual(sum(ids.count(300) for ids, _ in chunks), 0)
        self.assertIn('<|task_start|>', bytes(t for ids, groups in chunks for t,g in zip(ids,groups) if g and t < 256).decode())
        content = bytes(t for ids, groups in chunks for t,g in zip(ids,groups) if g == 3).decode()
        self.assertEqual(content, 'search')
        self.assertEqual(bytes(t for ids, groups in chunks for t,g in zip(ids,groups) if g == 2).decode(), 'I will check.')
        record['reference']['tool_calls'] = []
        chunks = target_chunks(record, tokenizer, 8, protocol='task_text_sft_v1')
        self.assertFalse(any(300 in ids or 3 in groups for ids, groups in chunks))


class AdapterTests(unittest.TestCase):
    def test_lora_save_reload_keeps_vocab_and_embedding_weights_frozen(self):
        import torch
        from types import SimpleNamespace
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        from train.src.checkpoints import save_adapter, load_adapter
        from train.src.model import add_lora, loss_for_record
        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);basepath = root / 'base';basepath.mkdir()
            raw = Tokenizer(WordLevel({'[UNK]':0,'hello':1,'search':2,'world':3,'<|':4,'task_start':5,'|>':6},unk_token='[UNK]'))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]')
            tokenizer.save_pretrained(basepath)
            initial = Qwen2ForCausalLM(Qwen2Config(vocab_size=7,hidden_size=16,intermediate_size=32,
                num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,tie_word_embeddings=False))
            initial.save_pretrained(basepath)
            model = SimpleNamespace(llm=copy.deepcopy(initial),config=copy.deepcopy(initial.config))
            model.llm.requires_grad_(False)
            before_in = model.llm.get_input_embeddings().weight.detach().clone()
            before_out = model.llm.get_output_embeddings().weight.detach().clone()
            encoded = {'inputs_embeds':model.llm.get_input_embeddings()(torch.tensor([1,4,5,6,2,3])).detach(),
                       'labels':torch.tensor([-100,4,5,6,2,3]),'groups':torch.tensor([0,1,1,1,3,2])}
            cfg = dict(lora_rank=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],
                       decision_weight=1.,content_weight=1.,call_weight=1.,loss_projection_chunk=2)
            llm = add_lora(model,cfg)
            loss_for_record(llm,encoded,cfg,'cpu').backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in llm.named_parameters() if 'lora_' in n))
            self.assertIsNone(llm.get_input_embeddings().weight.grad)
            self.assertIsNone(llm.get_output_embeddings().weight.grad)
            torch.optim.AdamW([p for p in llm.parameters() if p.requires_grad],lr=.01).step()
            self.assertTrue(torch.equal(llm.get_input_embeddings().weight,before_in))
            self.assertTrue(torch.equal(llm.get_output_embeddings().weight,before_out))
            self.assertEqual(len(tokenizer),7)
            llm.eval();ids=torch.tensor([[1,4,5,6,2]])
            expected=llm(ids).logits.detach()
            save_adapter(llm,tokenizer,basepath,root/'adapter')
            self.assertFalse((root/'adapter/agent_rows.safetensors').exists())
            restored=SimpleNamespace(llm=copy.deepcopy(initial),config=copy.deepcopy(initial.config),prepare_processor=lambda **kw:None)
            load_adapter(restored,basepath,root/'adapter');restored.llm.eval()
            self.assertTrue(torch.allclose(expected,restored.llm(ids).logits.detach(),atol=1e-5))
            meta_path=root/'adapter/agent_protocol.json';meta=json.loads(meta_path.read_text())
            meta['separator_token_ids']=[0];meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):load_adapter(restored,basepath,root/'adapter')

if __name__ == '__main__':unittest.main()
