"""新全参MiniCPM的原子分隔符加载/解析，不运行真实权重。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from benchmark.data.task_text import TaskTextStream,SEPARATOR


class Tokenizer:
    all_special_ids=[300,301,302,303,304,305]
    eos_token_id=305
    def get_vocab(self):return {SEPARATOR:300}
    def encode(self,text,**kwargs):
        result=[]
        for i,part in enumerate(text.split(SEPARATOR)):
            if i:result.append(300)
            result.extend(part.encode())
        return result
    def convert_tokens_to_ids(self,t):return {SEPARATOR:300,'<|speak|>':301,'<|listen|>':302,'<|turn_eos|>':303,'<|chunk_eos|>':304,'</unit>':305}.get(t,-1)
    def decode(self,ids,**kwargs):
        return ''.join(SEPARATOR if i==300 and not kwargs.get('skip_special_tokens',False) else chr(i) if i<256 else '' for i in ids)


class AtomicBenchmarkTests(unittest.TestCase):
    def test_atomic_separator_survives_special_filter_and_cross_chunk_completion(self):
        s=TaskTextStream(Tokenizer())
        self.assertIsNone(s.feed([301,ord('R'),300,304],1))
        event=s.feed([301,ord('T'),303,305],2,ended=True)
        self.assertEqual(event['response_text'],'R');self.assertEqual(event['agent_instruction'],'T')
        self.assertEqual(event['agent_call_onset_seconds'],1)

    def test_metadata_selects_prompt_and_schema_before_scoring(self):
        from benchmark.adapters.minicpmo import prepare_args
        with tempfile.TemporaryDirectory() as directory:
            args=NS(model_path=Path(directory),agent_output_protocol='text',agent_call_schema='legacy')
            prepare_args(args);self.assertEqual(args.agent_output_protocol,'text')
            (args.model_path/'full_training_model.json').write_text(json.dumps(dict(protocol='task_start_token_sft_v1')))
            prepare_args(args)
            self.assertEqual((args.agent_output_protocol,args.agent_call_schema),('task-text','task-start'))

    def test_full_checkpoint_uses_new_validator_and_extended_tokenizer(self):
        from benchmark.adapters.minicpmo import load
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model_path=root/'model';model_path.mkdir();cache=root/'cache';cache.mkdir()
            (model_path/'full_training_model.json').write_text(json.dumps(dict(protocol='task_start_token_sft_v1')))
            duplex=NS(tokenizer=Tokenizer(),decoder=NS(forbidden_token_ids=[]),forbidden_token_ids=[],model=NS())
            model=Mock();model.eval.return_value=model;model.to.return_value=model;model.as_duplex.return_value=duplex
            args=NS(model_path=model_path,cache_dir=cache,cpu_threads=1,device='cpu',lora_adapter=None,
                    unused_tts_device='none',sliding_window_mode='off',window_high_tokens=32,window_low_tokens=16)
            with patch('transformers.AutoModel.from_pretrained',return_value=model), \
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=duplex.tokenizer), \
                 patch('train.src.checkpoints.full_model_identity') as validate:
                state=load(args)
            validate.assert_called_once_with(model_path)
            self.assertTrue(state.atomic_task_start)
            model.prepare_processor.assert_called_once_with(tokenizer=duplex.tokenizer)

    def test_probe_recovers_atomic_marker_from_raw_ids_not_native_text(self):
        import torch
        from benchmark.probe.native_models import generate_minicpmo
        class Model:
            def __init__(self):self.llm=torch.nn.Linear(2,2)
            def generate(self,*args,**kwargs):return 'R T',NS(sequences=torch.tensor([[ord('R'),300,ord('T'),305]]))
            def chat(self,**kwargs):return self.generate()[0]
        state=NS(model=Model(),processor=NS(tokenizer=Tokenizer()),atomic_task_start=True)
        args=NS(max_new_tokens=20,fps=1,max_pixels=1024)
        probe=NS(video=Path('unused'),has_audio=False,user_prompt='u',system_prompt='s',prefix_seconds=1)
        with patch('benchmark.probe.native_models.sampled_frames',return_value=[]):result=generate_minicpmo(args,state,probe)
        self.assertEqual(str(result),'R'+SEPARATOR+'T');self.assertTrue(result.generation_complete)


if __name__=='__main__':unittest.main()
