"""CPU微型媒体模型与原子tokenizer，不加载真实权重。"""
from types import SimpleNamespace
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from train.src.domain.agent import SEPARATOR


class ByteTokenizer:
    all_special_ids=[300,301,302,303,304,305]
    def get_vocab(self):return {**{str(i):i for i in range(300)},SEPARATOR:300,'<|speak|>':301,'<|listen|>':302,'<|turn_eos|>':303,'<|chunk_eos|>':304,'</unit>':305}
    def __len__(self):return 306
    def encode(self,text,**kwargs):
        parts=text.split(SEPARATOR);ids=[]
        for index,part in enumerate(parts):
            if index:ids.append(300)
            ids.extend(part.encode())
        return ids
    def convert_tokens_to_ids(self,token):return self.get_vocab().get(token,-1)
    def decode(self,ids,**kwargs):
        parts=[];buf=[]
        for token in ids:
            if token==300:
                parts.append(bytes(buf).decode());buf=[]
                if not kwargs.get('skip_special_tokens',False):parts.append(SEPARATOR)
            elif token<256:buf.append(token)
        parts.append(bytes(buf).decode());return ''.join(parts)


class TinyMedia(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=320, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
        self.vpm, self.apm = torch.nn.Linear(3, 16), torch.nn.Linear(3, 16)
        self.resampler, self.audio_projection_layer = torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)
        self.tts = torch.nn.Linear(3, 3)

    def init_streaming_processor(self):
        self.audio_past_key_values = None

class TinyDuplex:
    def __init__(self, model):
        self.model = model
        self.tokenizer = ByteTokenizer()
        self.processor = SimpleNamespace(process_image=lambda *a, **k: None,
                                         process_audio_streaming=lambda *a, **k: None)
        self.decoder = SimpleNamespace(forbidden_token_ids=[304])
        self.listen_token_id, self.chunk_eos_token_id, self.tts_bos_token_id = 302, 304, 301

    def _reset_streaming_state(self):
        pass

    @torch.no_grad()
    def streaming_prefill(self, text_list=None, **kwargs):
        if text_list:
            self.decoder.feed(self.decoder.embed_token(self.tokenizer.encode(text_list[0])))
        else:
            self.decoder.feed(self.model.resampler(self.model.vpm(torch.ones(1, 3))))
            self.decoder.feed(self.model.audio_projection_layer(self.model.apm(torch.ones(1, 3))))
        return dict(success=True)
