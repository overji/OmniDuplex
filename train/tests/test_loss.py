import unittest
try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from train.src.model import add_lora, loss_for_record
    import peft
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Run with the training environment for torch/PEFT tests')
class LossTests(unittest.TestCase):
    def test_shifted_target_loss_matches_full_logits_and_updates_only_lora(self):
        from types import SimpleNamespace
        torch.manual_seed(7)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1))
        base.requires_grad_(False)
        cfg = dict(lora_rank=2, lora_alpha=4, lora_dropout=0., target_modules=['q_proj', 'v_proj'],
                   decision_weight=1., content_weight=1., call_weight=1., loss_projection_chunk=2)
        llm = add_lora(SimpleNamespace(llm=base), cfg)
        llm.train()
        ids = torch.tensor([1, 2, 3, 4, 5, 6])
        encoded = dict(inputs_embeds=base.get_input_embeddings()(ids).detach(),
                       labels=torch.tensor([-100, -100, 3, 4, -100, 6]),
                       groups=torch.tensor([0, 0, 1, 2, 0, 2]))
        actual = loss_for_record(llm, encoded, cfg, 'cpu')
        logits = llm(inputs_embeds=encoded['inputs_embeds'][None], use_cache=False).logits[0]
        expected = torch.nn.functional.cross_entropy(logits[1:2], torch.tensor([3]))
        expected += torch.nn.functional.cross_entropy(logits[[2, 4]], torch.tensor([4, 6]))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
        actual.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in llm.named_parameters() if 'lora_' in n))
        self.assertFalse(any(p.grad is not None for n,p in llm.named_parameters() if 'lora_' not in n))

if __name__ == '__main__':
    unittest.main()
