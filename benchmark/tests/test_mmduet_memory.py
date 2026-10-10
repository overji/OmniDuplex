import unittest

from benchmark.adapters.mmduet2 import chunk_mlp_forward


class MMDuetMemoryTests(unittest.TestCase):
    def test_chunked_gated_mlp_preserves_all_tokens_and_bounds_activations(self):
        import torch
        torch.manual_seed(7)
        gate = torch.nn.Linear(8, 19)
        up = torch.nn.Linear(8, 19)
        down = torch.nn.Linear(19, 8)
        lengths = []

        def original(x):
            lengths.append(x.shape[-2])
            return down(torch.nn.functional.silu(gate(x)) * up(x))

        values = torch.randn(2, 11, 8)
        chunked = chunk_mlp_forward(original, 4)
        with torch.inference_mode():
            expected = original(values)
            lengths.clear()
            actual = chunked(values)
            self.assertEqual(lengths, [4, 4, 3])
            torch.testing.assert_close(actual, expected)
            short = values[:, :2]
            torch.testing.assert_close(chunked(short), original(short))
        with self.assertRaisesRegex(RuntimeError, "inference mode"):
            chunked(values)
        for size in (0, -1):
            with self.assertRaises(ValueError):
                chunk_mlp_forward(original, size)


if __name__ == '__main__':
    unittest.main()
