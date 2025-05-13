import torch
import unittest

from diffsynth_engine.tokenizers.t5 import T5TokenizerFast
from diffsynth_engine.utils.constants import FLUX_TOKENIZER_2_CONF_PATH
from tests.common.test_case import TestCase


class TestT5TokenizerFast(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = T5TokenizerFast.from_pretrained(FLUX_TOKENIZER_2_CONF_PATH)

    def test_tokenize(self):
        cases = [
            {
                "texts": "Hello, World!",
                "expected": ["▁Hello", ",", "▁World", "!", "</s>"],
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": [
                    ["▁Hello", ",", "▁World", "!", "</s>"],
                    [
                        "▁D",
                        "iff",
                        "S",
                        "y",
                        "n",
                        "th",
                        "-",
                        "Engine",
                        "▁developed",
                        "▁by",
                        "▁Mus",
                        "e",
                        "▁AI",
                        "+",
                        "Model",
                        "scope",
                        "</s>",
                    ],
                ],
            },
        ]

        for case in cases:
            texts, expected = case["texts"], case["expected"]
            result = self.tokenizer.tokenize(texts)
            self.assertEqual(expected, result)

    def test_encode(self):
        cases = [
            {
                "texts": "Hello, World!",
                "expected": [8774, 6, 1150, 55, 1],
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": [
                    [8774, 6, 1150, 55, 1],
                    [309, 5982, 134, 63, 29, 189, 18, 31477, 1597, 57, 6887, 15, 7833, 1220, 24663, 11911, 1],
                ],
            },
        ]

        for case in cases:
            texts, expected = case["texts"], case["expected"]
            if isinstance(texts, str):
                result = self.tokenizer.encode(texts)
            else:
                result = self.tokenizer.batch_encode(texts)
            self.assertEqual(expected, result)

    def test_decode(self):
        cases = [
            {
                "ids": [8774, 6, 1150, 55, 1],
                "expected": "Hello, World!",
                "kwargs": {"skip_special_tokens": True},
            },
            {
                "ids": [8774, 6, 1150, 55, 1],
                "expected": "Hello, World!</s>",
                "kwargs": {"skip_special_tokens": False},
            },
            {
                "ids": [
                    [8774, 6, 1150, 55, 1],
                    [309, 5982, 134, 63, 29, 189, 18, 31477, 1597, 57, 6887, 15, 7833, 1220, 24663, 11911, 1],
                ],
                "expected": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "kwargs": {"skip_special_tokens": True},
            },
        ]

        for case in cases:
            ids, expected, kwargs = case["ids"], case["expected"], case["kwargs"]
            if isinstance(ids[0], int):
                result = self.tokenizer.decode(ids, **kwargs)
            else:
                result = self.tokenizer.batch_decode(ids, **kwargs)
            self.assertEqual(expected, result)

    def test_call(self):
        cases = [
            {
                "texts": "Hello, World!",
                "expected": torch.tensor([[8774, 6, 1150, 55, 1]], dtype=torch.long),
                "shape": (1, 512),
                "kwargs": {},
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": torch.tensor(
                    [
                        [8774, 6, 1150, 55, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [309, 5982, 134, 63, 29, 189, 18, 31477, 1597, 57, 6887, 15, 7833, 1220, 24663, 11911, 1],
                    ]
                ),
                "shape": (2, 512),
                "kwargs": {},
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": torch.tensor(
                    [[8774, 6, 1150, 55, 1, 0, 0, 0, 0, 0], [309, 5982, 134, 63, 29, 189, 18, 31477, 1597, 1]]
                ),
                "shape": (2, 10),
                "kwargs": {"max_length": 10},
            },
        ]

        for case in cases:
            texts, expected, shape, kwargs = case["texts"], case["expected"], case["shape"], case["kwargs"]
            result = self.tokenizer(texts, **kwargs)["input_ids"]
            self.assertEqual(shape, result.shape)
            truncated = result[:, : expected.shape[1]]
            self.assertTrue(torch.equal(expected, truncated))

if __name__ == "__main__":
    unittest.main()