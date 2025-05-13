import torch
import unittest

from diffsynth_engine.tokenizers.clip import CLIPTokenizer
from diffsynth_engine.utils.constants import FLUX_TOKENIZER_1_CONF_PATH
from tests.common.test_case import TestCase


class TestCLIPTokenizer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CLIPTokenizer.from_pretrained(FLUX_TOKENIZER_1_CONF_PATH)

    def test_tokenize(self):
        cases = [
            {
                "texts": "Hello, World!",
                "expected": ["hello</w>", ",</w>", "world</w>", "!</w>"],
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": [
                    ["hello</w>", ",</w>", "world</w>", "!</w>"],
                    [
                        "diff",
                        "synth</w>",
                        "-</w>",
                        "engine</w>",
                        "developed</w>",
                        "by</w>",
                        "muse</w>",
                        "ai</w>",
                        "+</w>",
                        "model",
                        "scope</w>",
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
                "expected": [3306, 267, 1002, 256],
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": [
                    [3306, 267, 1002, 256],
                    [44073, 24462, 268, 5857, 8763, 638, 15686, 2215, 266, 4591, 7979],
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
                "ids": [49406, 3306, 267, 1002, 256, 49407],
                "expected": "hello, world!",
                "kwargs": {"skip_special_tokens": True},
            },
            {
                "ids": [49406, 3306, 267, 1002, 256, 49407],
                "expected": "<|startoftext|>hello, world! <|endoftext|>",
                "kwargs": {"skip_special_tokens": False},
            },
            {
                "ids": [
                    [3306, 267, 1002, 256, 49407],
                    [44073, 24462, 268, 5857, 8763, 638, 15686, 2215, 266, 4591, 7979, 49407],
                ],
                "expected": ["hello, world!", "diffsynth - engine developed by muse ai + modelscope"],
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
                "expected": torch.tensor([[49406, 3306, 267, 1002, 256, 49407]], dtype=torch.long),
                "shape": (1, 77),
                "kwargs": {},
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": torch.tensor(
                    [
                        [49406, 3306, 267, 1002, 256, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407],
                        [49406, 44073, 24462, 268, 5857, 8763, 638, 15686, 2215, 266, 4591, 7979, 49407],
                    ]
                ),
                "shape": (2, 77),
                "kwargs": {},
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": torch.tensor(
                    [
                        [49406, 3306, 267, 1002, 256, 49407, 49407, 49407, 49407, 49407],
                        [49406, 44073, 24462, 268, 5857, 8763, 638, 15686, 2215, 49407],
                    ]
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