import torch

from diffsynth_engine.tokenizers.qwen2 import Qwen2TokenizerFast
from diffsynth_engine.utils.constants import QWEN_IMAGE_TOKENIZER_CONF_PATH
from tests.common.test_case import TestCase


class TestQwen2TokenizerFast(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN_IMAGE_TOKENIZER_CONF_PATH)

    # TODO: fix
    def test_tokenize(self):
        cases = [
            {
                "texts": "Hello, World!",
                "expected": ["Hello", ",", "ĠWorld", "!"],
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": [
                    ["▁Hello", ",", "▁World", "!", "</s>"],
                    ["Diff", "Syn", "th", "-", "Engine", "Ġdeveloped", "Ġby", "ĠMuse", "ĠAI", "+", "Model", "scope"],
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
                "expected": [9707, 11, 4337, 0],
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": [
                    [9707, 11, 4337, 0],
                    [21751, 37134, 339, 12, 4571, 7881, 553, 50008, 15235, 10, 1712, 4186],
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
                "ids": [9707, 11, 4337, 0],
                "expected": "Hello, World!",
                "kwargs": {"skip_special_tokens": True},
            },
            {
                "ids": [9707, 11, 4337, 0],
                "expected": "Hello, World!",
                "kwargs": {"skip_special_tokens": False},
            },
            {
                "ids": [
                    [9707, 11, 4337, 0],
                    [21751, 37134, 339, 12, 4571, 7881, 553, 50008, 15235, 10, 1712, 4186],
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
            print(result)
            self.assertEqual(expected, result)

    def test_call(self):
        cases = [
            {
                "texts": "Hello, World!",
                "expected": torch.tensor([[9707, 11, 4337, 0]], dtype=torch.long),
                "shape": (1, 4),
                "kwargs": {},
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": torch.tensor(
                    [
                        [9707, 11, 4337, 0, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643],
                        [21751, 37134, 339, 12, 4571, 7881, 553, 50008, 15235, 10, 1712, 4186],
                    ]
                ),
                "shape": (2, 12),
                "kwargs": {},
            },
            {
                "texts": ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"],
                "expected": torch.tensor(
                    [
                        [151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 9707, 11, 4337, 0],
                        [21751, 37134, 339, 12, 4571, 7881, 553, 50008, 15235, 10, 1712, 4186],
                    ]
                ),
                "shape": (2, 12),
                "kwargs": {"padding_side": "left"},
            },
        ]

        for case in cases:
            texts, expected, shape, kwargs = case["texts"], case["expected"], case["shape"], case["kwargs"]
            result = self.tokenizer(texts, **kwargs)["input_ids"]
            self.assertEqual(shape, result.shape)
            truncated = result[:, : expected.shape[1]]
            self.assertTrue(torch.equal(expected, truncated))
