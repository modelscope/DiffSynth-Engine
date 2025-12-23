import torch

from diffsynth_engine.tokenizers.qwen2 import Qwen2TokenizerFast
from diffsynth_engine.utils.constants import QWEN_IMAGE_TOKENIZER_CONF_PATH, Z_IMAGE_TOKENIZER_CONF_PATH
from tests.common.test_case import TestCase


class TestQwen2TokenizerFast(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN_IMAGE_TOKENIZER_CONF_PATH)
        cls.z_image_tokenizer = Qwen2TokenizerFast.from_pretrained(Z_IMAGE_TOKENIZER_CONF_PATH)

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
            {
                "texts": """
DiffSynth-Engine is a high-performance engine geared towards buidling efficient inference pipelines for diffusion models.
Key Features:
Thoughtfully-Designed Implementation: We carefully re-implemented key components in Diffusion pipelines, such as sampler and scheduler, without introducing external dependencies on libraries like k-diffusion, ldm, or sgm.
Extensive Model Support: Compatible with popular formats (e.g., CivitAI) of base models and LoRA models , catering to diverse use cases.
Versatile Resource Management: Comprehensive support for varous model quantization (e.g., FP8, INT8) and offloading strategies, enabling loading of larger diffusion models (e.g., Flux.1 Dev) on limited hardware budget of GPU memory.
Optimized Performance: Carefully-crafted inference pipeline to achieve fast generation across various hardware environments.
Cross-Platform Support: Runnable on Windows, macOS (Apple Silicon), and Linux, ensuring a smooth experience across different operating systems.
""",
                "expected": torch.tensor(
                    [
                        [
                            198,
                            21751,
                            37134,
                            339,
                            12,
                            4571,
                            374,
                            264,
                            1550,
                            57474,
                            4712,
                            58447,
                            6974,
                            1031,
                            307,
                            2718,
                            11050,
                            44378,
                            57673,
                            369,
                            57330,
                            4119,
                            624,
                            1592,
                            19710,
                            510,
                            84169,
                            3641,
                            12,
                            77133,
                            30813,
                            25,
                            1205,
                            15516,
                            312,
                            36925,
                            14231,
                            1376,
                            6813,
                            304,
                            28369,
                            7560,
                            57673,
                            11,
                            1741,
                            438,
                            41799,
                            323,
                            28809,
                            11,
                            2041,
                            31918,
                            9250,
                            19543,
                            389,
                            20186,
                            1075,
                            595,
                            1737,
                            3092,
                            7560,
                            11,
                            326,
                            13849,
                            11,
                            476,
                            274,
                            26186,
                            624,
                            6756,
                            4025,
                            4903,
                            9186,
                            25,
                            66265,
                            448,
                            5411,
                            19856,
                            320,
                            68,
                            1302,
                            2572,
                            79135,
                            275,
                            15469,
                            8,
                            315,
                            2331,
                            4119,
                            323,
                            6485,
                            5609,
                            4119,
                            1154,
                            53829,
                            311,
                            16807,
                            990,
                            5048,
                            624,
                            83956,
                            9010,
                            11765,
                            9551,
                            25,
                            66863,
                            1824,
                            369,
                            762,
                            782,
                            1614,
                            10272,
                            2022,
                            320,
                            68,
                            1302,
                            2572,
                            33551,
                            23,
                            11,
                            9221,
                            23,
                            8,
                            323,
                            1007,
                            10628,
                            14830,
                            151645,
                        ]
                    ]
                ),
                "shape": (1, 128),
                "kwargs": {"max_length": 128, "padding_side": "left"},
            },
        ]

        for case in cases:
            texts, expected, shape, kwargs = case["texts"], case["expected"], case["shape"], case["kwargs"]
            result = self.tokenizer(texts, **kwargs)["input_ids"]
            self.assertEqual(shape, result.shape)
            truncated = result[:, : expected.shape[1]]
            self.assertTrue(torch.equal(expected, truncated))

    def test_z_image(self):
        expects = self.get_input_tensor("z_image/z_image_text_encoder.safetensors")
        prompt = [
            "<|im_start|>user\nYoung Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.<|im_end|>\n<|im_start|>assistant\n"
        ]
        result = self.z_image_tokenizer(prompt, max_length=512, padding_side="right", padding_strategy="max_length")
        self.assertTrue(torch.equal(expects["text_input_ids"], result["input_ids"]))
        self.assertTrue(torch.equal(expects["prompt_masks"], result["attention_mask"]))
