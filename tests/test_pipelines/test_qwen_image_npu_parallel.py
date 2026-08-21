"""NPU 多卡并行 Qwen Image 测试（Ulysses SP）"""
import os
import unittest

import torch

try:
    import torch_npu

    NPU_AVAILABLE = torch.npu.is_available()
    NPU_COUNT = torch.npu.device_count() if NPU_AVAILABLE else 0
except ImportError:
    NPU_AVAILABLE = False
    NPU_COUNT = 0

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


@unittest.skipUnless(NPU_AVAILABLE and NPU_COUNT >= 4, "Need at least 4 NPUs")
class TestQwenImageNPU4Card(ImageTestCase):
    """4 卡 Ulysses SP 测试"""

    @classmethod
    def setUpClass(cls):
        os.environ["USE_MINDIESD_FUSE"] = "true"
        model_path = fetch_model("Qwen/Qwen-Image")
        config = QwenImagePipelineConfig(
            model_path=model_path,
            model_dtype=torch.bfloat16,
            device="npu",
            attn_type="mindie",
            parallelism=4,
            sp_ulysses_degree=4,
            sp_ring_degree=1,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.npu.empty_cache()

    def test_txt2img_ulysses_4card(self):
        prompt = "A painting of a cat in a zen garden"
        negative_prompt = "ugly, blurry, low quality"
        output = self.engine.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            width=1328,
            height=1328,
            num_inference_steps=28,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image.png", threshold=0.97)


@unittest.skipUnless(NPU_AVAILABLE and NPU_COUNT >= 8, "Need at least 8 NPUs")
class TestQwenImageNPU8Card(ImageTestCase):
    """8 卡 Ulysses SP 测试"""

    @classmethod
    def setUpClass(cls):
        os.environ["USE_MINDIESD_FUSE"] = "true"
        model_path = fetch_model("Qwen/Qwen-Image")
        config = QwenImagePipelineConfig(
            model_path=model_path,
            model_dtype=torch.bfloat16,
            device="npu",
            attn_type="mindie",
            parallelism=8,
            sp_ulysses_degree=8,
            sp_ring_degree=1,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.npu.empty_cache()

    def test_txt2img_ulysses_8card(self):
        prompt = "A painting of a cat in a zen garden"
        negative_prompt = "ugly, blurry, low quality"
        output = self.engine.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            width=1328,
            height=1328,
            num_inference_steps=28,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image.png", threshold=0.97)


if __name__ == "__main__":
    unittest.main()
