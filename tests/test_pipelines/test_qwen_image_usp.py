import unittest

import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImagePipelineUSP(ImageTestCase):
    """2x2 sequence parallelism: ulysses=2 + ring=2 (4 GPUs)."""

    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image")
        config = QwenImagePipelineConfig(
            model_path=model_path,
            parallelism=4,
            sp_ulysses_degree=2,
            sp_ring_degree=2,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.cuda.empty_cache()

    def test_txt2img_sp_2x2(self):
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
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image.png", threshold=0.96)


if __name__ == "__main__":
    unittest.main()
