import unittest

import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageVaeParallel(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image")
        config = QwenImagePipelineConfig(
            model_path=model_path,
            parallelism=2,
            use_cfg_parallel=True,
            sp_ulysses_degree=1,
            sp_ring_degree=1,
            vae_tiled=True,
            use_vae_parallel=True,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.engine

    def test_txt2img(self):
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
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image.png", threshold=0.99)


if __name__ == "__main__":
    unittest.main()
