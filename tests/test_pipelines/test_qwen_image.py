import unittest

import torch

from diffsynth_engine.pipelines.qwen_image import QwenImagePipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImagePipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image")
        cls.pipe = QwenImagePipeline.from_pretrained(model_path_or_config=model_path)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_txt2img(self):
        prompt = "A painting of a cat in a zen garden"
        negative_prompt = "ugly, blurry, low quality"
        output = self.pipe(
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
