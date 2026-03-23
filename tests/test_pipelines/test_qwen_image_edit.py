import unittest

import torch

from diffsynth_engine.pipelines.qwen_image import QwenImageEditPipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageEditPipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image-Edit")
        cls.pipe = QwenImageEditPipeline.from_pretrained(model_path_or_config=model_path)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_single_image_edit(self):
        """Test single image editing with Edit pipeline"""
        input_image = self.get_input_image("qwen_image_edit_input.png")
        prompt = "Replace '通义千问' with '呜哩AI'"
        negative_prompt = " "

        output = self.pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            num_inference_steps=50,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit.png", threshold=0.99)


if __name__ == "__main__":
    unittest.main()
