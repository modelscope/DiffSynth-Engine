import unittest

import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageEditPlusPipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image-Edit-2511")
        config = QwenImagePipelineConfig(model_path=model_path)
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.engine
        torch.cuda.empty_cache()

    def test_single_image_edit(self):
        """Test single image editing with Edit Plus pipeline"""
        input_image = self.get_input_image("qwen_image_edit_input.png")
        prompt = "Replace '通义千问' with '呜哩AI'"
        negative_prompt = " "

        output = self.engine.generate(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            num_inference_steps=50,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit_plus_single_2511.png", threshold=0.99)

    def test_multi_image_edit(self):
        """Test multiple images editing with Edit Plus pipeline"""
        input_images = [
            self.get_input_image("qwen_image_edit_input_1.png").convert("RGB"),
            self.get_input_image("qwen_image_edit_input_2.png").convert("RGB"),
        ]
        prompt = "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而庄重。"
        negative_prompt = " "

        output = self.engine.generate(
            image=input_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            num_inference_steps=40,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit_plus_multi_2511.png", threshold=0.97)


if __name__ == "__main__":
    unittest.main()
