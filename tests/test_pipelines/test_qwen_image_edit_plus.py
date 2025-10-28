import unittest
import torch

from diffsynth_engine import QwenImagePipelineConfig
from diffsynth_engine.pipelines import QwenImagePipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageEditPlusPipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        config = QwenImagePipelineConfig(
            model_path=fetch_model("Qwen/Qwen-Image-Edit-2509", path="transformer/*.safetensors"),
            encoder_path=fetch_model("Qwen/Qwen-Image-Edit-2509", path="text_encoder/*.safetensors"),
            vae_path=fetch_model("Qwen/Qwen-Image-Edit-2509", path="vae/*.safetensors"),
            model_dtype=torch.bfloat16,
            encoder_dtype=torch.bfloat16,
            vae_dtype=torch.float32,
        )
        cls.pipe = QwenImagePipeline.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_txt2img(self):
        image = self.pipe(
            prompt="根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而庄重。",
            input_image=[self.get_input_image("qwen_1.png"), self.get_input_image("qwen_2.png")],
            negative_prompt=" ",
            cfg_scale=4.0,
            width=1328,
            height=1328,
            num_inference_steps=40,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit_plus.png", threshold=0.95)


if __name__ == "__main__":
    unittest.main()
