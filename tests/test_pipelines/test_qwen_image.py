import unittest
import torch

from diffsynth_engine import QwenImagePipelineConfig
from diffsynth_engine.pipelines import QwenImagePipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImagePipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        config = QwenImagePipelineConfig(
            model_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="transformer/*.safetensors"),
            encoder_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="text_encoder/*.safetensors"),
            vae_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="vae/*.safetensors"),
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
            prompt="A painting of a cat in a zen garden",
            negative_prompt="ugly",
            cfg_scale=4.0,
            width=1328,
            height=1328,
            num_inference_steps=28,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image.png", threshold=0.99)


if __name__ == "__main__":
    unittest.main()
