import unittest
import torch
import math

from diffsynth_engine import QwenImagePipelineConfig
from diffsynth_engine.pipelines import QwenImagePipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImagePipelineSVDQuant(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        config = QwenImagePipelineConfig(
            model_path=fetch_model(
                "nunchaku-tech/nunchaku-qwen-image-edit-2509", path="svdq-int4_r128-qwen-image-edit-2509.safetensors"
            ),
            encoder_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="text_encoder/*.safetensors"),
            vae_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="vae/*.safetensors"),
            model_dtype=torch.bfloat16,
            encoder_dtype=torch.bfloat16,
            vae_dtype=torch.float32,
            offload_mode="cpu_offload",
        )
        cls.pipe = QwenImagePipeline.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_txt2img(self):
        image = self.pipe(
            prompt="Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep. ",
            negative_prompt=" ",
            input_image=[
                self.get_input_image("man.png"),
                self.get_input_image("puppy.png"),
                self.get_input_image("sofa.png"),
            ],
            cfg_scale=4.0,
            num_inference_steps=40,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_svd_quant.png", threshold=0.90)

    def test_lora(self):
        self.pipe.load_lora(
            path=fetch_model(
                "lightx2v/Qwen-Image-Lightning", path="Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
            ),
            scale=1.0,
            fused=True,
        )
        self.pipe.apply_scheduler_config({"exponential_shift_mu": math.log(3)})
        image = self.pipe(
            prompt="Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep. ",
            negative_prompt=" ",
            input_image=[
                self.get_input_image("man.png"),
                self.get_input_image("puppy.png"),
                self.get_input_image("sofa.png"),
            ],
            cfg_scale=1.0,
            num_inference_steps=4,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_svd_quant_lora.png", threshold=0.90)
        self.pipe.unload_loras()


if __name__ == "__main__":
    unittest.main()
