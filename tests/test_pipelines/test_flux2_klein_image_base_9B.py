import unittest
import torch

from diffsynth_engine import Flux2KleinPipelineConfig
from diffsynth_engine.pipelines import Flux2KleinPipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestFlux2KleinPipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        config = Flux2KleinPipelineConfig(
            model_path=fetch_model("black-forest-labs/FLUX.2-klein-base-9B", path="transformer/*.safetensors"),
            encoder_path=fetch_model("black-forest-labs/FLUX.2-klein-9B", path="text_encoder/*.safetensors"),
            vae_path=fetch_model("black-forest-labs/FLUX.2-klein-9B", path="vae/*.safetensors"),
            model_dtype=torch.bfloat16,
            encoder_dtype=torch.bfloat16,
            vae_dtype=torch.bfloat16,
            model_size="9B",
        )
        cls.pipe = Flux2KleinPipeline.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_txt2img(self):
        prompt = "Masterpiece, best quality. Anime-style portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles."
        image = self.pipe(
            prompt=prompt,
            negative_prompt="",
            cfg_scale=4,
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=1,
        )
        self.assertImageEqualAndSaveFailed(image, "flux2/image_base_9B.jpg", threshold=0.90)

        prompt = "change the color of the clothes to red"
        image = self.pipe(
            prompt=prompt,
            negative_prompt="",
            edit_image=image,
            cfg_scale=4,
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=2,
        )
        self.assertImageEqualAndSaveFailed(image, "flux2/image_edit_base_9B.jpg", threshold=0.90)


if __name__ == "__main__":
    unittest.main()
