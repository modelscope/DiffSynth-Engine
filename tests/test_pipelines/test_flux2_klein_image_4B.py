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
            model_path=fetch_model("black-forest-labs/FLUX.2-klein-4B", path="transformer/*.safetensors"),
            encoder_path=fetch_model("black-forest-labs/FLUX.2-klein-4B", path="text_encoder/*.safetensors"),
            vae_path=fetch_model("black-forest-labs/FLUX.2-klein-4B", path="vae/*.safetensors"),
            model_dtype=torch.bfloat16,
            encoder_dtype=torch.bfloat16,
            vae_dtype=torch.bfloat16,
            model_size="4B",
        )
        cls.pipe = Flux2KleinPipeline.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_txt2img(self):
        prompt = "Masterpiece, best quality. Anime-style portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles."
        image = self.pipe(
            prompt=prompt,
            cfg_scale=1,
            width=1024,
            height=1024,
            num_inference_steps=4,
            seed=0,
        )
        self.assertImageEqualAndSaveFailed(image, "flux2/image.jpg", threshold=0.95)

        prompt = "change the color of the clothes to red"
        image = self.pipe(
            prompt=prompt,
            edit_image=image,
            cfg_scale=1,
            width=1024,
            height=1024,
            num_inference_steps=4,
            seed=1,
        )
        self.assertImageEqualAndSaveFailed(image, "flux2/image_edit.jpg", threshold=0.95)


if __name__ == "__main__":
    unittest.main()
