import unittest

import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageLayeredPipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image-Layered")
        config = QwenImagePipelineConfig(model_path=model_path)
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.engine
        torch.cuda.empty_cache()

    def test_image_layered(self):
        input_image = self.get_input_image("qwen_image_layered_input.png").convert("RGBA")
        prompt = ""

        output = self.engine.generate(
            image=input_image,
            prompt=prompt,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            layers=3,
            resolution=640,
            cfg_normalize=False,
            use_en_prompt=True,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        images = output.images[0]
        self.assertEqual(len(images), 3)

        for i, layer_image in enumerate(images):
            self.assertImageEqualAndSaveFailed(
                layer_image,
                f"qwen_image/qwen_image_layered_{i}.png",
                threshold=0.97,
            )


if __name__ == "__main__":
    unittest.main()
