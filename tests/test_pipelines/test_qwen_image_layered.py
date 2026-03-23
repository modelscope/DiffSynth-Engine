import unittest

import torch

from diffsynth_engine.pipelines.qwen_image import QwenImageLayeredPipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageLayeredPipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image-Layered")
        cls.pipe = QwenImageLayeredPipeline.from_pretrained(model_path_or_config=model_path)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_image_layered(self):
        input_image = self.get_input_image("qwen_image_layered_input.png").convert("RGBA")
        prompt = ""

        output = self.pipe(
            image=input_image,
            prompt=prompt,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            layers=4,
            resolution=640,
            cfg_normalize=False,
            use_en_prompt=True,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        images = output.images[0]
        self.assertEqual(len(images), 4)

        # Compare each layer with reference images
        from tests.common.utils import compute_normalized_ssim

        ssim_results = []
        for i, layer_image in enumerate(images):
            expect_image = self.get_expect_image(f"qwen_image/qwen_image_layered_{i}.png")
            ssim = compute_normalized_ssim(layer_image, expect_image)
            ssim_results.append((i, ssim))
            print(f"Layer {i} (qwen_image_layered_{i}.png): SSIM = {ssim:.6f}")

        for i, layer_image in enumerate(images):
            self.assertImageEqualAndSaveFailed(
                layer_image,
                f"qwen_image/qwen_image_layered_{i}.png",
                threshold=0.98,
            )


if __name__ == "__main__":
    unittest.main()
