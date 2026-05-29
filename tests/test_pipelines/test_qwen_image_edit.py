import unittest

import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageEditPipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image-Edit")
        config = QwenImagePipelineConfig(model_path=model_path)
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.cuda.empty_cache()

    def test_single_image_edit(self):
        """Test single image editing with Edit pipeline"""
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
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit.png", threshold=0.99)


class TestQwenImageEditPipelineParallel(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image-Edit")
        config = QwenImagePipelineConfig(
            model_path=model_path,
            parallelism=2,
            use_cfg_parallel=True,
            sp_ulysses_degree=1,
            sp_ring_degree=1,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.cuda.empty_cache()

    def test_single_image_edit_parallel(self):
        """Test single image editing with Edit pipeline in parallel mode"""
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
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit.png", threshold=0.99)


if __name__ == "__main__":
    unittest.main()
