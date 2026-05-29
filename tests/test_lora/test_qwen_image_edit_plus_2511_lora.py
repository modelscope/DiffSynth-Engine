"""Unit tests for LoRA operations via DiffSynthEngine with QwenImageEditPlusPipeline."""

import os
import unittest

import torch

from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.engine import DiffSynthEngine
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageEditPlusLoRA(ImageTestCase):
    """Basic LoRA lifecycle test for QwenImageEditPlusPipeline."""

    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image-Edit-2511")
        cls.lora_dir = fetch_model("DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA")

        config = QwenImagePipelineConfig(
            model_path=model_path,
            pipeline_class_name="QwenImageEditPlusPipeline",
            device="cuda",
            model_dtype=torch.bfloat16,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

        cls.prompt = "Edit image 3 based on the transformation from image 1 to image 2."
        cls.negative_prompt = "泛黄，AI感，不真实，丑陋，油腻的皮肤，异常的肢体，不协调的肢体"
        cls.width = 720
        cls.height = 1280
        cls.steps = 30
        cls.seed = 1

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _lora_args(self):
        return {
            "lora_id": "in-context-editing-lora",
            "path": os.path.join(self.lora_dir, "model.safetensors"),
            "scale": 0.8,
        }

    def _get_input_images(self):
        return [
            self.get_input_image("qwen_image_edit_input_lora_1.png").convert("RGB"),
            self.get_input_image("qwen_image_edit_input_lora_2.png").convert("RGB"),
            self.get_input_image("qwen_image_edit_input_lora_3.png").convert("RGB"),
        ]

    def _generate(self):
        return self.engine.generate(
            image=self._get_input_images(),
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            true_cfg_scale=4.0,
            num_inference_steps=self.steps,
            generator=torch.Generator(device="cuda").manual_seed(self.seed),
        )

    def test_01_lifecycle(self):
        """load → merge → reset."""
        self.engine.reset_loras()
        [lora_id] = self.engine.load_loras(self._lora_args())
        self.assertEqual(self.engine.list_loras(lora_id)[0]["status"], "active")

        image = self._generate().images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image_lora/qwen_image_edit_plus_2511_lora_single.png")

        self.engine.reset_loras()
        self.assertEqual(self.engine.list_loras(), [])

        image = self._generate().images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image_lora/qwen_image_edit_plus_2511_lora_base.png")


if __name__ == "__main__":
    unittest.main()
