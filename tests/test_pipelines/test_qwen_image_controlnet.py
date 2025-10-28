import unittest
import torch

from diffsynth_engine import QwenImagePipelineConfig, QwenImageControlNetParams, QwenImageControlType
from diffsynth_engine.pipelines import QwenImagePipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageControlnet(ImageTestCase):
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

    def test_incontext_canny(self):
        param = QwenImageControlNetParams(
            control_type=QwenImageControlType.in_context,
            image=self.get_input_image("canny.png"),
            scale=1.0,
            model=fetch_model("DiffSynth-Studio/Qwen-Image-In-Context-Control-Union"),
        )
        image = self.pipe(
            prompt="Context_Control. A young girl stands gracefully at the edge of a serene beach, her long, flowing hair gently tousled by the sea breeze. She wears a soft, pastel-colored dress that complements the tranquil blues and greens of the coastal scenery. The golden hues of the setting sun cast a warm glow on her face, highlighting her serene expression. The background features a vast, azure ocean with gentle waves lapping at the shore, surrounded by distant cliffs and a clear, cloudless sky. The composition emphasizes the girl's serene presence amidst the natural beauty, with a balanced blend of warm and cool tones.",
            negative_prompt=" ",
            num_inference_steps=28,
            seed=42,
            controlnet_params=param,
        )
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_canny.png", threshold=0.95)
        self.pipe.unload_loras()

    def test_incontext_depth(self):
        param = QwenImageControlNetParams(
            control_type=QwenImageControlType.in_context,
            image=self.get_input_image("qwen_image_depth.png"),
            scale=1.0,
            model=fetch_model("DiffSynth-Studio/Qwen-Image-In-Context-Control-Union"),
        )
        image = self.pipe(
            prompt="Context_Control. 一个穿着淡蓝色的漂亮女孩正在翩翩起舞，背景是梦幻的星空，光影交错，细节精致。",
            negative_prompt="网格化，规则的网格，模糊, 低分辨率, 低质量, 变形, 畸形, 错误的解剖学, 变形的手, 变形的身体, 变形的脸, 变形的头发, 变形的眼睛, 变形的嘴巴",
            num_inference_steps=28,
            seed=42,
            controlnet_params=param,
        )
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_depth.png", threshold=0.95)
        self.pipe.unload_loras()


if __name__ == "__main__":
    unittest.main()
