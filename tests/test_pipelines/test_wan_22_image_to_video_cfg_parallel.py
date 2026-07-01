import unittest

import numpy as np
import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import WanPipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import VideoTestCase


class TestWan22ImageToVideoPipelineCfgParallel(VideoTestCase):
    """Non-expand_timesteps I2V (Wan2.2 A14B).

    latent_model_input = cat([latents(16), condition(20)], dim=1) -> 36 channels,
    which currently triggers a shape-mismatch bug in _predict_noise_with_cfg's
    zeros_like allocation when use_cfg_parallel=True.
    """

    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        config = WanPipelineConfig(
            model_path=model_path,
            pipeline_class_name="WanImageToVideoPipeline",
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

    def test_image_to_video_cfg_parallel(self):
        image = self.get_input_image("wan_22_i2v_input.png")
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        # DistributedEngine does not expose .pipeline (it lives in workers).
        # For Wan I2V-A14B: vae.scale_factor_spatial=8, transformer.patch_size=(1,2,2) -> mod=16.
        mod_value = 16
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

        video = self.engine.generate(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=81,
            guidance_scale=3.5,
            num_inference_steps=40,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        output_frames = video.frames[0]
        self.assertVideoMsSsimEqualAndSaveFailed(output_frames, "wan/wan_22_i2v.mp4", threshold=0.98, fps=16)


if __name__ == "__main__":
    unittest.main()
