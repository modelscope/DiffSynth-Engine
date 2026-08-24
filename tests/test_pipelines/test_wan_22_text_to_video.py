import unittest

import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import WanPipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import VideoTestCase


class TestWan22TextToVideoPipeline(VideoTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        config = WanPipelineConfig(
            model_path=model_path,
            pipeline_class_name="WanTextToVideoPipeline",
            parallelism=4,
            use_cfg_parallel=True,
            sp_ulysses_degree=2,
            sp_ring_degree=1,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.cuda.empty_cache()

    def test_text_to_video(self):
        prompt = (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
        )
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

        video = self.engine.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=81,
            width=1280,
            height=720,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            num_inference_steps=40,
            generator=torch.Generator(device="cpu").manual_seed(2024),
        )

        output_frames = video.frames[0]
        self.assertVideoMsSsimEqualAndSaveFailed(output_frames, "wan/wan_22_t2v.mp4", threshold=0.98, fps=16)


if __name__ == "__main__":
    unittest.main()
