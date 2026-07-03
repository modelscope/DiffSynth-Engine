import unittest

import PIL.Image
import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import WanPipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import VideoTestCase


def prepare_video_and_mask(
    first_img: PIL.Image.Image,
    last_img: PIL.Image.Image,
    height: int,
    width: int,
    num_frames: int,
):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = [first_img]
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask


class TestWanVACEPipeline(VideoTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Wan-AI/Wan2.1-VACE-14B-diffusers")
        config = WanPipelineConfig(
            model_path=model_path,
            pipeline_class_name="WanVACEPipeline",
            parallelism=4,
            use_cfg_parallel=True,
            sp_ulysses_degree=2,
            sp_ring_degree=1,
            flow_shift=5.0,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.cuda.empty_cache()

    def test_vace(self):
        first_frame = self.get_input_image("wan_vace_first_frame.png")
        last_frame = self.get_input_image("wan_vace_last_frame.png")

        prompt = (
            "CG animation style, a small blue bird takes off from the ground, flapping its wings. "
            "The bird's feathers are delicate, with a unique pattern on its chest. "
            "The background shows a blue sky with white clouds under bright sunshine. "
            "The camera follows the bird upward, capturing its flight and the vastness of the sky "
            "from a close-up, low-angle perspective."
        )
        negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
            "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
            "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
            "in the background, walking backwards"
        )

        height = 512
        width = 512
        num_frames = 81
        video, mask = prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)

        result = self.engine.generate(
            video=video,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        output_frames = result.frames[0]
        self.assertVideoMsSsimEqualAndSaveFailed(output_frames, "wan/wan_vace.mp4", threshold=0.93, fps=16)


if __name__ == "__main__":
    unittest.main()
