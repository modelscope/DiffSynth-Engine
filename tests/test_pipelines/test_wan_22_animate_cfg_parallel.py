import unittest

import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import WanPipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import VideoTestCase


class TestWan22AnimatePipelineCfgParallel(VideoTestCase):
    """Wan2.2 Animate under cfg parallel.

    latent_model_input = cat([latents, reference_latents], dim=1) has more than
    16 channels, so this exercises the same zeros_like shape path as I2V-A14B.
    """

    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Wan-AI/Wan2.2-Animate-14B-Diffusers")
        config = WanPipelineConfig(
            model_path=model_path,
            pipeline_class_name="WanAnimatePipeline",
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

    def test_animate_cfg_parallel(self):
        image = self.get_input_image("wan_22_animate_input.png")

        pose_video_reader = self.get_input_video("wan_22_animate_pose.mp4")
        face_video_reader = self.get_input_video("wan_22_animate_face.mp4")
        pose_video = [pose_video_reader[i] for i in range(len(pose_video_reader))]
        face_video = [face_video_reader[i] for i in range(len(face_video_reader))]

        prompt = "People in the video are doing actions."

        video = self.engine.generate(
            image=image,
            pose_video=pose_video,
            face_video=face_video,
            prompt=prompt,
            mode="animate",
            segment_frame_length=77,
            prev_segment_conditioning_frames=1,
            guidance_scale=1.0,
            num_inference_steps=20,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        output_frames = video.frames[0]
        self.assertVideoMsSsimEqualAndSaveFailed(output_frames, "wan/wan_22_animate.mp4", threshold=0.98, fps=30)


if __name__ == "__main__":
    unittest.main()
