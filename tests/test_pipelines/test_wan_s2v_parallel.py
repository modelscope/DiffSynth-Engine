import unittest

from diffsynth_engine import WanSpeech2VideoPipelineConfig
from diffsynth_engine.pipelines import WanSpeech2VideoPipeline
from diffsynth_engine.utils.download import fetch_model

from tests.test_pipelines.test_wan_s2v import TestWanSpeech2Video


class TestWanSpeech2VideoParallel(TestWanSpeech2Video):
    @classmethod
    def setUpClass(cls):
        config = WanSpeech2VideoPipelineConfig(
            model_path=fetch_model(
                "Wan-AI/Wan2.2-S2V-14B",
                path=[
                    "diffusion_pytorch_model-00001-of-00004.safetensors",
                    "diffusion_pytorch_model-00002-of-00004.safetensors",
                    "diffusion_pytorch_model-00003-of-00004.safetensors",
                    "diffusion_pytorch_model-00004-of-00004.safetensors",
                ],
            ),
            parallelism=4,
            use_cfg_parallel=True,
        )
        cls.pipe = WanSpeech2VideoPipeline.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe


if __name__ == "__main__":
    unittest.main()
