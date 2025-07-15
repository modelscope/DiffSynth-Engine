import torch
import unittest
import numpy as np

from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine import fetch_model
from tests.common.test_case import VideoTestCase


class TestWanVAEParallel(VideoTestCase):
    @classmethod
    def setUpClass(cls):
        cls._vae_model_path = fetch_model("muse/wan2.1-vae", path="vae.safetensors")
        loaded_state_dict = load_file(cls._vae_model_path)
        vae = WanVideoVAE.from_state_dict(loaded_state_dict)
        cls.vae = ParallelWrapper(vae, cfg_degree=1, sp_ulysses_degree=4, sp_ring_degree=1, tp_degree=1)
        cls._input_video = cls.get_input_video("astronaut_320_320.mp4")

    @classmethod
    def tearDownClass(cls):
        del cls.vae

    def test_encode_parallel(self):
        expected_tensor = self.get_expect_tensor("wan/wan_vae.safetensors")
        expected = expected_tensor["encoded"]
        video_frames = [
            torch.tensor(np.array(frame, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
            for frame in self._input_video.frames
        ]
        video_tensor = torch.stack(video_frames, dim=2)
        with torch.no_grad():
            result = self.vae.encode(video_tensor, device="cuda", tiled=True).cpu()
        self.assertTensorEqual(result, expected)

    def test_decode_parallel(self):
        expected_tensor = self.get_expect_tensor("wan/wan_vae.safetensors")
        latent_tensor, expected = expected_tensor["encoded"], expected_tensor["decoded"]
        with torch.no_grad():
            result = self.vae.decode(latent_tensor, device="cuda", tiled=True)[0].cpu()
        self.assertTensorEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
