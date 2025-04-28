import torch
import numpy as np
from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine import fetch_model
from tests.common.test_case import VideoTestCase


class TestWanVAE(VideoTestCase):
    @classmethod
    def setUpClass(cls):
        cls._vae_model_path = fetch_model("muse/wan2.1-vae", path="vae.safetensors")
        loaded_state_dict = load_file(cls._vae_model_path)
        cls.vae = WanVideoVAE.from_state_dict(loaded_state_dict)
        cls._input_video = cls.get_input_video("astronaut_320_320.mp4")

    def test_encode(self):
        expected_tensor = self.get_expect_tensor("wan/wan_vae.safetensors")
        expected = expected_tensor["encoded"]
        video_frames = [
            torch.tensor(np.array(frame, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
            for frame in self._input_video.frames
        ]
        video_tensor = torch.stack(video_frames, dim=2)
        with torch.no_grad():
            result = self.vae.encode(video_tensor, device="cuda:0", tiled=True).cpu()
        self.assertTensorEqual(result, expected)

    def test_decode(self):
        expected_tensor = self.get_expect_tensor("wan/wan_vae.safetensors")
        latent_tensor, expected = expected_tensor["encoded"], expected_tensor["decoded"]
        with torch.no_grad():
            result = self.vae.decode(latent_tensor, device="cuda:0", tiled=True)[0].cpu()
        self.assertTensorEqual(result, expected)
