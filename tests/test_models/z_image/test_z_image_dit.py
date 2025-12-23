import torch

from diffsynth_engine.models.z_image.z_image_dit import (
    ZImageDiT,
)
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase
from tests.common.utils import load_model_checkpoint


class TestZImageDit(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Tongyi-MAI/Z-Image-Turbo", path="transformer/*.safetensors")
        loaded_state_dict = load_model_checkpoint(model_path, device="cpu", dtype=torch.bfloat16)
        cls.dit = ZImageDiT.from_state_dict(
            state_dict=loaded_state_dict,
            device="cuda:0",
            dtype=torch.bfloat16,
        ).eval()

    def test_dit(self):
        expects = self.get_input_tensor("z_image/z_image_dit.safetensors")
        latent_model_input_list = [expects["latent_model_input_list"].to("cuda:0")]
        timestep_model_input = expects["timestep_model_input"].to("cuda:0")
        prompt_embeds_model_input = [expects["prompt_embeds_model_input"].to("cuda:0")]
        self.dit(latent_model_input_list, timestep_model_input, prompt_embeds_model_input)
