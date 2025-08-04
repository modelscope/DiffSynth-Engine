import unittest
import torch

from diffsynth_engine.models.qwen_image.qwen_image_dit import QwenImageDiT
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import TestCase
from tests.common.utils import load_model_checkpoint


class TestQwenImageDiT(TestCase):
    @classmethod
    def setUpClass(cls):
        ckpt_path = fetch_model("MusePublic/Qwen-image", revision="v1", path="transformer/*.safetensors")
        dit_state_dict = load_model_checkpoint(ckpt_path, device="cpu", dtype=torch.bfloat16)
        cls.device = "cuda:0"
        cls.dit = QwenImageDiT.from_state_dict(dit_state_dict, device=cls.device, dtype=torch.bfloat16).eval()

    def test_dit(self):
        input_tensor = self.get_input_tensor("qwen_image/qwen_image_dit_input.safetensors")
        expected_tensor = self.get_expect_tensor("qwen_image/qwen_image_dit_output.safetensors")

        image = self.dit.unpatchify(input_tensor["hidden_states"].to(device=self.device), 1024 // 8, 1024 // 8)
        text = input_tensor["encoder_hidden_states"].to(device=self.device)
        timestep = input_tensor["timestep"].to(device=self.device) * 1000
        txt_seq_lens = torch.tensor([14]).to(device=self.device)
        expected_tensor = self.dit.unpatchify(expected_tensor["output"].to(device=self.device), 1024 // 8, 1024 // 8)
        output = self.dit(image=image, text=text, timestep=timestep, txt_seq_lens=txt_seq_lens)
        self.assertTensorEqual(output, expected_tensor)


if __name__ == "__main__":
    unittest.main()
