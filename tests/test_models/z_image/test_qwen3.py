import json
import torch

from diffsynth_engine.models.z_image.qwen3 import (
    Qwen3Model,
    Qwen3Config,
)
from diffsynth_engine.utils.constants import (
    Z_IMAGE_TEXT_ENCODER_CONFIG_FILE,
)
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase
from tests.common.utils import load_model_checkpoint


class TestQwen3(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        with open(Z_IMAGE_TEXT_ENCODER_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = Qwen3Config(**json.load(f))
        model_path = fetch_model("Tongyi-MAI/Z-Image-Turbo", path="text_encoder/*.safetensors")
        loaded_state_dict = load_model_checkpoint(model_path, device="cpu", dtype=torch.bfloat16)
        cls.text_encoder = Qwen3Model.from_state_dict(
            loaded_state_dict,
            config=config,
            device="cuda:0",
            dtype=torch.bfloat16,
        ).eval()

    def test_encode_text(self):
        expects = self.get_input_tensor("z_image/z_image_text_encoder.safetensors")
        self.text_encoder.rotary_emb.inv_freq = self.text_encoder.rotary_emb.original_inv_freq
        self.text_encoder(
            input_ids=expects["text_input_ids"].to("cuda:0"),
            attention_mask=expects["prompt_masks"].to(torch.bool).to("cuda:0"),
        )
