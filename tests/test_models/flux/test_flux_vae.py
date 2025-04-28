import unittest
import torch
import numpy as np
from diffsynth_engine.utils.loader import load_file, save_file
from diffsynth_engine.models.flux import FluxVAEEncoder, FluxVAEDecoder
from diffsynth_engine.utils.download import ensure_directory_exists
from diffsynth_engine import fetch_model
from tests.common.test_case import ImageTestCase, RUN_EXTRA_TEST


class TestFluxVAE(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        cls._vae_model_path = fetch_model("muse/flux_vae", revision="20241015120836", path="ae.safetensors")
        loaded_state_dict = load_file(cls._vae_model_path)
        cls.encoder = FluxVAEEncoder.from_state_dict(loaded_state_dict, device="cuda:0", dtype=torch.float32).eval()
        cls.decoder = FluxVAEDecoder.from_state_dict(loaded_state_dict, device="cuda:0", dtype=torch.float32).eval()
        cls.latent_channels = 16
        cls.shift_factor = 0.1159
        cls.scaling_factor = 0.3611
        cls._input_image = cls.get_input_image("wukong_1024_1024.png").convert("RGB")

    def test_encode(self):
        expected_tensor = self.get_expect_tensor("flux/flux_vae.safetensors")
        expected = expected_tensor["encoded"]
        image_tensor = torch.tensor(np.array(self._input_image, dtype=np.float32) * (2 / 255) - 1)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            result = self.encoder(image_tensor).cpu()
        self.assertTensorEqual(result, expected)

    def test_decode(self):
        expected_tensor = self.get_expect_tensor("flux/flux_vae.safetensors")
        latent_tensor, expected = expected_tensor["encoded"], expected_tensor["decoded"]
        latent_tensor = latent_tensor.to("cuda:0")
        with torch.no_grad():
            result = self.decoder(latent_tensor).cpu()
        self.assertTensorEqual(result, expected)

    @unittest.skipUnless(RUN_EXTRA_TEST, "RUN_EXTRA_TEST is not set")
    def test_and_save_tensors(self):
        from diffusers.models import AutoencoderKL

        def _convert(state_dict):
            rename_dict = {
                "decoder.conv_in.bias": "decoder.conv_in.bias",
                "decoder.conv_in.weight": "decoder.conv_in.weight",
                "decoder.norm_out.bias": "decoder.conv_norm_out.bias",
                "decoder.norm_out.weight": "decoder.conv_norm_out.weight",
                "decoder.conv_out.bias": "decoder.conv_out.bias",
                "decoder.conv_out.weight": "decoder.conv_out.weight",
                "decoder.mid.attn_1.norm.bias": "decoder.mid_block.attentions.0.group_norm.bias",
                "decoder.mid.attn_1.norm.weight": "decoder.mid_block.attentions.0.group_norm.weight",
                "decoder.mid.attn_1.k.bias": "decoder.mid_block.attentions.0.to_k.bias",
                "decoder.mid.attn_1.k.weight": "decoder.mid_block.attentions.0.to_k.weight",
                "decoder.mid.attn_1.proj_out.bias": "decoder.mid_block.attentions.0.to_out.0.bias",
                "decoder.mid.attn_1.proj_out.weight": "decoder.mid_block.attentions.0.to_out.0.weight",
                "decoder.mid.attn_1.q.bias": "decoder.mid_block.attentions.0.to_q.bias",
                "decoder.mid.attn_1.q.weight": "decoder.mid_block.attentions.0.to_q.weight",
                "decoder.mid.attn_1.v.bias": "decoder.mid_block.attentions.0.to_v.bias",
                "decoder.mid.attn_1.v.weight": "decoder.mid_block.attentions.0.to_v.weight",
                "decoder.mid.block_1.conv1.bias": "decoder.mid_block.resnets.0.conv1.bias",
                "decoder.mid.block_1.conv1.weight": "decoder.mid_block.resnets.0.conv1.weight",
                "decoder.mid.block_1.conv2.bias": "decoder.mid_block.resnets.0.conv2.bias",
                "decoder.mid.block_1.conv2.weight": "decoder.mid_block.resnets.0.conv2.weight",
                "decoder.mid.block_1.norm1.bias": "decoder.mid_block.resnets.0.norm1.bias",
                "decoder.mid.block_1.norm1.weight": "decoder.mid_block.resnets.0.norm1.weight",
                "decoder.mid.block_1.norm2.bias": "decoder.mid_block.resnets.0.norm2.bias",
                "decoder.mid.block_1.norm2.weight": "decoder.mid_block.resnets.0.norm2.weight",
                "decoder.mid.block_2.conv1.bias": "decoder.mid_block.resnets.1.conv1.bias",
                "decoder.mid.block_2.conv1.weight": "decoder.mid_block.resnets.1.conv1.weight",
                "decoder.mid.block_2.conv2.bias": "decoder.mid_block.resnets.1.conv2.bias",
                "decoder.mid.block_2.conv2.weight": "decoder.mid_block.resnets.1.conv2.weight",
                "decoder.mid.block_2.norm1.bias": "decoder.mid_block.resnets.1.norm1.bias",
                "decoder.mid.block_2.norm1.weight": "decoder.mid_block.resnets.1.norm1.weight",
                "decoder.mid.block_2.norm2.bias": "decoder.mid_block.resnets.1.norm2.bias",
                "decoder.mid.block_2.norm2.weight": "decoder.mid_block.resnets.1.norm2.weight",
                "decoder.up.3.block.0.conv1.bias": "decoder.up_blocks.0.resnets.0.conv1.bias",
                "decoder.up.3.block.0.conv1.weight": "decoder.up_blocks.0.resnets.0.conv1.weight",
                "decoder.up.3.block.0.conv2.bias": "decoder.up_blocks.0.resnets.0.conv2.bias",
                "decoder.up.3.block.0.conv2.weight": "decoder.up_blocks.0.resnets.0.conv2.weight",
                "decoder.up.3.block.0.norm1.bias": "decoder.up_blocks.0.resnets.0.norm1.bias",
                "decoder.up.3.block.0.norm1.weight": "decoder.up_blocks.0.resnets.0.norm1.weight",
                "decoder.up.3.block.0.norm2.bias": "decoder.up_blocks.0.resnets.0.norm2.bias",
                "decoder.up.3.block.0.norm2.weight": "decoder.up_blocks.0.resnets.0.norm2.weight",
                "decoder.up.3.block.1.conv1.bias": "decoder.up_blocks.0.resnets.1.conv1.bias",
                "decoder.up.3.block.1.conv1.weight": "decoder.up_blocks.0.resnets.1.conv1.weight",
                "decoder.up.3.block.1.conv2.bias": "decoder.up_blocks.0.resnets.1.conv2.bias",
                "decoder.up.3.block.1.conv2.weight": "decoder.up_blocks.0.resnets.1.conv2.weight",
                "decoder.up.3.block.1.norm1.bias": "decoder.up_blocks.0.resnets.1.norm1.bias",
                "decoder.up.3.block.1.norm1.weight": "decoder.up_blocks.0.resnets.1.norm1.weight",
                "decoder.up.3.block.1.norm2.bias": "decoder.up_blocks.0.resnets.1.norm2.bias",
                "decoder.up.3.block.1.norm2.weight": "decoder.up_blocks.0.resnets.1.norm2.weight",
                "decoder.up.3.block.2.conv1.bias": "decoder.up_blocks.0.resnets.2.conv1.bias",
                "decoder.up.3.block.2.conv1.weight": "decoder.up_blocks.0.resnets.2.conv1.weight",
                "decoder.up.3.block.2.conv2.bias": "decoder.up_blocks.0.resnets.2.conv2.bias",
                "decoder.up.3.block.2.conv2.weight": "decoder.up_blocks.0.resnets.2.conv2.weight",
                "decoder.up.3.block.2.norm1.bias": "decoder.up_blocks.0.resnets.2.norm1.bias",
                "decoder.up.3.block.2.norm1.weight": "decoder.up_blocks.0.resnets.2.norm1.weight",
                "decoder.up.3.block.2.norm2.bias": "decoder.up_blocks.0.resnets.2.norm2.bias",
                "decoder.up.3.block.2.norm2.weight": "decoder.up_blocks.0.resnets.2.norm2.weight",
                "decoder.up.3.upsample.conv.bias": "decoder.up_blocks.0.upsamplers.0.conv.bias",
                "decoder.up.3.upsample.conv.weight": "decoder.up_blocks.0.upsamplers.0.conv.weight",
                "decoder.up.2.block.0.conv1.bias": "decoder.up_blocks.1.resnets.0.conv1.bias",
                "decoder.up.2.block.0.conv1.weight": "decoder.up_blocks.1.resnets.0.conv1.weight",
                "decoder.up.2.block.0.conv2.bias": "decoder.up_blocks.1.resnets.0.conv2.bias",
                "decoder.up.2.block.0.conv2.weight": "decoder.up_blocks.1.resnets.0.conv2.weight",
                "decoder.up.2.block.0.norm1.bias": "decoder.up_blocks.1.resnets.0.norm1.bias",
                "decoder.up.2.block.0.norm1.weight": "decoder.up_blocks.1.resnets.0.norm1.weight",
                "decoder.up.2.block.0.norm2.bias": "decoder.up_blocks.1.resnets.0.norm2.bias",
                "decoder.up.2.block.0.norm2.weight": "decoder.up_blocks.1.resnets.0.norm2.weight",
                "decoder.up.2.block.1.conv1.bias": "decoder.up_blocks.1.resnets.1.conv1.bias",
                "decoder.up.2.block.1.conv1.weight": "decoder.up_blocks.1.resnets.1.conv1.weight",
                "decoder.up.2.block.1.conv2.bias": "decoder.up_blocks.1.resnets.1.conv2.bias",
                "decoder.up.2.block.1.conv2.weight": "decoder.up_blocks.1.resnets.1.conv2.weight",
                "decoder.up.2.block.1.norm1.bias": "decoder.up_blocks.1.resnets.1.norm1.bias",
                "decoder.up.2.block.1.norm1.weight": "decoder.up_blocks.1.resnets.1.norm1.weight",
                "decoder.up.2.block.1.norm2.bias": "decoder.up_blocks.1.resnets.1.norm2.bias",
                "decoder.up.2.block.1.norm2.weight": "decoder.up_blocks.1.resnets.1.norm2.weight",
                "decoder.up.2.block.2.conv1.bias": "decoder.up_blocks.1.resnets.2.conv1.bias",
                "decoder.up.2.block.2.conv1.weight": "decoder.up_blocks.1.resnets.2.conv1.weight",
                "decoder.up.2.block.2.conv2.bias": "decoder.up_blocks.1.resnets.2.conv2.bias",
                "decoder.up.2.block.2.conv2.weight": "decoder.up_blocks.1.resnets.2.conv2.weight",
                "decoder.up.2.block.2.norm1.bias": "decoder.up_blocks.1.resnets.2.norm1.bias",
                "decoder.up.2.block.2.norm1.weight": "decoder.up_blocks.1.resnets.2.norm1.weight",
                "decoder.up.2.block.2.norm2.bias": "decoder.up_blocks.1.resnets.2.norm2.bias",
                "decoder.up.2.block.2.norm2.weight": "decoder.up_blocks.1.resnets.2.norm2.weight",
                "decoder.up.2.upsample.conv.bias": "decoder.up_blocks.1.upsamplers.0.conv.bias",
                "decoder.up.2.upsample.conv.weight": "decoder.up_blocks.1.upsamplers.0.conv.weight",
                "decoder.up.1.block.0.conv1.bias": "decoder.up_blocks.2.resnets.0.conv1.bias",
                "decoder.up.1.block.0.conv1.weight": "decoder.up_blocks.2.resnets.0.conv1.weight",
                "decoder.up.1.block.0.conv2.bias": "decoder.up_blocks.2.resnets.0.conv2.bias",
                "decoder.up.1.block.0.conv2.weight": "decoder.up_blocks.2.resnets.0.conv2.weight",
                "decoder.up.1.block.0.nin_shortcut.bias": "decoder.up_blocks.2.resnets.0.conv_shortcut.bias",
                "decoder.up.1.block.0.nin_shortcut.weight": "decoder.up_blocks.2.resnets.0.conv_shortcut.weight",
                "decoder.up.1.block.0.norm1.bias": "decoder.up_blocks.2.resnets.0.norm1.bias",
                "decoder.up.1.block.0.norm1.weight": "decoder.up_blocks.2.resnets.0.norm1.weight",
                "decoder.up.1.block.0.norm2.bias": "decoder.up_blocks.2.resnets.0.norm2.bias",
                "decoder.up.1.block.0.norm2.weight": "decoder.up_blocks.2.resnets.0.norm2.weight",
                "decoder.up.1.block.1.conv1.bias": "decoder.up_blocks.2.resnets.1.conv1.bias",
                "decoder.up.1.block.1.conv1.weight": "decoder.up_blocks.2.resnets.1.conv1.weight",
                "decoder.up.1.block.1.conv2.bias": "decoder.up_blocks.2.resnets.1.conv2.bias",
                "decoder.up.1.block.1.conv2.weight": "decoder.up_blocks.2.resnets.1.conv2.weight",
                "decoder.up.1.block.1.norm1.bias": "decoder.up_blocks.2.resnets.1.norm1.bias",
                "decoder.up.1.block.1.norm1.weight": "decoder.up_blocks.2.resnets.1.norm1.weight",
                "decoder.up.1.block.1.norm2.bias": "decoder.up_blocks.2.resnets.1.norm2.bias",
                "decoder.up.1.block.1.norm2.weight": "decoder.up_blocks.2.resnets.1.norm2.weight",
                "decoder.up.1.block.2.conv1.bias": "decoder.up_blocks.2.resnets.2.conv1.bias",
                "decoder.up.1.block.2.conv1.weight": "decoder.up_blocks.2.resnets.2.conv1.weight",
                "decoder.up.1.block.2.conv2.bias": "decoder.up_blocks.2.resnets.2.conv2.bias",
                "decoder.up.1.block.2.conv2.weight": "decoder.up_blocks.2.resnets.2.conv2.weight",
                "decoder.up.1.block.2.norm1.bias": "decoder.up_blocks.2.resnets.2.norm1.bias",
                "decoder.up.1.block.2.norm1.weight": "decoder.up_blocks.2.resnets.2.norm1.weight",
                "decoder.up.1.block.2.norm2.bias": "decoder.up_blocks.2.resnets.2.norm2.bias",
                "decoder.up.1.block.2.norm2.weight": "decoder.up_blocks.2.resnets.2.norm2.weight",
                "decoder.up.1.upsample.conv.bias": "decoder.up_blocks.2.upsamplers.0.conv.bias",
                "decoder.up.1.upsample.conv.weight": "decoder.up_blocks.2.upsamplers.0.conv.weight",
                "decoder.up.0.block.0.conv1.bias": "decoder.up_blocks.3.resnets.0.conv1.bias",
                "decoder.up.0.block.0.conv1.weight": "decoder.up_blocks.3.resnets.0.conv1.weight",
                "decoder.up.0.block.0.conv2.bias": "decoder.up_blocks.3.resnets.0.conv2.bias",
                "decoder.up.0.block.0.conv2.weight": "decoder.up_blocks.3.resnets.0.conv2.weight",
                "decoder.up.0.block.0.nin_shortcut.bias": "decoder.up_blocks.3.resnets.0.conv_shortcut.bias",
                "decoder.up.0.block.0.nin_shortcut.weight": "decoder.up_blocks.3.resnets.0.conv_shortcut.weight",
                "decoder.up.0.block.0.norm1.bias": "decoder.up_blocks.3.resnets.0.norm1.bias",
                "decoder.up.0.block.0.norm1.weight": "decoder.up_blocks.3.resnets.0.norm1.weight",
                "decoder.up.0.block.0.norm2.bias": "decoder.up_blocks.3.resnets.0.norm2.bias",
                "decoder.up.0.block.0.norm2.weight": "decoder.up_blocks.3.resnets.0.norm2.weight",
                "decoder.up.0.block.1.conv1.bias": "decoder.up_blocks.3.resnets.1.conv1.bias",
                "decoder.up.0.block.1.conv1.weight": "decoder.up_blocks.3.resnets.1.conv1.weight",
                "decoder.up.0.block.1.conv2.bias": "decoder.up_blocks.3.resnets.1.conv2.bias",
                "decoder.up.0.block.1.conv2.weight": "decoder.up_blocks.3.resnets.1.conv2.weight",
                "decoder.up.0.block.1.norm1.bias": "decoder.up_blocks.3.resnets.1.norm1.bias",
                "decoder.up.0.block.1.norm1.weight": "decoder.up_blocks.3.resnets.1.norm1.weight",
                "decoder.up.0.block.1.norm2.bias": "decoder.up_blocks.3.resnets.1.norm2.bias",
                "decoder.up.0.block.1.norm2.weight": "decoder.up_blocks.3.resnets.1.norm2.weight",
                "decoder.up.0.block.2.conv1.bias": "decoder.up_blocks.3.resnets.2.conv1.bias",
                "decoder.up.0.block.2.conv1.weight": "decoder.up_blocks.3.resnets.2.conv1.weight",
                "decoder.up.0.block.2.conv2.bias": "decoder.up_blocks.3.resnets.2.conv2.bias",
                "decoder.up.0.block.2.conv2.weight": "decoder.up_blocks.3.resnets.2.conv2.weight",
                "decoder.up.0.block.2.norm1.bias": "decoder.up_blocks.3.resnets.2.norm1.bias",
                "decoder.up.0.block.2.norm1.weight": "decoder.up_blocks.3.resnets.2.norm1.weight",
                "decoder.up.0.block.2.norm2.bias": "decoder.up_blocks.3.resnets.2.norm2.bias",
                "decoder.up.0.block.2.norm2.weight": "decoder.up_blocks.3.resnets.2.norm2.weight",
                "encoder.conv_in.bias": "encoder.conv_in.bias",
                "encoder.conv_in.weight": "encoder.conv_in.weight",
                "encoder.norm_out.bias": "encoder.conv_norm_out.bias",
                "encoder.norm_out.weight": "encoder.conv_norm_out.weight",
                "encoder.conv_out.bias": "encoder.conv_out.bias",
                "encoder.conv_out.weight": "encoder.conv_out.weight",
                "encoder.down.0.downsample.conv.bias": "encoder.down_blocks.0.downsamplers.0.conv.bias",
                "encoder.down.0.downsample.conv.weight": "encoder.down_blocks.0.downsamplers.0.conv.weight",
                "encoder.down.0.block.0.conv1.bias": "encoder.down_blocks.0.resnets.0.conv1.bias",
                "encoder.down.0.block.0.conv1.weight": "encoder.down_blocks.0.resnets.0.conv1.weight",
                "encoder.down.0.block.0.conv2.bias": "encoder.down_blocks.0.resnets.0.conv2.bias",
                "encoder.down.0.block.0.conv2.weight": "encoder.down_blocks.0.resnets.0.conv2.weight",
                "encoder.down.0.block.0.norm1.bias": "encoder.down_blocks.0.resnets.0.norm1.bias",
                "encoder.down.0.block.0.norm1.weight": "encoder.down_blocks.0.resnets.0.norm1.weight",
                "encoder.down.0.block.0.norm2.bias": "encoder.down_blocks.0.resnets.0.norm2.bias",
                "encoder.down.0.block.0.norm2.weight": "encoder.down_blocks.0.resnets.0.norm2.weight",
                "encoder.down.0.block.1.conv1.bias": "encoder.down_blocks.0.resnets.1.conv1.bias",
                "encoder.down.0.block.1.conv1.weight": "encoder.down_blocks.0.resnets.1.conv1.weight",
                "encoder.down.0.block.1.conv2.bias": "encoder.down_blocks.0.resnets.1.conv2.bias",
                "encoder.down.0.block.1.conv2.weight": "encoder.down_blocks.0.resnets.1.conv2.weight",
                "encoder.down.0.block.1.norm1.bias": "encoder.down_blocks.0.resnets.1.norm1.bias",
                "encoder.down.0.block.1.norm1.weight": "encoder.down_blocks.0.resnets.1.norm1.weight",
                "encoder.down.0.block.1.norm2.bias": "encoder.down_blocks.0.resnets.1.norm2.bias",
                "encoder.down.0.block.1.norm2.weight": "encoder.down_blocks.0.resnets.1.norm2.weight",
                "encoder.down.1.downsample.conv.bias": "encoder.down_blocks.1.downsamplers.0.conv.bias",
                "encoder.down.1.downsample.conv.weight": "encoder.down_blocks.1.downsamplers.0.conv.weight",
                "encoder.down.1.block.0.conv1.bias": "encoder.down_blocks.1.resnets.0.conv1.bias",
                "encoder.down.1.block.0.conv1.weight": "encoder.down_blocks.1.resnets.0.conv1.weight",
                "encoder.down.1.block.0.conv2.bias": "encoder.down_blocks.1.resnets.0.conv2.bias",
                "encoder.down.1.block.0.conv2.weight": "encoder.down_blocks.1.resnets.0.conv2.weight",
                "encoder.down.1.block.0.nin_shortcut.bias": "encoder.down_blocks.1.resnets.0.conv_shortcut.bias",
                "encoder.down.1.block.0.nin_shortcut.weight": "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
                "encoder.down.1.block.0.norm1.bias": "encoder.down_blocks.1.resnets.0.norm1.bias",
                "encoder.down.1.block.0.norm1.weight": "encoder.down_blocks.1.resnets.0.norm1.weight",
                "encoder.down.1.block.0.norm2.bias": "encoder.down_blocks.1.resnets.0.norm2.bias",
                "encoder.down.1.block.0.norm2.weight": "encoder.down_blocks.1.resnets.0.norm2.weight",
                "encoder.down.1.block.1.conv1.bias": "encoder.down_blocks.1.resnets.1.conv1.bias",
                "encoder.down.1.block.1.conv1.weight": "encoder.down_blocks.1.resnets.1.conv1.weight",
                "encoder.down.1.block.1.conv2.bias": "encoder.down_blocks.1.resnets.1.conv2.bias",
                "encoder.down.1.block.1.conv2.weight": "encoder.down_blocks.1.resnets.1.conv2.weight",
                "encoder.down.1.block.1.norm1.bias": "encoder.down_blocks.1.resnets.1.norm1.bias",
                "encoder.down.1.block.1.norm1.weight": "encoder.down_blocks.1.resnets.1.norm1.weight",
                "encoder.down.1.block.1.norm2.bias": "encoder.down_blocks.1.resnets.1.norm2.bias",
                "encoder.down.1.block.1.norm2.weight": "encoder.down_blocks.1.resnets.1.norm2.weight",
                "encoder.down.2.downsample.conv.bias": "encoder.down_blocks.2.downsamplers.0.conv.bias",
                "encoder.down.2.downsample.conv.weight": "encoder.down_blocks.2.downsamplers.0.conv.weight",
                "encoder.down.2.block.0.conv1.bias": "encoder.down_blocks.2.resnets.0.conv1.bias",
                "encoder.down.2.block.0.conv1.weight": "encoder.down_blocks.2.resnets.0.conv1.weight",
                "encoder.down.2.block.0.conv2.bias": "encoder.down_blocks.2.resnets.0.conv2.bias",
                "encoder.down.2.block.0.conv2.weight": "encoder.down_blocks.2.resnets.0.conv2.weight",
                "encoder.down.2.block.0.nin_shortcut.bias": "encoder.down_blocks.2.resnets.0.conv_shortcut.bias",
                "encoder.down.2.block.0.nin_shortcut.weight": "encoder.down_blocks.2.resnets.0.conv_shortcut.weight",
                "encoder.down.2.block.0.norm1.bias": "encoder.down_blocks.2.resnets.0.norm1.bias",
                "encoder.down.2.block.0.norm1.weight": "encoder.down_blocks.2.resnets.0.norm1.weight",
                "encoder.down.2.block.0.norm2.bias": "encoder.down_blocks.2.resnets.0.norm2.bias",
                "encoder.down.2.block.0.norm2.weight": "encoder.down_blocks.2.resnets.0.norm2.weight",
                "encoder.down.2.block.1.conv1.bias": "encoder.down_blocks.2.resnets.1.conv1.bias",
                "encoder.down.2.block.1.conv1.weight": "encoder.down_blocks.2.resnets.1.conv1.weight",
                "encoder.down.2.block.1.conv2.bias": "encoder.down_blocks.2.resnets.1.conv2.bias",
                "encoder.down.2.block.1.conv2.weight": "encoder.down_blocks.2.resnets.1.conv2.weight",
                "encoder.down.2.block.1.norm1.bias": "encoder.down_blocks.2.resnets.1.norm1.bias",
                "encoder.down.2.block.1.norm1.weight": "encoder.down_blocks.2.resnets.1.norm1.weight",
                "encoder.down.2.block.1.norm2.bias": "encoder.down_blocks.2.resnets.1.norm2.bias",
                "encoder.down.2.block.1.norm2.weight": "encoder.down_blocks.2.resnets.1.norm2.weight",
                "encoder.down.3.block.0.conv1.bias": "encoder.down_blocks.3.resnets.0.conv1.bias",
                "encoder.down.3.block.0.conv1.weight": "encoder.down_blocks.3.resnets.0.conv1.weight",
                "encoder.down.3.block.0.conv2.bias": "encoder.down_blocks.3.resnets.0.conv2.bias",
                "encoder.down.3.block.0.conv2.weight": "encoder.down_blocks.3.resnets.0.conv2.weight",
                "encoder.down.3.block.0.norm1.bias": "encoder.down_blocks.3.resnets.0.norm1.bias",
                "encoder.down.3.block.0.norm1.weight": "encoder.down_blocks.3.resnets.0.norm1.weight",
                "encoder.down.3.block.0.norm2.bias": "encoder.down_blocks.3.resnets.0.norm2.bias",
                "encoder.down.3.block.0.norm2.weight": "encoder.down_blocks.3.resnets.0.norm2.weight",
                "encoder.down.3.block.1.conv1.bias": "encoder.down_blocks.3.resnets.1.conv1.bias",
                "encoder.down.3.block.1.conv1.weight": "encoder.down_blocks.3.resnets.1.conv1.weight",
                "encoder.down.3.block.1.conv2.bias": "encoder.down_blocks.3.resnets.1.conv2.bias",
                "encoder.down.3.block.1.conv2.weight": "encoder.down_blocks.3.resnets.1.conv2.weight",
                "encoder.down.3.block.1.norm1.bias": "encoder.down_blocks.3.resnets.1.norm1.bias",
                "encoder.down.3.block.1.norm1.weight": "encoder.down_blocks.3.resnets.1.norm1.weight",
                "encoder.down.3.block.1.norm2.bias": "encoder.down_blocks.3.resnets.1.norm2.bias",
                "encoder.down.3.block.1.norm2.weight": "encoder.down_blocks.3.resnets.1.norm2.weight",
                "encoder.mid.attn_1.norm.bias": "encoder.mid_block.attentions.0.group_norm.bias",
                "encoder.mid.attn_1.norm.weight": "encoder.mid_block.attentions.0.group_norm.weight",
                "encoder.mid.attn_1.k.bias": "encoder.mid_block.attentions.0.to_k.bias",
                "encoder.mid.attn_1.k.weight": "encoder.mid_block.attentions.0.to_k.weight",
                "encoder.mid.attn_1.proj_out.bias": "encoder.mid_block.attentions.0.to_out.0.bias",
                "encoder.mid.attn_1.proj_out.weight": "encoder.mid_block.attentions.0.to_out.0.weight",
                "encoder.mid.attn_1.q.bias": "encoder.mid_block.attentions.0.to_q.bias",
                "encoder.mid.attn_1.q.weight": "encoder.mid_block.attentions.0.to_q.weight",
                "encoder.mid.attn_1.v.bias": "encoder.mid_block.attentions.0.to_v.bias",
                "encoder.mid.attn_1.v.weight": "encoder.mid_block.attentions.0.to_v.weight",
                "encoder.mid.block_1.conv1.bias": "encoder.mid_block.resnets.0.conv1.bias",
                "encoder.mid.block_1.conv1.weight": "encoder.mid_block.resnets.0.conv1.weight",
                "encoder.mid.block_1.conv2.bias": "encoder.mid_block.resnets.0.conv2.bias",
                "encoder.mid.block_1.conv2.weight": "encoder.mid_block.resnets.0.conv2.weight",
                "encoder.mid.block_1.norm1.bias": "encoder.mid_block.resnets.0.norm1.bias",
                "encoder.mid.block_1.norm1.weight": "encoder.mid_block.resnets.0.norm1.weight",
                "encoder.mid.block_1.norm2.bias": "encoder.mid_block.resnets.0.norm2.bias",
                "encoder.mid.block_1.norm2.weight": "encoder.mid_block.resnets.0.norm2.weight",
                "encoder.mid.block_2.conv1.bias": "encoder.mid_block.resnets.1.conv1.bias",
                "encoder.mid.block_2.conv1.weight": "encoder.mid_block.resnets.1.conv1.weight",
                "encoder.mid.block_2.conv2.bias": "encoder.mid_block.resnets.1.conv2.bias",
                "encoder.mid.block_2.conv2.weight": "encoder.mid_block.resnets.1.conv2.weight",
                "encoder.mid.block_2.norm1.bias": "encoder.mid_block.resnets.1.norm1.bias",
                "encoder.mid.block_2.norm1.weight": "encoder.mid_block.resnets.1.norm1.weight",
                "encoder.mid.block_2.norm2.bias": "encoder.mid_block.resnets.1.norm2.bias",
                "encoder.mid.block_2.norm2.weight": "encoder.mid_block.resnets.1.norm2.weight",
            }

            _state_dict = {}
            for key, param in state_dict.items():
                if key in rename_dict:
                    if "mid.attn" in key:
                        param = param.squeeze()
                    _state_dict[rename_dict[key]] = param
            return _state_dict

        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            latent_channels=self.latent_channels,
            scaling_factor=self.scaling_factor,
            shift_factor=self.shift_factor,
            use_quant_conv=False,
            use_post_quant_conv=False,
        )
        vae = vae.to(device="cuda:0", dtype=torch.float32).eval()
        loaded_state_dict = load_file(self._vae_model_path)
        vae.load_state_dict(_convert(loaded_state_dict))

        image_tensor = torch.tensor(np.array(self._input_image, dtype=np.float32) * (2 / 255) - 1)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            expected_encoded = vae.encoder(image_tensor)[:, : self.latent_channels]
            expected_encoded = (expected_encoded - self.shift_factor) * self.scaling_factor
            encoded = self.encoder(image_tensor)
            self.assertTensorEqual(encoded, expected_encoded)

            latent = encoded.clone()
            latent = latent / self.scaling_factor + self.shift_factor
            expected_decoded = vae.decoder(latent)
            decoded = self.decoder(encoded)
            self.assertTensorEqual(decoded, expected_decoded)

        expect = {"encoded": encoded, "decoded": decoded}
        save_path = self.testdata_dir / "expect/flux/flux_vae.safetensors"
        ensure_directory_exists(save_path)
        save_file(expect, save_path)
