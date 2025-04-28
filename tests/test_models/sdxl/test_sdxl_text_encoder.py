import unittest
import torch
from diffsynth_engine.utils.loader import load_file, save_file

from diffsynth_engine.tokenizers import CLIPTokenizer
from diffsynth_engine.models.sdxl import SDXLTextEncoder, SDXLTextEncoder2
from diffsynth_engine.utils.constants import SDXL_TOKENIZER_CONF_PATH, SDXL_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils.download import ensure_directory_exists
from tests.common.test_case import TestCase, RUN_EXTRA_TEST
from diffsynth_engine import fetch_model


class TestSDXLTextEncoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_1 = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_CONF_PATH)
        cls.tokenizer_2 = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_2_CONF_PATH)

        model_path = fetch_model("muse/sd_xl_base_1.0", revision="20240425120250", path="sd_xl_base_1.0.safetensors")
        loaded_state_dict = load_file(model_path)
        cls.text_encoder_1 = SDXLTextEncoder.from_state_dict(
            loaded_state_dict, device="cuda:0", dtype=torch.float16
        ).eval()
        cls.text_encoder_2 = SDXLTextEncoder2.from_state_dict(
            loaded_state_dict, device="cuda:0", dtype=torch.float16
        ).eval()
        cls.clip_skip = 2
        cls.texts = ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"]

    def test_encoder_1(self):
        text_ids = self.tokenizer_1(self.texts)["input_ids"].to(device="cuda:0")
        with torch.no_grad():
            embeds = self.text_encoder_1(text_ids, clip_skip=self.clip_skip).cpu()
        expected_tensors = self.get_expect_tensor("sdxl/sdxl_text_encoder_1.safetensors")
        expected_embeds = expected_tensors["embeds"]
        self.assertTensorEqual(embeds, expected_embeds)

    def test_encoder_2(self):
        text_ids = self.tokenizer_2(self.texts)["input_ids"].to(device="cuda:0")
        with torch.no_grad():
            embeds, pooled_embeds = self.text_encoder_2(text_ids, clip_skip=self.clip_skip)
            embeds, pooled_embeds = embeds.cpu(), pooled_embeds.cpu()
        expected_tensors = self.get_expect_tensor("sdxl/sdxl_text_encoder_2.safetensors")
        expected_embeds = expected_tensors["embeds"]
        expected_pooled_embeds = expected_tensors["pooled_embeds"]
        self.assertTensorEqual(embeds, expected_embeds)
        self.assertTensorEqual(pooled_embeds, expected_pooled_embeds)

    @unittest.skipUnless(RUN_EXTRA_TEST, "RUN_EXTRA_TEST is not set")
    def test_encoder_1_and_save_tensors(self):
        from transformers.models.clip import CLIPTextModel, CLIPTextConfig

        def _convert(state_dict):
            return {
                key.replace("conditioner.embedders.0.transformer.", ""): param
                for key, param in state_dict.items()
                if key.startswith("conditioner.embedders.0.") and not key.endswith(".position_ids")
            }

        config = CLIPTextConfig(
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_act="quick_gelu",
            eos_token_id=2,
        )
        clip_l_model = CLIPTextModel(config).to(device="cuda:0", dtype=torch.float16).eval()
        loaded_state_dict = load_file(self.model_path)
        clip_l_model.load_state_dict(_convert(loaded_state_dict))

        text_ids = self.tokenizer_1(self.texts)["input_ids"].to(device="cuda:0")
        with torch.no_grad():
            model_output = clip_l_model(text_ids, output_hidden_states=True)
            expected_embeds = model_output.hidden_states[-self.clip_skip]
            embeds = self.text_encoder_1(text_ids, clip_skip=self.clip_skip)
        self.assertTensorEqual(embeds, expected_embeds)

        expect = {"embeds": embeds}
        save_path = self.testdata_dir / "expect/sdxl/sdxl_text_encoder_1.safetensors"
        ensure_directory_exists(save_path)
        save_file(expect, save_path)

    @unittest.skipUnless(RUN_EXTRA_TEST, "RUN_EXTRA_TEST is not set")
    def test_encoder_2_and_save_tensors(self):
        from transformers.models.clip import CLIPTextModelWithProjection, CLIPTextConfig

        def _convert(state_dict):
            rename_dict = {
                "positional_embedding": "text_model.embeddings.position_embedding.weight",
                "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
                "ln_final.bias": "text_model.final_layer_norm.bias",
                "ln_final.weight": "text_model.final_layer_norm.weight",
                "text_projection": "text_projection.weight",
            }
            for i in range(32):
                rename_dict.update(
                    {
                        f"transformer.resblocks.{i}.ln_1.bias": f"text_model.encoder.layers.{i}.layer_norm1.bias",
                        f"transformer.resblocks.{i}.ln_1.weight": f"text_model.encoder.layers.{i}.layer_norm1.weight",
                        f"transformer.resblocks.{i}.ln_2.bias": f"text_model.encoder.layers.{i}.layer_norm2.bias",
                        f"transformer.resblocks.{i}.ln_2.weight": f"text_model.encoder.layers.{i}.layer_norm2.weight",
                        f"transformer.resblocks.{i}.mlp.c_fc.bias": f"text_model.encoder.layers.{i}.mlp.fc1.bias",
                        f"transformer.resblocks.{i}.mlp.c_fc.weight": f"text_model.encoder.layers.{i}.mlp.fc1.weight",
                        f"transformer.resblocks.{i}.mlp.c_proj.bias": f"text_model.encoder.layers.{i}.mlp.fc2.bias",
                        f"transformer.resblocks.{i}.mlp.c_proj.weight": f"text_model.encoder.layers.{i}.mlp.fc2.weight",
                        f"transformer.resblocks.{i}.attn.in_proj_bias": [
                            f"text_model.encoder.layers.{i}.self_attn.q_proj.bias",
                            f"text_model.encoder.layers.{i}.self_attn.k_proj.bias",
                            f"text_model.encoder.layers.{i}.self_attn.v_proj.bias",
                        ],
                        f"transformer.resblocks.{i}.attn.in_proj_weight": [
                            f"text_model.encoder.layers.{i}.self_attn.q_proj.weight",
                            f"text_model.encoder.layers.{i}.self_attn.k_proj.weight",
                            f"text_model.encoder.layers.{i}.self_attn.v_proj.weight",
                        ],
                        f"transformer.resblocks.{i}.attn.out_proj.bias": f"text_model.encoder.layers.{i}.self_attn.out_proj.bias",
                        f"transformer.resblocks.{i}.attn.out_proj.weight": f"text_model.encoder.layers.{i}.self_attn.out_proj.weight",
                    }
                )

            _state_dict = {}
            for key, param in state_dict.items():
                if not key.startswith("conditioner.embedders.1.") or key.endswith(".logit_scale"):
                    continue
                key = key.replace("conditioner.embedders.1.model.", "")
                if key == "text_projection":
                    param = param.T
                if isinstance(rename_dict[key], list):
                    length = param.shape[0] // 3
                    for i, name in enumerate(rename_dict[key]):
                        _state_dict[name] = param[i * length : i * length + length]
                else:
                    _state_dict[rename_dict[key]] = param
            return _state_dict

        config = CLIPTextConfig(
            hidden_size=1280,
            intermediate_size=5120,
            projection_dim=1280,
            num_hidden_layers=32,
            num_attention_heads=20,
            hidden_act="gelu",
            eos_token_id=2,
        )
        clip_g_model = CLIPTextModelWithProjection(config).to(device="cuda:0", dtype=torch.float16).eval()
        load_state_dict = load_file(self.model_path)
        clip_g_model.load_state_dict(_convert(load_state_dict))

        text_ids = self.tokenizer_2(self.texts)["input_ids"].to(device="cuda:0")
        with torch.no_grad():
            model_output = clip_g_model(text_ids, output_hidden_states=True)
            expected_embeds = model_output.hidden_states[-self.clip_skip]
            expected_pooled_embeds = model_output.text_embeds
            embeds, pooled_embeds = self.text_encoder_2(text_ids, clip_skip=self.clip_skip)
        self.assertTensorEqual(embeds, expected_embeds)
        self.assertTensorEqual(pooled_embeds, expected_pooled_embeds)

        expect = {"embeds": embeds, "pooled_embeds": pooled_embeds}
        save_path = self.testdata_dir / "expect/sdxl/sdxl_text_encoder_2.safetensors"
        ensure_directory_exists(save_path)
        save_file(expect, save_path)
