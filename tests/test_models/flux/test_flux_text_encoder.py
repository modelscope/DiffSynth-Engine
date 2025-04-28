import unittest
import torch
from diffsynth_engine.utils.loader import load_file, save_file
from diffsynth_engine.tokenizers import CLIPTokenizer, T5TokenizerFast
from diffsynth_engine.models.flux import FluxTextEncoder1, FluxTextEncoder2
from diffsynth_engine.utils.constants import FLUX_TOKENIZER_1_CONF_PATH, FLUX_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils.download import ensure_directory_exists
from diffsynth_engine import fetch_model
from tests.common.test_case import TestCase, RUN_EXTRA_TEST


class TestFluxTextEncoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_1 = CLIPTokenizer.from_pretrained(FLUX_TOKENIZER_1_CONF_PATH)
        cls.tokenizer_2 = T5TokenizerFast.from_pretrained(FLUX_TOKENIZER_2_CONF_PATH)

        cls._clip_l_model_path = fetch_model("muse/flux_clip_l", revision="20241209", path="clip_l_bf16.safetensors")
        cls._t5_model_path = fetch_model(
            "muse/google_t5_v1_1_xxl", revision="20241024105236", path="t5xxl_v1_1_bf16.safetensors"
        )
        loaded_state_dict = load_file(cls._clip_l_model_path)
        cls.text_encoder_1 = FluxTextEncoder1.from_state_dict(
            loaded_state_dict, device="cuda:0", dtype=torch.bfloat16
        ).eval()
        loaded_state_dict = load_file(cls._t5_model_path)
        cls.text_encoder_2 = FluxTextEncoder2.from_state_dict(
            loaded_state_dict, device="cuda:0", dtype=torch.bfloat16
        ).eval()
        # use eager attention to aligned with T5
        for encoder in cls.text_encoder_2.encoders:
            encoder.attn.attn_implementation = "eager"
        cls.texts = ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"]

    def test_encoder_1(self):
        text_ids = self.tokenizer_1(self.texts)["input_ids"].to(device="cuda:0")
        with torch.no_grad():
            embeds, pooled_embeds = self.text_encoder_1(text_ids)
            embeds, pooled_embeds = embeds.cpu(), pooled_embeds.cpu()
        expected_tensors = self.get_expect_tensor("flux/flux_text_encoder_1.safetensors")
        expected_embeds = expected_tensors["embeds"]
        expected_pooled_embeds = expected_tensors["pooled_embeds"]
        self.assertTensorEqual(embeds, expected_embeds)
        self.assertTensorEqual(pooled_embeds, expected_pooled_embeds)

    def test_encoder_2(self):
        text_ids = self.tokenizer_2(self.texts)["input_ids"].to(device="cuda:0")
        with torch.no_grad():
            embeds = self.text_encoder_2(text_ids).cpu()
        expected_tensors = self.get_expect_tensor("flux/flux_text_encoder_2.safetensors")
        expected_embeds = expected_tensors["embeds"]
        self.assertTensorEqual(embeds, expected_embeds)

    @unittest.skipUnless(RUN_EXTRA_TEST, "RUN_EXTRA_TEST is not set")
    def test_encoder_1_and_save_tensors(self):
        from transformers.models.clip import CLIPTextModel, CLIPTextConfig

        config = CLIPTextConfig(
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_act="quick_gelu",
        )
        clip_model = CLIPTextModel(config).to(device="cuda:0", dtype=torch.bfloat16).eval()
        loaded_state_dict = load_file(self._clip_l_model_path)
        clip_model.load_state_dict(loaded_state_dict)

        text_ids = self.tokenizer_1(self.texts)["input_ids"].to(device="cuda:0")
        with torch.no_grad():
            clip_model_output = clip_model(text_ids)
            expected_embeds, expected_pooled_embeds = (
                clip_model_output.last_hidden_state,
                clip_model_output.pooler_output,
            )

            text_encoder_1_output = self.text_encoder_1(text_ids)
            embeds, pooled_embeds = text_encoder_1_output
        self.assertTensorEqual(embeds, expected_embeds)
        self.assertTensorEqual(pooled_embeds, expected_pooled_embeds)

        expect = {"embeds": embeds, "pooled_embeds": pooled_embeds}
        save_path = self.testdata_dir / "expect/flux/flux_text_encoder_1.safetensors"
        ensure_directory_exists(save_path)
        save_file(expect, save_path)

    @unittest.skipUnless(RUN_EXTRA_TEST, "RUN_EXTRA_TEST is not set")
    def test_encoder_2_and_save_tensors(self):
        from transformers.models.t5 import T5EncoderModel, T5Config

        config = T5Config(
            d_model=4096,
            d_kv=64,
            d_ff=10240,
            num_layers=24,
            num_heads=64,
            feed_forward_proj="gated-gelu",
        )
        t5_model = T5EncoderModel(config).to(device="cuda:0", dtype=torch.bfloat16).eval()
        load_state_dict = load_file(self._t5_model_path)
        load_state_dict["shared.weight"] = load_state_dict["encoder.embed_tokens.weight"]
        t5_model.load_state_dict(load_state_dict)

        text_ids = self.tokenizer_2(self.texts)["input_ids"].to(device="cuda:0")
        with torch.no_grad():
            expected_embeds = t5_model(text_ids).last_hidden_state
            embeds = self.text_encoder_2(text_ids)
        self.assertTensorEqual(embeds, expected_embeds)

        expect = {"embeds": embeds}
        save_path = self.testdata_dir / "expect/flux/flux_text_encoder_2.safetensors"
        ensure_directory_exists(save_path)
        save_file(expect, save_path)
