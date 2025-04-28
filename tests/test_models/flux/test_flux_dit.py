import unittest
import torch
from diffsynth_engine.utils.loader import load_file, save_file
from einops import repeat

from diffsynth_engine.models.flux import FluxDiT
from diffsynth_engine.models.utils import no_init_weights
from diffsynth_engine.utils.download import ensure_directory_exists
from diffsynth_engine import fetch_model
from tests.common.test_case import TestCase, RUN_EXTRA_TEST


class TestFluxDiT(TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("muse/flux-with-vae", revision="20240902173035", path="flux1-dev-with-vae.safetensors")
        loaded_state_dict = load_file(model_path)
        cls.dit = FluxDiT.from_state_dict(loaded_state_dict, device="cuda:0", dtype=torch.bfloat16).eval()

    def test_dit(self):
        expected_tensor = self.get_expect_tensor("flux/flux_dit.safetensors")
        expected = expected_tensor["output"]

        batch_size, num_channels_latents, height, width = 1, 16, 128, 128
        clip_embed_dim = 768
        t5_seq_len, t5_embed_dim = 512, 4096
        generator = torch.Generator("cuda:0").manual_seed(42)
        latents = torch.randn(
            (batch_size, num_channels_latents, height, width),
            generator=generator,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        timestep = torch.tensor([1000.0], device="cuda:0", dtype=torch.bfloat16)
        pooled_prompt_emb = torch.randn(
            (batch_size, clip_embed_dim), generator=generator, device="cuda:0", dtype=torch.bfloat16
        )
        prompt_emb = torch.randn(
            (batch_size, t5_seq_len, t5_embed_dim), generator=generator, device="cuda:0", dtype=torch.bfloat16
        )
        guidance = torch.tensor([1.0], device="cuda:0", dtype=torch.bfloat16)
        text_ids = torch.zeros((1, t5_seq_len, 3), device="cuda:0", dtype=torch.bfloat16)
        image_ids = self.dit.prepare_image_ids(latents)

        with torch.no_grad():
            output = self.dit(
                latents,
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                guidance=guidance,
                image_ids=image_ids,
                text_ids=text_ids,
            ).cpu()
            self.assertTensorEqual(output, expected)

    @unittest.skipUnless(RUN_EXTRA_TEST, "RUN_EXTRA_TEST is not set")
    def test_and_save_tensors(self):
        from diffusers import FluxPipeline, FluxTransformer2DModel

        def _convert(state_dict):
            rename_dict = {
                "txt_in.bias": "context_embedder.bias",
                "txt_in.weight": "context_embedder.weight",
                "img_in.bias": "x_embedder.bias",
                "img_in.weight": "x_embedder.weight",
                "time_in.in_layer.bias": "time_text_embed.timestep_embedder.linear_1.bias",
                "time_in.in_layer.weight": "time_text_embed.timestep_embedder.linear_1.weight",
                "time_in.out_layer.bias": "time_text_embed.timestep_embedder.linear_2.bias",
                "time_in.out_layer.weight": "time_text_embed.timestep_embedder.linear_2.weight",
                "guidance_in.in_layer.bias": "time_text_embed.guidance_embedder.linear_1.bias",
                "guidance_in.in_layer.weight": "time_text_embed.guidance_embedder.linear_1.weight",
                "guidance_in.out_layer.bias": "time_text_embed.guidance_embedder.linear_2.bias",
                "guidance_in.out_layer.weight": "time_text_embed.guidance_embedder.linear_2.weight",
                "vector_in.in_layer.bias": "time_text_embed.text_embedder.linear_1.bias",
                "vector_in.in_layer.weight": "time_text_embed.text_embedder.linear_1.weight",
                "vector_in.out_layer.bias": "time_text_embed.text_embedder.linear_2.bias",
                "vector_in.out_layer.weight": "time_text_embed.text_embedder.linear_2.weight",
                "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
                "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
                "final_layer.linear.bias": "proj_out.bias",
                "final_layer.linear.weight": "proj_out.weight",
            }
            double_blocks_rename_dict = {
                "img_attn.norm.query_norm.scale": "attn.norm_q.weight",
                "img_attn.norm.key_norm.scale": "attn.norm_k.weight",
                "img_attn.qkv.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
                "img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
                "img_attn.proj.bias": "attn.to_out.0.bias",
                "img_attn.proj.weight": "attn.to_out.0.weight",
                "txt_attn.norm.query_norm.scale": "attn.norm_added_q.weight",
                "txt_attn.norm.key_norm.scale": "attn.norm_added_k.weight",
                "txt_attn.qkv.bias": ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
                "txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
                "txt_attn.proj.bias": "attn.to_add_out.bias",
                "txt_attn.proj.weight": "attn.to_add_out.weight",
                "img_mlp.0.bias": "ff.net.0.proj.bias",
                "img_mlp.0.weight": "ff.net.0.proj.weight",
                "img_mlp.2.bias": "ff.net.2.bias",
                "img_mlp.2.weight": "ff.net.2.weight",
                "img_mod.lin.bias": "norm1.linear.bias",
                "img_mod.lin.weight": "norm1.linear.weight",
                "txt_mlp.0.bias": "ff_context.net.0.proj.bias",
                "txt_mlp.0.weight": "ff_context.net.0.proj.weight",
                "txt_mlp.2.bias": "ff_context.net.2.bias",
                "txt_mlp.2.weight": "ff_context.net.2.weight",
                "txt_mod.lin.bias": "norm1_context.linear.bias",
                "txt_mod.lin.weight": "norm1_context.linear.weight",
            }
            single_blocks_rename_dict = {
                "norm.query_norm.scale": "attn.norm_q.weight",
                "norm.key_norm.scale": "attn.norm_k.weight",
                "linear1.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
                "linear1.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
                "linear2.bias": "proj_out.bias",
                "linear2.weight": "proj_out.weight",
                "modulation.lin.bias": "norm.linear.bias",
                "modulation.lin.weight": "norm.linear.weight",
            }

            _state_dict = {}
            for key, param in state_dict.items():
                if key in rename_dict:
                    if key.startswith("final_layer.adaLN_modulation.1."):
                        param = torch.concat([param[3072:], param[:3072]], dim=0)
                    _state_dict[rename_dict[key]] = param
                elif key.startswith("double_blocks."):
                    sp = key.split(".")
                    block_id, suffix = sp[1], ".".join(sp[2:])
                    if isinstance(double_blocks_rename_dict[suffix], str):
                        key = f"transformer_blocks.{block_id}.{double_blocks_rename_dict[suffix]}"
                        _state_dict[key] = param
                    else:
                        params = param[:3072], param[3072:6144], param[6144:]
                        for s, p in zip(double_blocks_rename_dict[suffix], params):
                            _state_dict[f"transformer_blocks.{block_id}.{s}"] = p
                elif key.startswith("single_blocks."):
                    sp = key.split(".")
                    block_id, suffix = sp[1], ".".join(sp[2:])
                    if isinstance(single_blocks_rename_dict[suffix], str):
                        key = f"single_transformer_blocks.{block_id}.{single_blocks_rename_dict[suffix]}"
                        _state_dict[key] = param
                    else:
                        params = param[:3072], param[3072:6144], param[6144:9216], param[9216:]
                        for s, p in zip(single_blocks_rename_dict[suffix], params):
                            _state_dict[f"single_transformer_blocks.{block_id}.{s}"] = p
            return _state_dict

        with no_init_weights():
            model = FluxTransformer2DModel(guidance_embeds=True)
            model = model.to(device="cuda:0", dtype=torch.bfloat16).eval()
        loaded_state_dict = load_file(self._model_path)
        model.load_state_dict(_convert(loaded_state_dict))

        batch_size, num_channels_latents, height, width = 1, 16, 128, 128
        clip_embed_dim = 768
        t5_seq_len, t5_embed_dim = 512, 4096
        generator = torch.Generator("cuda:0").manual_seed(42)
        latents = torch.randn(
            (batch_size, num_channels_latents, height, width),
            generator=generator,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        timestep = torch.tensor([1000.0], device="cuda:0", dtype=torch.bfloat16)
        pooled_prompt_emb = torch.randn(
            (batch_size, clip_embed_dim), generator=generator, device="cuda:0", dtype=torch.bfloat16
        )
        prompt_emb = torch.randn(
            (batch_size, t5_seq_len, t5_embed_dim), generator=generator, device="cuda:0", dtype=torch.bfloat16
        )
        guidance = torch.tensor([1.0], device="cuda:0", dtype=torch.bfloat16)
        text_ids = torch.zeros((t5_seq_len, 3), device="cuda:0", dtype=torch.bfloat16)

        packed_latents = FluxPipeline._pack_latents(latents, batch_size, num_channels_latents, height, width)
        image_ids = FluxPipeline._prepare_latent_image_ids(
            batch_size, height, width, device="cuda:0", dtype=torch.bfloat16
        )

        with torch.no_grad():
            output = model(
                packed_latents,
                encoder_hidden_states=prompt_emb,
                pooled_projections=pooled_prompt_emb,
                timestep=timestep / 1000,
                img_ids=image_ids,
                txt_ids=text_ids,
                guidance=guidance,
            )
            expected = FluxPipeline._unpack_latents(output.sample, height, width, 2)

            image_ids = repeat(image_ids, "s d -> b s d", b=batch_size)
            text_ids = repeat(text_ids, "s d -> b s d", b=batch_size)
            output = self.dit(
                latents,
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                guidance=guidance,
                image_ids=image_ids,
                text_ids=text_ids,
            )
            self.assertTensorEqual(output, expected)

        expect = {"output": output}
        save_path = self.testdata_dir / "expect/flux/flux_dit.safetensors"
        ensure_directory_exists(save_path)
        save_file(expect, save_path)
