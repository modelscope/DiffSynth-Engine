import torch
import torch.nn as nn
from typing import Dict

from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.basic.unet_helper import (
    ResnetBlock,
    AttentionBlock,
    PushBlock,
    DownSampler,
)


class ControlNetConditioningLayer(nn.Module):
    def __init__(self, channels=(3, 16, 32, 96, 256, 320), device="cuda:0", dtype=torch.float16):
        super().__init__()
        self.blocks = torch.nn.ModuleList([])
        self.blocks.append(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, device=device, dtype=dtype)
        )
        self.blocks.append(torch.nn.SiLU())
        for i in range(1, len(channels) - 2):
            self.blocks.append(
                torch.nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1, device=device, dtype=dtype)
            )
            self.blocks.append(torch.nn.SiLU())
            self.blocks.append(
                torch.nn.Conv2d(
                    channels[i], channels[i + 1], kernel_size=3, padding=1, stride=2, device=device, dtype=dtype
                )
            )
            self.blocks.append(torch.nn.SiLU())
        self.blocks.append(
            torch.nn.Conv2d(channels[-2], channels[-1], kernel_size=3, padding=1, device=device, dtype=dtype)
        )

    def forward(self, conditioning):
        for block in self.blocks:
            conditioning = block(conditioning)
        return conditioning


class SDControlNetStateDictConverter(StateDictConverter):
    def __init__(self):
        super().__init__()

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # architecture
        block_types = [
            "ResnetBlock",
            "AttentionBlock",
            "PushBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PushBlock",
            "DownSampler",
            "PushBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PushBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PushBlock",
            "DownSampler",
            "PushBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PushBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PushBlock",
            "DownSampler",
            "PushBlock",
            "ResnetBlock",
            "PushBlock",
            "ResnetBlock",
            "PushBlock",
            "ResnetBlock",
            "AttentionBlock",
            "ResnetBlock",
            "PopBlock",
            "ResnetBlock",
            "PopBlock",
            "ResnetBlock",
            "PopBlock",
            "ResnetBlock",
            "UpSampler",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
            "UpSampler",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
            "UpSampler",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
            "PopBlock",
            "ResnetBlock",
            "AttentionBlock",
        ]

        # controlnet_rename_dict
        controlnet_rename_dict = {
            "controlnet_cond_embedding.conv_in.weight": "controlnet_conv_in.blocks.0.weight",
            "controlnet_cond_embedding.conv_in.bias": "controlnet_conv_in.blocks.0.bias",
            "controlnet_cond_embedding.blocks.0.weight": "controlnet_conv_in.blocks.2.weight",
            "controlnet_cond_embedding.blocks.0.bias": "controlnet_conv_in.blocks.2.bias",
            "controlnet_cond_embedding.blocks.1.weight": "controlnet_conv_in.blocks.4.weight",
            "controlnet_cond_embedding.blocks.1.bias": "controlnet_conv_in.blocks.4.bias",
            "controlnet_cond_embedding.blocks.2.weight": "controlnet_conv_in.blocks.6.weight",
            "controlnet_cond_embedding.blocks.2.bias": "controlnet_conv_in.blocks.6.bias",
            "controlnet_cond_embedding.blocks.3.weight": "controlnet_conv_in.blocks.8.weight",
            "controlnet_cond_embedding.blocks.3.bias": "controlnet_conv_in.blocks.8.bias",
            "controlnet_cond_embedding.blocks.4.weight": "controlnet_conv_in.blocks.10.weight",
            "controlnet_cond_embedding.blocks.4.bias": "controlnet_conv_in.blocks.10.bias",
            "controlnet_cond_embedding.blocks.5.weight": "controlnet_conv_in.blocks.12.weight",
            "controlnet_cond_embedding.blocks.5.bias": "controlnet_conv_in.blocks.12.bias",
            "controlnet_cond_embedding.conv_out.weight": "controlnet_conv_in.blocks.14.weight",
            "controlnet_cond_embedding.conv_out.bias": "controlnet_conv_in.blocks.14.bias",
        }

        # Rename each parameter
        name_list = sorted([name for name in state_dict])
        rename_dict = {}
        block_id = {"ResnetBlock": -1, "AttentionBlock": -1, "DownSampler": -1, "UpSampler": -1}
        last_block_type_with_id = {"ResnetBlock": "", "AttentionBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            if names[0] in ["conv_in", "conv_norm_out", "conv_out"]:
                pass
            elif name in controlnet_rename_dict:
                names = controlnet_rename_dict[name].split(".")
            elif names[0] == "controlnet_down_blocks":
                names[0] = "controlnet_blocks"
            elif names[0] == "controlnet_mid_block":
                names = ["controlnet_blocks", "12", names[-1]]
            elif names[0] in ["time_embedding", "add_embedding"]:
                if names[0] == "add_embedding":
                    names[0] = "add_time_embedding"
                names[1] = {"linear_1": "timestep_embedder.0", "linear_2": "timestep_embedder.2"}[names[1]]
            elif names[0] in ["down_blocks", "mid_block", "up_blocks"]:
                if names[0] == "mid_block":
                    names.insert(1, "0")
                block_type = {
                    "resnets": "ResnetBlock",
                    "attentions": "AttentionBlock",
                    "downsamplers": "DownSampler",
                    "upsamplers": "UpSampler",
                }[names[2]]
                block_type_with_id = ".".join(names[:4])
                if block_type_with_id != last_block_type_with_id[block_type]:
                    block_id[block_type] += 1
                last_block_type_with_id[block_type] = block_type_with_id
                while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                    block_id[block_type] += 1
                block_type_with_id = ".".join(names[:4])
                names = ["blocks", str(block_id[block_type])] + names[4:]
                if "ff" in names:
                    ff_index = names.index("ff")
                    component = ".".join(names[ff_index : ff_index + 3])
                    component = {"ff.net.0": "act_fn", "ff.net.2": "ff"}[component]
                    names = names[:ff_index] + [component] + names[ff_index + 3 :]
                if "to_out" in names:
                    names.pop(names.index("to_out") + 1)
            else:
                raise ValueError(f"Unknown parameters: {name}")
            rename_dict[name] = ".".join(names)

        # Convert state_dict
        state_dict_ = {}
        for name, param in state_dict.items():
            if ".proj_in." in name or ".proj_out." in name:
                param = param.squeeze()
            if rename_dict[name] in [
                "controlnet_blocks.1.bias",
                "controlnet_blocks.2.bias",
                "controlnet_blocks.3.bias",
                "controlnet_blocks.5.bias",
                "controlnet_blocks.6.bias",
                "controlnet_blocks.8.bias",
                "controlnet_blocks.9.bias",
                "controlnet_blocks.10.bias",
                "controlnet_blocks.11.bias",
                "controlnet_blocks.12.bias",
            ]:
                continue
            state_dict_[rename_dict[name]] = param
        return state_dict_

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = {
            "control_model.time_embed.0.weight": "time_embedding.timestep_embedder.0.weight",
            "control_model.time_embed.0.bias": "time_embedding.timestep_embedder.0.bias",
            "control_model.time_embed.2.weight": "time_embedding.timestep_embedder.2.weight",
            "control_model.time_embed.2.bias": "time_embedding.timestep_embedder.2.bias",
            "control_model.input_blocks.0.0.weight": "conv_in.weight",
            "control_model.input_blocks.0.0.bias": "conv_in.bias",
            "control_model.input_blocks.1.0.in_layers.0.weight": "blocks.0.norm1.weight",
            "control_model.input_blocks.1.0.in_layers.0.bias": "blocks.0.norm1.bias",
            "control_model.input_blocks.1.0.in_layers.2.weight": "blocks.0.conv1.weight",
            "control_model.input_blocks.1.0.in_layers.2.bias": "blocks.0.conv1.bias",
            "control_model.input_blocks.1.0.emb_layers.1.weight": "blocks.0.time_emb_proj.weight",
            "control_model.input_blocks.1.0.emb_layers.1.bias": "blocks.0.time_emb_proj.bias",
            "control_model.input_blocks.1.0.out_layers.0.weight": "blocks.0.norm2.weight",
            "control_model.input_blocks.1.0.out_layers.0.bias": "blocks.0.norm2.bias",
            "control_model.input_blocks.1.0.out_layers.3.weight": "blocks.0.conv2.weight",
            "control_model.input_blocks.1.0.out_layers.3.bias": "blocks.0.conv2.bias",
            "control_model.input_blocks.1.1.norm.weight": "blocks.1.norm.weight",
            "control_model.input_blocks.1.1.norm.bias": "blocks.1.norm.bias",
            "control_model.input_blocks.1.1.proj_in.weight": "blocks.1.proj_in.weight",
            "control_model.input_blocks.1.1.proj_in.bias": "blocks.1.proj_in.bias",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight": "blocks.1.transformer_blocks.0.attn1.to_q.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight": "blocks.1.transformer_blocks.0.attn1.to_k.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight": "blocks.1.transformer_blocks.0.attn1.to_v.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.1.transformer_blocks.0.attn1.to_out.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.1.transformer_blocks.0.attn1.to_out.bias",
            "control_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.1.transformer_blocks.0.act_fn.proj.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.1.transformer_blocks.0.act_fn.proj.bias",
            "control_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight": "blocks.1.transformer_blocks.0.ff.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias": "blocks.1.transformer_blocks.0.ff.bias",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight": "blocks.1.transformer_blocks.0.attn2.to_q.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": "blocks.1.transformer_blocks.0.attn2.to_k.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight": "blocks.1.transformer_blocks.0.attn2.to_v.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.1.transformer_blocks.0.attn2.to_out.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.1.transformer_blocks.0.attn2.to_out.bias",
            "control_model.input_blocks.1.1.transformer_blocks.0.norm1.weight": "blocks.1.transformer_blocks.0.norm1.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.norm1.bias": "blocks.1.transformer_blocks.0.norm1.bias",
            "control_model.input_blocks.1.1.transformer_blocks.0.norm2.weight": "blocks.1.transformer_blocks.0.norm2.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.norm2.bias": "blocks.1.transformer_blocks.0.norm2.bias",
            "control_model.input_blocks.1.1.transformer_blocks.0.norm3.weight": "blocks.1.transformer_blocks.0.norm3.weight",
            "control_model.input_blocks.1.1.transformer_blocks.0.norm3.bias": "blocks.1.transformer_blocks.0.norm3.bias",
            "control_model.input_blocks.1.1.proj_out.weight": "blocks.1.proj_out.weight",
            "control_model.input_blocks.1.1.proj_out.bias": "blocks.1.proj_out.bias",
            "control_model.input_blocks.2.0.in_layers.0.weight": "blocks.3.norm1.weight",
            "control_model.input_blocks.2.0.in_layers.0.bias": "blocks.3.norm1.bias",
            "control_model.input_blocks.2.0.in_layers.2.weight": "blocks.3.conv1.weight",
            "control_model.input_blocks.2.0.in_layers.2.bias": "blocks.3.conv1.bias",
            "control_model.input_blocks.2.0.emb_layers.1.weight": "blocks.3.time_emb_proj.weight",
            "control_model.input_blocks.2.0.emb_layers.1.bias": "blocks.3.time_emb_proj.bias",
            "control_model.input_blocks.2.0.out_layers.0.weight": "blocks.3.norm2.weight",
            "control_model.input_blocks.2.0.out_layers.0.bias": "blocks.3.norm2.bias",
            "control_model.input_blocks.2.0.out_layers.3.weight": "blocks.3.conv2.weight",
            "control_model.input_blocks.2.0.out_layers.3.bias": "blocks.3.conv2.bias",
            "control_model.input_blocks.2.1.norm.weight": "blocks.4.norm.weight",
            "control_model.input_blocks.2.1.norm.bias": "blocks.4.norm.bias",
            "control_model.input_blocks.2.1.proj_in.weight": "blocks.4.proj_in.weight",
            "control_model.input_blocks.2.1.proj_in.bias": "blocks.4.proj_in.bias",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q.weight": "blocks.4.transformer_blocks.0.attn1.to_q.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k.weight": "blocks.4.transformer_blocks.0.attn1.to_k.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v.weight": "blocks.4.transformer_blocks.0.attn1.to_v.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.4.transformer_blocks.0.attn1.to_out.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.4.transformer_blocks.0.attn1.to_out.bias",
            "control_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.4.transformer_blocks.0.act_fn.proj.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.4.transformer_blocks.0.act_fn.proj.bias",
            "control_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.weight": "blocks.4.transformer_blocks.0.ff.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.bias": "blocks.4.transformer_blocks.0.ff.bias",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q.weight": "blocks.4.transformer_blocks.0.attn2.to_q.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": "blocks.4.transformer_blocks.0.attn2.to_k.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight": "blocks.4.transformer_blocks.0.attn2.to_v.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.4.transformer_blocks.0.attn2.to_out.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.4.transformer_blocks.0.attn2.to_out.bias",
            "control_model.input_blocks.2.1.transformer_blocks.0.norm1.weight": "blocks.4.transformer_blocks.0.norm1.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.norm1.bias": "blocks.4.transformer_blocks.0.norm1.bias",
            "control_model.input_blocks.2.1.transformer_blocks.0.norm2.weight": "blocks.4.transformer_blocks.0.norm2.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.norm2.bias": "blocks.4.transformer_blocks.0.norm2.bias",
            "control_model.input_blocks.2.1.transformer_blocks.0.norm3.weight": "blocks.4.transformer_blocks.0.norm3.weight",
            "control_model.input_blocks.2.1.transformer_blocks.0.norm3.bias": "blocks.4.transformer_blocks.0.norm3.bias",
            "control_model.input_blocks.2.1.proj_out.weight": "blocks.4.proj_out.weight",
            "control_model.input_blocks.2.1.proj_out.bias": "blocks.4.proj_out.bias",
            "control_model.input_blocks.3.0.op.weight": "blocks.6.conv.weight",
            "control_model.input_blocks.3.0.op.bias": "blocks.6.conv.bias",
            "control_model.input_blocks.4.0.in_layers.0.weight": "blocks.8.norm1.weight",
            "control_model.input_blocks.4.0.in_layers.0.bias": "blocks.8.norm1.bias",
            "control_model.input_blocks.4.0.in_layers.2.weight": "blocks.8.conv1.weight",
            "control_model.input_blocks.4.0.in_layers.2.bias": "blocks.8.conv1.bias",
            "control_model.input_blocks.4.0.emb_layers.1.weight": "blocks.8.time_emb_proj.weight",
            "control_model.input_blocks.4.0.emb_layers.1.bias": "blocks.8.time_emb_proj.bias",
            "control_model.input_blocks.4.0.out_layers.0.weight": "blocks.8.norm2.weight",
            "control_model.input_blocks.4.0.out_layers.0.bias": "blocks.8.norm2.bias",
            "control_model.input_blocks.4.0.out_layers.3.weight": "blocks.8.conv2.weight",
            "control_model.input_blocks.4.0.out_layers.3.bias": "blocks.8.conv2.bias",
            "control_model.input_blocks.4.0.skip_connection.weight": "blocks.8.conv_shortcut.weight",
            "control_model.input_blocks.4.0.skip_connection.bias": "blocks.8.conv_shortcut.bias",
            "control_model.input_blocks.4.1.norm.weight": "blocks.9.norm.weight",
            "control_model.input_blocks.4.1.norm.bias": "blocks.9.norm.bias",
            "control_model.input_blocks.4.1.proj_in.weight": "blocks.9.proj_in.weight",
            "control_model.input_blocks.4.1.proj_in.bias": "blocks.9.proj_in.bias",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": "blocks.9.transformer_blocks.0.attn1.to_q.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": "blocks.9.transformer_blocks.0.attn1.to_k.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": "blocks.9.transformer_blocks.0.attn1.to_v.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.9.transformer_blocks.0.attn1.to_out.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.9.transformer_blocks.0.attn1.to_out.bias",
            "control_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.9.transformer_blocks.0.act_fn.proj.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.9.transformer_blocks.0.act_fn.proj.bias",
            "control_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight": "blocks.9.transformer_blocks.0.ff.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias": "blocks.9.transformer_blocks.0.ff.bias",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": "blocks.9.transformer_blocks.0.attn2.to_q.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "blocks.9.transformer_blocks.0.attn2.to_k.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "blocks.9.transformer_blocks.0.attn2.to_v.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.9.transformer_blocks.0.attn2.to_out.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.9.transformer_blocks.0.attn2.to_out.bias",
            "control_model.input_blocks.4.1.transformer_blocks.0.norm1.weight": "blocks.9.transformer_blocks.0.norm1.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.norm1.bias": "blocks.9.transformer_blocks.0.norm1.bias",
            "control_model.input_blocks.4.1.transformer_blocks.0.norm2.weight": "blocks.9.transformer_blocks.0.norm2.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.norm2.bias": "blocks.9.transformer_blocks.0.norm2.bias",
            "control_model.input_blocks.4.1.transformer_blocks.0.norm3.weight": "blocks.9.transformer_blocks.0.norm3.weight",
            "control_model.input_blocks.4.1.transformer_blocks.0.norm3.bias": "blocks.9.transformer_blocks.0.norm3.bias",
            "control_model.input_blocks.4.1.proj_out.weight": "blocks.9.proj_out.weight",
            "control_model.input_blocks.4.1.proj_out.bias": "blocks.9.proj_out.bias",
            "control_model.input_blocks.5.0.in_layers.0.weight": "blocks.11.norm1.weight",
            "control_model.input_blocks.5.0.in_layers.0.bias": "blocks.11.norm1.bias",
            "control_model.input_blocks.5.0.in_layers.2.weight": "blocks.11.conv1.weight",
            "control_model.input_blocks.5.0.in_layers.2.bias": "blocks.11.conv1.bias",
            "control_model.input_blocks.5.0.emb_layers.1.weight": "blocks.11.time_emb_proj.weight",
            "control_model.input_blocks.5.0.emb_layers.1.bias": "blocks.11.time_emb_proj.bias",
            "control_model.input_blocks.5.0.out_layers.0.weight": "blocks.11.norm2.weight",
            "control_model.input_blocks.5.0.out_layers.0.bias": "blocks.11.norm2.bias",
            "control_model.input_blocks.5.0.out_layers.3.weight": "blocks.11.conv2.weight",
            "control_model.input_blocks.5.0.out_layers.3.bias": "blocks.11.conv2.bias",
            "control_model.input_blocks.5.1.norm.weight": "blocks.12.norm.weight",
            "control_model.input_blocks.5.1.norm.bias": "blocks.12.norm.bias",
            "control_model.input_blocks.5.1.proj_in.weight": "blocks.12.proj_in.weight",
            "control_model.input_blocks.5.1.proj_in.bias": "blocks.12.proj_in.bias",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": "blocks.12.transformer_blocks.0.attn1.to_q.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": "blocks.12.transformer_blocks.0.attn1.to_k.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": "blocks.12.transformer_blocks.0.attn1.to_v.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.12.transformer_blocks.0.attn1.to_out.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.12.transformer_blocks.0.attn1.to_out.bias",
            "control_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.12.transformer_blocks.0.act_fn.proj.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.12.transformer_blocks.0.act_fn.proj.bias",
            "control_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight": "blocks.12.transformer_blocks.0.ff.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias": "blocks.12.transformer_blocks.0.ff.bias",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": "blocks.12.transformer_blocks.0.attn2.to_q.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "blocks.12.transformer_blocks.0.attn2.to_k.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "blocks.12.transformer_blocks.0.attn2.to_v.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.12.transformer_blocks.0.attn2.to_out.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.12.transformer_blocks.0.attn2.to_out.bias",
            "control_model.input_blocks.5.1.transformer_blocks.0.norm1.weight": "blocks.12.transformer_blocks.0.norm1.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.norm1.bias": "blocks.12.transformer_blocks.0.norm1.bias",
            "control_model.input_blocks.5.1.transformer_blocks.0.norm2.weight": "blocks.12.transformer_blocks.0.norm2.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.norm2.bias": "blocks.12.transformer_blocks.0.norm2.bias",
            "control_model.input_blocks.5.1.transformer_blocks.0.norm3.weight": "blocks.12.transformer_blocks.0.norm3.weight",
            "control_model.input_blocks.5.1.transformer_blocks.0.norm3.bias": "blocks.12.transformer_blocks.0.norm3.bias",
            "control_model.input_blocks.5.1.proj_out.weight": "blocks.12.proj_out.weight",
            "control_model.input_blocks.5.1.proj_out.bias": "blocks.12.proj_out.bias",
            "control_model.input_blocks.6.0.op.weight": "blocks.14.conv.weight",
            "control_model.input_blocks.6.0.op.bias": "blocks.14.conv.bias",
            "control_model.input_blocks.7.0.in_layers.0.weight": "blocks.16.norm1.weight",
            "control_model.input_blocks.7.0.in_layers.0.bias": "blocks.16.norm1.bias",
            "control_model.input_blocks.7.0.in_layers.2.weight": "blocks.16.conv1.weight",
            "control_model.input_blocks.7.0.in_layers.2.bias": "blocks.16.conv1.bias",
            "control_model.input_blocks.7.0.emb_layers.1.weight": "blocks.16.time_emb_proj.weight",
            "control_model.input_blocks.7.0.emb_layers.1.bias": "blocks.16.time_emb_proj.bias",
            "control_model.input_blocks.7.0.out_layers.0.weight": "blocks.16.norm2.weight",
            "control_model.input_blocks.7.0.out_layers.0.bias": "blocks.16.norm2.bias",
            "control_model.input_blocks.7.0.out_layers.3.weight": "blocks.16.conv2.weight",
            "control_model.input_blocks.7.0.out_layers.3.bias": "blocks.16.conv2.bias",
            "control_model.input_blocks.7.0.skip_connection.weight": "blocks.16.conv_shortcut.weight",
            "control_model.input_blocks.7.0.skip_connection.bias": "blocks.16.conv_shortcut.bias",
            "control_model.input_blocks.7.1.norm.weight": "blocks.17.norm.weight",
            "control_model.input_blocks.7.1.norm.bias": "blocks.17.norm.bias",
            "control_model.input_blocks.7.1.proj_in.weight": "blocks.17.proj_in.weight",
            "control_model.input_blocks.7.1.proj_in.bias": "blocks.17.proj_in.bias",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight": "blocks.17.transformer_blocks.0.attn1.to_q.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight": "blocks.17.transformer_blocks.0.attn1.to_k.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight": "blocks.17.transformer_blocks.0.attn1.to_v.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.17.transformer_blocks.0.attn1.to_out.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.17.transformer_blocks.0.attn1.to_out.bias",
            "control_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.17.transformer_blocks.0.act_fn.proj.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.17.transformer_blocks.0.act_fn.proj.bias",
            "control_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight": "blocks.17.transformer_blocks.0.ff.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias": "blocks.17.transformer_blocks.0.ff.bias",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight": "blocks.17.transformer_blocks.0.attn2.to_q.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "blocks.17.transformer_blocks.0.attn2.to_k.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "blocks.17.transformer_blocks.0.attn2.to_v.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.17.transformer_blocks.0.attn2.to_out.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.17.transformer_blocks.0.attn2.to_out.bias",
            "control_model.input_blocks.7.1.transformer_blocks.0.norm1.weight": "blocks.17.transformer_blocks.0.norm1.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.norm1.bias": "blocks.17.transformer_blocks.0.norm1.bias",
            "control_model.input_blocks.7.1.transformer_blocks.0.norm2.weight": "blocks.17.transformer_blocks.0.norm2.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.norm2.bias": "blocks.17.transformer_blocks.0.norm2.bias",
            "control_model.input_blocks.7.1.transformer_blocks.0.norm3.weight": "blocks.17.transformer_blocks.0.norm3.weight",
            "control_model.input_blocks.7.1.transformer_blocks.0.norm3.bias": "blocks.17.transformer_blocks.0.norm3.bias",
            "control_model.input_blocks.7.1.proj_out.weight": "blocks.17.proj_out.weight",
            "control_model.input_blocks.7.1.proj_out.bias": "blocks.17.proj_out.bias",
            "control_model.input_blocks.8.0.in_layers.0.weight": "blocks.19.norm1.weight",
            "control_model.input_blocks.8.0.in_layers.0.bias": "blocks.19.norm1.bias",
            "control_model.input_blocks.8.0.in_layers.2.weight": "blocks.19.conv1.weight",
            "control_model.input_blocks.8.0.in_layers.2.bias": "blocks.19.conv1.bias",
            "control_model.input_blocks.8.0.emb_layers.1.weight": "blocks.19.time_emb_proj.weight",
            "control_model.input_blocks.8.0.emb_layers.1.bias": "blocks.19.time_emb_proj.bias",
            "control_model.input_blocks.8.0.out_layers.0.weight": "blocks.19.norm2.weight",
            "control_model.input_blocks.8.0.out_layers.0.bias": "blocks.19.norm2.bias",
            "control_model.input_blocks.8.0.out_layers.3.weight": "blocks.19.conv2.weight",
            "control_model.input_blocks.8.0.out_layers.3.bias": "blocks.19.conv2.bias",
            "control_model.input_blocks.8.1.norm.weight": "blocks.20.norm.weight",
            "control_model.input_blocks.8.1.norm.bias": "blocks.20.norm.bias",
            "control_model.input_blocks.8.1.proj_in.weight": "blocks.20.proj_in.weight",
            "control_model.input_blocks.8.1.proj_in.bias": "blocks.20.proj_in.bias",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight": "blocks.20.transformer_blocks.0.attn1.to_q.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight": "blocks.20.transformer_blocks.0.attn1.to_k.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight": "blocks.20.transformer_blocks.0.attn1.to_v.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.20.transformer_blocks.0.attn1.to_out.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.20.transformer_blocks.0.attn1.to_out.bias",
            "control_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.20.transformer_blocks.0.act_fn.proj.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.20.transformer_blocks.0.act_fn.proj.bias",
            "control_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight": "blocks.20.transformer_blocks.0.ff.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias": "blocks.20.transformer_blocks.0.ff.bias",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight": "blocks.20.transformer_blocks.0.attn2.to_q.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "blocks.20.transformer_blocks.0.attn2.to_k.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "blocks.20.transformer_blocks.0.attn2.to_v.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.20.transformer_blocks.0.attn2.to_out.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.20.transformer_blocks.0.attn2.to_out.bias",
            "control_model.input_blocks.8.1.transformer_blocks.0.norm1.weight": "blocks.20.transformer_blocks.0.norm1.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.norm1.bias": "blocks.20.transformer_blocks.0.norm1.bias",
            "control_model.input_blocks.8.1.transformer_blocks.0.norm2.weight": "blocks.20.transformer_blocks.0.norm2.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.norm2.bias": "blocks.20.transformer_blocks.0.norm2.bias",
            "control_model.input_blocks.8.1.transformer_blocks.0.norm3.weight": "blocks.20.transformer_blocks.0.norm3.weight",
            "control_model.input_blocks.8.1.transformer_blocks.0.norm3.bias": "blocks.20.transformer_blocks.0.norm3.bias",
            "control_model.input_blocks.8.1.proj_out.weight": "blocks.20.proj_out.weight",
            "control_model.input_blocks.8.1.proj_out.bias": "blocks.20.proj_out.bias",
            "control_model.input_blocks.9.0.op.weight": "blocks.22.conv.weight",
            "control_model.input_blocks.9.0.op.bias": "blocks.22.conv.bias",
            "control_model.input_blocks.10.0.in_layers.0.weight": "blocks.24.norm1.weight",
            "control_model.input_blocks.10.0.in_layers.0.bias": "blocks.24.norm1.bias",
            "control_model.input_blocks.10.0.in_layers.2.weight": "blocks.24.conv1.weight",
            "control_model.input_blocks.10.0.in_layers.2.bias": "blocks.24.conv1.bias",
            "control_model.input_blocks.10.0.emb_layers.1.weight": "blocks.24.time_emb_proj.weight",
            "control_model.input_blocks.10.0.emb_layers.1.bias": "blocks.24.time_emb_proj.bias",
            "control_model.input_blocks.10.0.out_layers.0.weight": "blocks.24.norm2.weight",
            "control_model.input_blocks.10.0.out_layers.0.bias": "blocks.24.norm2.bias",
            "control_model.input_blocks.10.0.out_layers.3.weight": "blocks.24.conv2.weight",
            "control_model.input_blocks.10.0.out_layers.3.bias": "blocks.24.conv2.bias",
            "control_model.input_blocks.11.0.in_layers.0.weight": "blocks.26.norm1.weight",
            "control_model.input_blocks.11.0.in_layers.0.bias": "blocks.26.norm1.bias",
            "control_model.input_blocks.11.0.in_layers.2.weight": "blocks.26.conv1.weight",
            "control_model.input_blocks.11.0.in_layers.2.bias": "blocks.26.conv1.bias",
            "control_model.input_blocks.11.0.emb_layers.1.weight": "blocks.26.time_emb_proj.weight",
            "control_model.input_blocks.11.0.emb_layers.1.bias": "blocks.26.time_emb_proj.bias",
            "control_model.input_blocks.11.0.out_layers.0.weight": "blocks.26.norm2.weight",
            "control_model.input_blocks.11.0.out_layers.0.bias": "blocks.26.norm2.bias",
            "control_model.input_blocks.11.0.out_layers.3.weight": "blocks.26.conv2.weight",
            "control_model.input_blocks.11.0.out_layers.3.bias": "blocks.26.conv2.bias",
            "control_model.zero_convs.0.0.weight": "controlnet_blocks.0.weight",
            "control_model.zero_convs.0.0.bias": "controlnet_blocks.0.bias",
            "control_model.zero_convs.1.0.weight": "controlnet_blocks.1.weight",
            "control_model.zero_convs.1.0.bias": "controlnet_blocks.0.bias",
            "control_model.zero_convs.2.0.weight": "controlnet_blocks.2.weight",
            "control_model.zero_convs.2.0.bias": "controlnet_blocks.0.bias",
            "control_model.zero_convs.3.0.weight": "controlnet_blocks.3.weight",
            "control_model.zero_convs.3.0.bias": "controlnet_blocks.0.bias",
            "control_model.zero_convs.4.0.weight": "controlnet_blocks.4.weight",
            "control_model.zero_convs.4.0.bias": "controlnet_blocks.4.bias",
            "control_model.zero_convs.5.0.weight": "controlnet_blocks.5.weight",
            "control_model.zero_convs.5.0.bias": "controlnet_blocks.4.bias",
            "control_model.zero_convs.6.0.weight": "controlnet_blocks.6.weight",
            "control_model.zero_convs.6.0.bias": "controlnet_blocks.4.bias",
            "control_model.zero_convs.7.0.weight": "controlnet_blocks.7.weight",
            "control_model.zero_convs.7.0.bias": "controlnet_blocks.7.bias",
            "control_model.zero_convs.8.0.weight": "controlnet_blocks.8.weight",
            "control_model.zero_convs.8.0.bias": "controlnet_blocks.7.bias",
            "control_model.zero_convs.9.0.weight": "controlnet_blocks.9.weight",
            "control_model.zero_convs.9.0.bias": "controlnet_blocks.7.bias",
            "control_model.zero_convs.10.0.weight": "controlnet_blocks.10.weight",
            "control_model.zero_convs.10.0.bias": "controlnet_blocks.7.bias",
            "control_model.zero_convs.11.0.weight": "controlnet_blocks.11.weight",
            "control_model.zero_convs.11.0.bias": "controlnet_blocks.7.bias",
            "control_model.input_hint_block.0.weight": "controlnet_conv_in.blocks.0.weight",
            "control_model.input_hint_block.0.bias": "controlnet_conv_in.blocks.0.bias",
            "control_model.input_hint_block.2.weight": "controlnet_conv_in.blocks.2.weight",
            "control_model.input_hint_block.2.bias": "controlnet_conv_in.blocks.2.bias",
            "control_model.input_hint_block.4.weight": "controlnet_conv_in.blocks.4.weight",
            "control_model.input_hint_block.4.bias": "controlnet_conv_in.blocks.4.bias",
            "control_model.input_hint_block.6.weight": "controlnet_conv_in.blocks.6.weight",
            "control_model.input_hint_block.6.bias": "controlnet_conv_in.blocks.6.bias",
            "control_model.input_hint_block.8.weight": "controlnet_conv_in.blocks.8.weight",
            "control_model.input_hint_block.8.bias": "controlnet_conv_in.blocks.8.bias",
            "control_model.input_hint_block.10.weight": "controlnet_conv_in.blocks.10.weight",
            "control_model.input_hint_block.10.bias": "controlnet_conv_in.blocks.10.bias",
            "control_model.input_hint_block.12.weight": "controlnet_conv_in.blocks.12.weight",
            "control_model.input_hint_block.12.bias": "controlnet_conv_in.blocks.12.bias",
            "control_model.input_hint_block.14.weight": "controlnet_conv_in.blocks.14.weight",
            "control_model.input_hint_block.14.bias": "controlnet_conv_in.blocks.14.bias",
            "control_model.middle_block.0.in_layers.0.weight": "blocks.28.norm1.weight",
            "control_model.middle_block.0.in_layers.0.bias": "blocks.28.norm1.bias",
            "control_model.middle_block.0.in_layers.2.weight": "blocks.28.conv1.weight",
            "control_model.middle_block.0.in_layers.2.bias": "blocks.28.conv1.bias",
            "control_model.middle_block.0.emb_layers.1.weight": "blocks.28.time_emb_proj.weight",
            "control_model.middle_block.0.emb_layers.1.bias": "blocks.28.time_emb_proj.bias",
            "control_model.middle_block.0.out_layers.0.weight": "blocks.28.norm2.weight",
            "control_model.middle_block.0.out_layers.0.bias": "blocks.28.norm2.bias",
            "control_model.middle_block.0.out_layers.3.weight": "blocks.28.conv2.weight",
            "control_model.middle_block.0.out_layers.3.bias": "blocks.28.conv2.bias",
            "control_model.middle_block.1.norm.weight": "blocks.29.norm.weight",
            "control_model.middle_block.1.norm.bias": "blocks.29.norm.bias",
            "control_model.middle_block.1.proj_in.weight": "blocks.29.proj_in.weight",
            "control_model.middle_block.1.proj_in.bias": "blocks.29.proj_in.bias",
            "control_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight": "blocks.29.transformer_blocks.0.attn1.to_q.weight",
            "control_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight": "blocks.29.transformer_blocks.0.attn1.to_k.weight",
            "control_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight": "blocks.29.transformer_blocks.0.attn1.to_v.weight",
            "control_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.29.transformer_blocks.0.attn1.to_out.weight",
            "control_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.29.transformer_blocks.0.attn1.to_out.bias",
            "control_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.29.transformer_blocks.0.act_fn.proj.weight",
            "control_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.29.transformer_blocks.0.act_fn.proj.bias",
            "control_model.middle_block.1.transformer_blocks.0.ff.net.2.weight": "blocks.29.transformer_blocks.0.ff.weight",
            "control_model.middle_block.1.transformer_blocks.0.ff.net.2.bias": "blocks.29.transformer_blocks.0.ff.bias",
            "control_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight": "blocks.29.transformer_blocks.0.attn2.to_q.weight",
            "control_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight": "blocks.29.transformer_blocks.0.attn2.to_k.weight",
            "control_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight": "blocks.29.transformer_blocks.0.attn2.to_v.weight",
            "control_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.29.transformer_blocks.0.attn2.to_out.weight",
            "control_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.29.transformer_blocks.0.attn2.to_out.bias",
            "control_model.middle_block.1.transformer_blocks.0.norm1.weight": "blocks.29.transformer_blocks.0.norm1.weight",
            "control_model.middle_block.1.transformer_blocks.0.norm1.bias": "blocks.29.transformer_blocks.0.norm1.bias",
            "control_model.middle_block.1.transformer_blocks.0.norm2.weight": "blocks.29.transformer_blocks.0.norm2.weight",
            "control_model.middle_block.1.transformer_blocks.0.norm2.bias": "blocks.29.transformer_blocks.0.norm2.bias",
            "control_model.middle_block.1.transformer_blocks.0.norm3.weight": "blocks.29.transformer_blocks.0.norm3.weight",
            "control_model.middle_block.1.transformer_blocks.0.norm3.bias": "blocks.29.transformer_blocks.0.norm3.bias",
            "control_model.middle_block.1.proj_out.weight": "blocks.29.proj_out.weight",
            "control_model.middle_block.1.proj_out.bias": "blocks.29.proj_out.bias",
            "control_model.middle_block.2.in_layers.0.weight": "blocks.30.norm1.weight",
            "control_model.middle_block.2.in_layers.0.bias": "blocks.30.norm1.bias",
            "control_model.middle_block.2.in_layers.2.weight": "blocks.30.conv1.weight",
            "control_model.middle_block.2.in_layers.2.bias": "blocks.30.conv1.bias",
            "control_model.middle_block.2.emb_layers.1.weight": "blocks.30.time_emb_proj.weight",
            "control_model.middle_block.2.emb_layers.1.bias": "blocks.30.time_emb_proj.bias",
            "control_model.middle_block.2.out_layers.0.weight": "blocks.30.norm2.weight",
            "control_model.middle_block.2.out_layers.0.bias": "blocks.30.norm2.bias",
            "control_model.middle_block.2.out_layers.3.weight": "blocks.30.conv2.weight",
            "control_model.middle_block.2.out_layers.3.bias": "blocks.30.conv2.bias",
            "control_model.middle_block_out.0.weight": "controlnet_blocks.12.weight",
            "control_model.middle_block_out.0.bias": "controlnet_blocks.7.bias",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if ".proj_in." in name or ".proj_out." in name:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "mid_block.resnets.1.time_emb_proj.weight" in state_dict:
            return self._from_diffusers(state_dict)
        else:
            return self._from_civitai(state_dict)


class SDControlNet(PreTrainedModel):
    converter = SDControlNetStateDictConverter()

    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.time_embedding = TimestepEmbeddings(dim_in=320, dim_out=1280, device=device, dtype=dtype)
        self.conv_in = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1, device=device, dtype=dtype)

        self.controlnet_conv_in = ControlNetConditioningLayer(
            channels=(3, 16, 32, 96, 256, 320), device=device, dtype=dtype
        )

        self.blocks = torch.nn.ModuleList(
            [
                # CrossAttnDownBlock2D
                ResnetBlock(320, 320, 1280, device=device, dtype=dtype),
                AttentionBlock(8, 40, 320, 1, 768, device=device, dtype=dtype),
                PushBlock(),
                ResnetBlock(320, 320, 1280, device=device, dtype=dtype),
                AttentionBlock(8, 40, 320, 1, 768, device=device, dtype=dtype),
                PushBlock(),
                DownSampler(320, device=device, dtype=dtype),
                PushBlock(),
                # CrossAttnDownBlock2D
                ResnetBlock(320, 640, 1280, device=device, dtype=dtype),
                AttentionBlock(8, 80, 640, 1, 768, device=device, dtype=dtype),
                PushBlock(),
                ResnetBlock(640, 640, 1280, device=device, dtype=dtype),
                AttentionBlock(8, 80, 640, 1, 768, device=device, dtype=dtype),
                PushBlock(),
                DownSampler(640, device=device, dtype=dtype),
                PushBlock(),
                # CrossAttnDownBlock2D
                ResnetBlock(640, 1280, 1280, device=device, dtype=dtype),
                AttentionBlock(8, 160, 1280, 1, 768, device=device, dtype=dtype),
                PushBlock(),
                ResnetBlock(1280, 1280, 1280, device=device, dtype=dtype),
                AttentionBlock(8, 160, 1280, 1, 768, device=device, dtype=dtype),
                PushBlock(),
                DownSampler(1280, device=device, dtype=dtype),
                PushBlock(),
                # DownBlock2D
                ResnetBlock(1280, 1280, 1280, device=device, dtype=dtype),
                PushBlock(),
                ResnetBlock(1280, 1280, 1280, device=device, dtype=dtype),
                PushBlock(),
                # UNetMidBlock2DCrossAttn
                ResnetBlock(1280, 1280, 1280, device=device, dtype=dtype),
                AttentionBlock(8, 160, 1280, 1, 768, device=device, dtype=dtype),
                ResnetBlock(1280, 1280, 1280, device=device, dtype=dtype),
                PushBlock(),
            ]
        )

        self.controlnet_blocks = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(320, 320, kernel_size=(1, 1), device=device, dtype=dtype),
                torch.nn.Conv2d(320, 320, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(320, 320, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(320, 320, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(640, 640, kernel_size=(1, 1), device=device, dtype=dtype),
                torch.nn.Conv2d(640, 640, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(640, 640, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1), device=device, dtype=dtype),
                torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
                torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1), bias=False, device=device, dtype=dtype),
            ]
        )

    def forward(self, sample, timestep, encoder_hidden_states, conditioning, **kwargs):
        # 1. time
        time_emb = self.time_embedding(timestep, dtype=sample.dtype)

        # 2. pre-process
        hidden_states = self.conv_in(sample) + self.controlnet_conv_in(conditioning)
        text_emb = encoder_hidden_states
        res_stack = [hidden_states]

        # 3. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, _, _, _ = block(hidden_states, time_emb, text_emb, res_stack)

        # 4. ControlNet blocks
        controlnet_res_stack = [block(res) for block, res in zip(self.controlnet_blocks, res_stack)]

        return controlnet_res_stack

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
    ):
        model = cls(device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
