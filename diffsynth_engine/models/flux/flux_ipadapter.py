import torch
from einops import rearrange
from torch import nn
from PIL import Image
from typing import Dict, List
from functools import partial
from diffsynth_engine.models.text_encoder.siglip import SiglipImageEncoder
from diffsynth_engine.models.basic.transformer_helper import RMSNorm
from diffsynth_engine.models.basic.attention import attention
from diffsynth_engine.models.base import PreTrainedModel
from diffsynth_engine.utils.download import fetch_model


class FluxIPAdapterAttention(nn.Module):
    def __init__(
        self,
        image_emb_dim: int = 4096,
        dim: int = 3072,
        head_num: int = 24,
        scale: float = 1.0,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.norm_k = RMSNorm(dim, eps=1e-5, elementwise_affine=False, device=device, dtype=dtype)
        self.to_k_ip = nn.Linear(image_emb_dim, dim, device=device, dtype=dtype, bias=False)
        self.to_v_ip = nn.Linear(image_emb_dim, dim, device=device, dtype=dtype, bias=False)
        self.head_num = head_num
        self.scale = scale

    def forward(self, query: torch.Tensor, image_emb: torch.Tensor, attn_kwargs=None):
        key = rearrange(self.norm_k(self.to_k_ip(image_emb)), "b s (h d) -> b s h d", h=self.head_num)
        value = rearrange(self.to_v_ip(image_emb), "b s (h d) -> b s h d", h=self.head_num)

        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        attn_out = attention(query, key, value, **attn_kwargs)
        return self.scale * rearrange(attn_out, "b s h d -> b s (h d)")

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype, **kwargs):
        model = cls(device="meta", dtype=dtype, **kwargs)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model


class FluxIPAdapterMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 1152,
        output_dim: int = 4096,
        num_tokens: int = 128,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim * 2, device=device, dtype=dtype),
            torch.nn.GELU(),
            torch.nn.Linear(input_dim * 2, output_dim * num_tokens, device=device, dtype=dtype),
        )
        self.norm = torch.nn.LayerNorm(output_dim, device=device, dtype=dtype)
        self.output_dim = output_dim

    def forward(self, image_emb):
        x = self.proj(image_emb)
        x = x.reshape(-1, self.num_tokens, self.output_dim)  # b s d
        x = self.norm(x)
        return x

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype, **kwargs):
        model = cls(device="meta", dtype=dtype, **kwargs)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model


class FluxIPAdapter(PreTrainedModel):
    def __init__(
        self,
        image_encoder: SiglipImageEncoder,
        image_proj: FluxIPAdapterMLP,
        attentions: List[FluxIPAdapterAttention],
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_proj = image_proj
        self.attentions = attentions
        self.double_block_num = 19
        self.single_block_num = 38

    def set_scale(self, scale: float):
        for attn in self.attentions:
            attn.scale = scale

    def inject(self, dit):
        def double_attention_callback(
            self, attn_out_a, attn_out_b, x_a, x_b, q_a, q_b, k_a, k_b, v_a, v_b, rope_emb, image_emb
        ):
            attn_out_a = attn_out_a + self.ip_attn(q_a, image_emb)
            return attn_out_a, attn_out_b

        for i in range(self.double_block_num):
            dit.blocks[i].attn.ip_attn = self.attentions[i]
            dit.blocks[i].attn.attention_callback = partial(double_attention_callback, self=dit.blocks[i].attn)

        def single_attention_callback(self, attn_out, x, q, k, v, rope_emb, image_emb):
            attn_out = attn_out + self.ip_attn(q, image_emb)
            return attn_out

        for i in range(self.single_block_num):
            dit.single_blocks[i].attn.ip_attn = self.attentions[i + self.double_block_num]
            dit.single_blocks[i].attn.attention_callback = partial(
                single_attention_callback, self=dit.single_blocks[i].attn
            )

    def remove(self, dit):
        def double_attention_callback(
            self, attn_out_a, attn_out_b, x_a, x_b, q_a, q_b, k_a, k_b, v_a, v_b, rope_emb, image_emb
        ):
            return attn_out_a, attn_out_b

        for i in range(self.double_block_num):
            dit.blocks[i].attn.ip_attn = None
            dit.blocks[i].attn.attention_callback = partial(double_attention_callback, self=dit.blocks[i].attn)

        def single_attention_callback(self, attn_out, x, q, k, v, rope_emb, image_emb):
            return attn_out

        for i in range(self.single_block_num):
            dit.single_blocks[i].attn.ip_attn = None
            dit.single_blocks[i].attn.attention_callback = partial(
                single_attention_callback, self=dit.single_blocks[i].attn
            )

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        image_emb = self.image_encoder(image)
        return self.image_proj(image_emb)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        double_block_num=19,
        single_block_num=38,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        model_path = fetch_model("muse/google-siglip-so400m-patch14-384", path="model.safetensors")
        image_encoder = SiglipImageEncoder.from_pretrained(model_path, device=device, dtype=dtype)

        image_proj_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("image_proj."):
                image_proj_state_dict[k.replace("image_proj.", "")] = v
        image_proj = FluxIPAdapterMLP.from_state_dict(image_proj_state_dict, device=device, dtype=dtype)

        attentions = []
        for i in range(double_block_num + single_block_num):
            attn_state_dict = {
                "to_k_ip.weight": state_dict[f"attentions.{i}.to_k_ip.weight"],
                "to_v_ip.weight": state_dict[f"attentions.{i}.to_v_ip.weight"],
            }
            attentions.append(FluxIPAdapterAttention.from_state_dict(attn_state_dict, device=device, dtype=dtype))
        model = cls(image_encoder, image_proj, attentions, device=device, dtype=dtype)
        return model
