import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Union, List
from PIL import Image
from diffsynth_engine.models.basic.attention import Attention
from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter


class SiglipVisionEmbeddings(nn.Module):
    def __init__(
        self, num_channels: int, num_positions: int, hidden_size: int, patch_size: int, device: str, dtype: torch.dtype
    ):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            device=device,
            dtype=dtype,
        )
        self.position_embedding = nn.Embedding(num_positions, hidden_size, device=device, dtype=dtype)
        self.position_ids = torch.arange(num_positions).expand((1, -1))

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        target_device = self.patch_embedding.weight.device
        self.position_ids = self.position_ids.to(target_device)
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, hidden_size, inner_dim, device, dtype):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner_dim, device=device, dtype=dtype)
        self.fc2 = nn.Linear(inner_dim, hidden_size, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, inner_dim: int, num_heads: int, eps: float, device: str, dtype: torch.dtype):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = Attention(
            q_dim=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            bias_q=True,
            bias_kv=True,
            bias_out=True,
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = SiglipMLP(hidden_size=hidden_size, inner_dim=inner_dim, device=device, dtype=dtype)

    def forward(self, x):
        x = self.self_attn(self.layer_norm1(x)) + x
        x = self.mlp(self.layer_norm2(x)) + x
        return x


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, hidden_size, inner_dim, num_heads, eps, device, dtype) -> None:
        super().__init__()

        self.probe = nn.Parameter(data=torch.randn(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, device=device, dtype=dtype
        )
        self.layernorm = nn.LayerNorm(normalized_shape=hidden_size, eps=eps, device=device, dtype=dtype)
        self.mlp = SiglipMLP(hidden_size=hidden_size, inner_dim=inner_dim, device=device, dtype=dtype)

    def forward(self, hidden_state) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1152,
        num_channels: int = 3,
        image_size: int = 384,
        patch_size: int = 14,
        layer_num: int = 27,
        inner_dim: int = 4304,
        num_heads: int = 16,
        eps: float = 1e-06,
        use_head: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(
            num_channels=num_channels,
            num_positions=(image_size // patch_size) ** 2,
            hidden_size=hidden_size,
            patch_size=patch_size,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(hidden_size, inner_dim, num_heads, eps, device, dtype) for _ in range(layer_num)]
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=eps, device=device, dtype=dtype)
        self.head = SiglipMultiheadAttentionPoolingHead(
            hidden_size, inner_dim=inner_dim, num_heads=num_heads, eps=eps, device=device, dtype=dtype
        )
        self.use_head = use_head

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        if self.use_head:
            x = self.head(x)
        return x


class SiglipImageEncoderConverter(StateDictConverter):
    def convert(self, state_dict: dict) -> dict:
        return state_dict


class SiglipImageEncoder(PreTrainedModel):
    converter = SiglipImageEncoderConverter()

    def __init__(self, use_head: bool = True, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.image_encoder = SiglipVisionTransformer(device=device, dtype=dtype, use_head=use_head)

    def image_preprocess(self, images: List[Image.Image]):
        images = [image.resize(size=(384, 384), resample=3) for image in images]
        rescaled_images = [np.array(image) / 255 for image in images]
        normalized_images = [(image - 0.5) / 0.5 for image in rescaled_images]
        image_tensor = torch.stack([torch.tensor(image) for image in normalized_images])
        param = next(self.parameters())
        image_tensor = image_tensor.to(param.device, param.dtype)
        return rearrange(image_tensor, "b h w c -> b c h w")

    @torch.no_grad()
    def forward(self, images: List[Image.Image] | Image.Image):
        if isinstance(images, Image.Image):
            images = [images]
        image_input = self.image_preprocess(images)
        result = self.image_encoder(image_input)
        return result

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike], device: str, dtype: torch.dtype, **kwargs):
        state_dict = load_file(str(pretrained_model_path))
        return cls.from_state_dict(state_dict, device=device, dtype=dtype, **kwargs)
