# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan_animate.py

# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm

from diffsynth_engine.distributed.utils import sequence_parallel_shard, sequence_parallel_unshard
from diffsynth_engine.forward_context import get_forward_context
from diffsynth_engine.layers.attention import USPAttention
from diffsynth_engine.models.base import DiffusionModel
from diffsynth_engine.models.wan.transformer_wan import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES = {
    "4": 512,
    "8": 512,
    "16": 512,
    "32": 512,
    "64": 256,
    "128": 128,
    "256": 64,
    "512": 32,
    "1024": 16,
}


class FusedLeakyReLU(nn.Module):
    """Fused LeakyRelu with scale factor and channel-wise bias."""

    def __init__(self, negative_slope: float = 0.2, scale: float = 2**0.5, bias_channels: Optional[int] = None):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.channels = bias_channels

        if self.channels is not None:
            self.bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self.bias is not None:
            expanded_shape = [1] * hidden_states.ndim
            expanded_shape[channel_dim] = self.bias.shape[0]
            bias = self.bias.reshape(*expanded_shape)
            hidden_states = hidden_states + bias
        return F.leaky_relu(hidden_states, self.negative_slope) * self.scale


class MotionConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        blur_kernel: Optional[Tuple[int, ...]] = None,
        blur_upsample_factor: int = 1,
        use_activation: bool = True,
    ):
        super().__init__()
        self.use_activation = use_activation
        self.in_channels = in_channels

        # Handle blurring (applying a FIR filter with the given kernel) if available
        self.blur = False
        if blur_kernel is not None:
            padding_amount = (len(blur_kernel) - stride) + (kernel_size - 1)
            self.blur_padding = ((padding_amount + 1) // 2, padding_amount // 2)

            kernel = torch.tensor(blur_kernel)
            if kernel.ndim == 1:
                kernel = kernel[None, :] * kernel[:, None]
            kernel = kernel / kernel.sum()
            if blur_upsample_factor > 1:
                kernel = kernel * (blur_upsample_factor**2)
            self.register_buffer("blur_kernel", kernel, persistent=False)
            self.blur = True

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias and not self.use_activation:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        if self.use_activation:
            self.act_fn = FusedLeakyReLU(bias_channels=out_channels)
        else:
            self.act_fn = None

    def forward(self, hidden_states: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self.blur:
            expanded_kernel = self.blur_kernel[None, None, :, :].expand(self.in_channels, 1, -1, -1)
            hidden_states = hidden_states.to(expanded_kernel.dtype)
            hidden_states = F.conv2d(hidden_states, expanded_kernel, padding=self.blur_padding, groups=self.in_channels)

        hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = F.conv2d(
            hidden_states, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding
        )

        if self.use_activation:
            hidden_states = self.act_fn(hidden_states, channel_dim=channel_dim)
        return hidden_states


class MotionLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        use_activation: bool = False,
    ):
        super().__init__()
        self.use_activation = use_activation

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = 1 / math.sqrt(in_dim)

        if bias and not self.use_activation:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        if self.use_activation:
            self.act_fn = FusedLeakyReLU(bias_channels=out_dim)
        else:
            self.act_fn = None

    def forward(self, input_tensor: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        output = F.linear(input_tensor, self.weight * self.scale, bias=self.bias)
        if self.use_activation:
            output = self.act_fn(output, channel_dim=channel_dim)
        return output


class MotionEncoderResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        kernel_size_skip: int = 1,
        blur_kernel: Tuple[int, ...] = (1, 3, 3, 1),
        downsample_factor: int = 2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.conv1 = MotionConv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            use_activation=True,
        )
        self.conv2 = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=0,
            blur_kernel=blur_kernel,
            use_activation=True,
        )
        self.conv_skip = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_skip,
            stride=self.downsample_factor,
            padding=0,
            bias=False,
            blur_kernel=blur_kernel,
            use_activation=False,
        )

    def forward(self, hidden_states: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        residual = self.conv1(hidden_states, channel_dim)
        residual = self.conv2(residual, channel_dim)
        skip = self.conv_skip(hidden_states, channel_dim)
        return (residual + skip) / math.sqrt(2)


class WanAnimateMotionEncoder(nn.Module):
    def __init__(
        self,
        size: int = 512,
        style_dim: int = 512,
        motion_dim: int = 20,
        out_dim: int = 512,
        motion_blocks: int = 5,
        channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.size = size

        if channels is None:
            channels = WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

        self.conv_in = MotionConv2d(3, channels[str(size)], 1, use_activation=True)

        self.res_blocks = nn.ModuleList()
        in_channels = channels[str(size)]
        log_size = int(math.log(size, 2))
        for i in range(log_size, 2, -1):
            out_channels = channels[str(2 ** (i - 1))]
            self.res_blocks.append(MotionEncoderResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.conv_out = MotionConv2d(in_channels, style_dim, 4, padding=0, bias=False, use_activation=False)

        linears = [MotionLinear(style_dim, style_dim) for _ in range(motion_blocks - 1)]
        linears.append(MotionLinear(style_dim, motion_dim))
        self.motion_network = nn.ModuleList(linears)

        self.motion_synthesis_weight = nn.Parameter(torch.randn(out_dim, motion_dim))

    def forward(self, face_image: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if (face_image.shape[-2] != self.size) or (face_image.shape[-1] != self.size):
            raise ValueError(
                f"Face pixel values has resolution ({face_image.shape[-1]}, {face_image.shape[-2]}) but is expected"
                f" to have resolution ({self.size}, {self.size})"
            )

        face_image = self.conv_in(face_image, channel_dim)
        for block in self.res_blocks:
            face_image = block(face_image, channel_dim)
        face_image = self.conv_out(face_image, channel_dim)
        motion_feat = face_image.squeeze(-1).squeeze(-1)

        for linear_layer in self.motion_network:
            motion_feat = linear_layer(motion_feat, channel_dim=channel_dim)

        # Motion synthesis via Linear Motion Decomposition
        weight = self.motion_synthesis_weight + 1e-8
        original_motion_dtype = motion_feat.dtype
        motion_feat = motion_feat.to(torch.float32)
        weight = weight.to(torch.float32)

        orthogonal_basis = torch.linalg.qr(weight)[0].to(device=motion_feat.device)

        motion_feat_diag = torch.diag_embed(motion_feat)
        motion_decomposition = torch.matmul(motion_feat_diag, orthogonal_basis.T)
        motion_vec = torch.sum(motion_decomposition, dim=1)

        motion_vec = motion_vec.to(dtype=original_motion_dtype)
        return motion_vec


class WanAnimateFaceEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        kernel_size: int = 3,
        eps: float = 1e-6,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.time_causal_padding = (kernel_size - 1, 0)
        self.pad_mode = pad_mode

        self.act = nn.SiLU()

        self.conv1_local = nn.Conv1d(in_dim, hidden_dim * num_heads, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)

        self.norm1 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)

        self.out_proj = nn.Linear(hidden_dim, out_dim)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, out_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # Reshape to channels-first to apply causal Conv1d over frame dim
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        hidden_states = self.conv1_local(hidden_states)
        hidden_states = hidden_states.unflatten(1, (self.num_heads, -1)).flatten(0, 1)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.act(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        hidden_states = self.conv3(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.act(hidden_states)

        hidden_states = self.out_proj(hidden_states)
        # [B * N, T, C_out] --> [B, T, N, C_out]
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)

        padding = self.padding_tokens.repeat(batch_size, hidden_states.shape[1], 1, 1).to(device=hidden_states.device)
        hidden_states = torch.cat([hidden_states, padding], dim=-2)

        return hidden_states


class WanAnimateFaceBlockCrossAttention(nn.Module):
    """
    Temporally-aligned cross attention with the face motion signal in the Wan Animate Face Blocks.

    This is a simplified version that directly implements the attention logic using USPAttention,
    instead of the processor pattern used in diffusers.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-6,
        cross_attention_dim_head: Optional[int] = None,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        # Pre-Attention Norms for hidden_states (video latents) and encoder_hidden_states (motion vector)
        self.pre_norm_q = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.pre_norm_kv = nn.LayerNorm(dim, eps, elementwise_affine=False)

        # QKV and Output Projections
        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=True)

        # QK Norm (applied after reshape, so over dim_head rather than dim_head * heads)
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)

        # USPAttention for attention computation
        forward_context = get_forward_context()
        self.usp_attn = USPAttention(
            num_heads=heads,
            head_size=dim_head,
            attn_type=forward_context.attn_type,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # encoder_hidden_states corresponds to the motion vec
        # attention_mask corresponds to the motion mask (if any)
        hidden_states = self.pre_norm_q(hidden_states)
        encoder_hidden_states = self.pre_norm_kv(encoder_hidden_states)

        # B --> batch_size, T --> reduced inference segment len, N --> face_encoder_num_heads + 1, C --> dim
        batch_size, num_time_steps, num_tokens, channels = encoder_hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # [B, S, H * D] --> [B, S, H, D]
        query = query.unflatten(2, (self.heads, -1))
        # [B, T, N, H * D_kv] --> [B, T, N, H, D_kv]
        key = key.view(batch_size, num_time_steps, num_tokens, self.heads, -1)
        value = value.view(batch_size, num_time_steps, num_tokens, self.heads, -1)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape for temporally-aligned attention:
        # query: [B, S, H, D] --> [B * T, S / T, H, D]
        query = query.unflatten(1, (num_time_steps, -1)).flatten(0, 1)
        # key/value: [B, T, N, H, D_kv] --> [B * T, N, H, D_kv]
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        hidden_states = self.usp_attn(query, key, value)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = hidden_states.unflatten(0, (batch_size, num_time_steps)).flatten(1, 2)

        hidden_states = self.to_out(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.flatten(start_dim=1)
            hidden_states = hidden_states * attention_mask

        return hidden_states


class WanAnimateTransformer3DModel(DiffusionModel):
    """
    A Transformer model for video-like data used in the WanAnimate model,
    supporting character animation and replacement.
    """

    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "motion_synthesis_weight",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: Optional[int] = 36,
        latent_channels: Optional[int] = 16,
        out_channels: Optional[int] = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = 1280,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        motion_encoder_channel_sizes: Optional[Dict[str, int]] = None,
        motion_encoder_size: int = 512,
        motion_style_dim: int = 512,
        motion_dim: int = 20,
        motion_encoder_dim: int = 512,
        face_encoder_hidden_dim: int = 1024,
        face_encoder_num_heads: int = 4,
        inject_face_latents_blocks: int = 5,
        motion_encoder_batch_size: int = 8,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        # Allow either only in_channels or only latent_channels to be set for convenience
        if in_channels is None and latent_channels is not None:
            in_channels = 2 * latent_channels + 4
        elif in_channels is not None and latent_channels is None:
            latent_channels = (in_channels - 4) // 2
        elif in_channels is not None and latent_channels is not None:
            assert in_channels == 2 * latent_channels + 4, "in_channels should be 2 * latent_channels + 4"
        else:
            raise ValueError("At least one of `in_channels` and `latent_channels` must be supplied.")
        out_channels = out_channels or latent_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.pose_patch_embedding = nn.Conv3d(latent_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Motion encoder
        self.motion_encoder = WanAnimateMotionEncoder(
            size=motion_encoder_size,
            style_dim=motion_style_dim,
            motion_dim=motion_dim,
            out_dim=motion_encoder_dim,
            channels=motion_encoder_channel_sizes,
        )

        # 4. Face encoder
        self.face_encoder = WanAnimateFaceEncoder(
            in_dim=motion_encoder_dim,
            out_dim=inner_dim,
            hidden_dim=face_encoder_hidden_dim,
            num_heads=face_encoder_num_heads,
        )

        # 5. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_attention_heads,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Face adapter (applied after every inject_face_latents_blocks-th block)
        self.face_adapter = nn.ModuleList(
            [
                WanAnimateFaceBlockCrossAttention(
                    dim=inner_dim,
                    heads=num_attention_heads,
                    dim_head=inner_dim // num_attention_heads,
                    eps=eps,
                    cross_attention_dim_head=inner_dim // num_attention_heads,
                )
                for _ in range(num_layers // inject_face_latents_blocks)
            ]
        )

        # 7. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        pose_hidden_states: Optional[torch.Tensor] = None,
        face_pixel_values: Optional[torch.Tensor] = None,
        motion_encode_batch_size: Optional[int] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass of Wan2.2-Animate transformer model.

        Args:
            hidden_states: Input noisy video latents of shape (B, 2C + 4, T + 1, H, W).
            timestep: The current timestep in the denoising loop.
            encoder_hidden_states: Text embeddings from the text encoder.
            encoder_hidden_states_image: CLIP visual features of the reference (character) image.
            pose_hidden_states: Pose video latents of shape (B, C, T, H, W).
            face_pixel_values: Face video in pixel space of shape (B, C', S, H', W').
            motion_encode_batch_size: Batch size for batched encoding of the face video via the motion encoder.
            return_dict: Whether to return the output as a dict or tuple.
        """
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. Rotary position embedding
        rotary_emb = self.rope(hidden_states)

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        pose_hidden_states = self.pose_patch_embedding(pose_hidden_states)
        # Add pose embeddings to hidden states (skip the first conditioning frame)
        hidden_states[:, :, 1:] = hidden_states[:, :, 1:] + pose_hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Save original sequence length for unshard
        original_seq_len = hidden_states.shape[1]

        # 3. Condition embeddings (time, text, image)
        # Wan Animate is based on Wan 2.1 and thus uses Wan 2.1's timestep logic
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=None
        )

        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Get motion features from the face video
        batch_size, face_channels, num_face_frames, face_height, face_width = face_pixel_values.shape
        # Rearrange from (B, C, T, H, W) to (B*T, C, H, W)
        face_pixel_values = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(-1, face_channels, face_height, face_width)

        # Batched motion encoder inference to trade off speed for memory
        motion_encode_batch_size = motion_encode_batch_size or self.config.motion_encoder_batch_size
        face_batches = torch.split(face_pixel_values, motion_encode_batch_size)
        motion_vec_batches = []
        for face_batch in face_batches:
            motion_vec_batch = self.motion_encoder(face_batch)
            motion_vec_batches.append(motion_vec_batch)
        motion_vec = torch.cat(motion_vec_batches)
        motion_vec = motion_vec.view(batch_size, num_face_frames, -1)

        # Get face features from the motion vector
        motion_vec = self.face_encoder(motion_vec)

        # Add padding at the beginning (prepend zeros for the conditioning frame)
        pad_face = torch.zeros_like(motion_vec[:, :1])
        motion_vec = torch.cat([pad_face, motion_vec], dim=1)

        # 5. Sequence parallel shard
        rotary_emb_cos, rotary_emb_sin = rotary_emb
        hidden_states, rotary_emb_cos, rotary_emb_sin = sequence_parallel_shard(
            [hidden_states, rotary_emb_cos, rotary_emb_sin],
            seq_dims=[1, 1, 1],
        )
        rotary_emb = (rotary_emb_cos, rotary_emb_sin)

        # 6. Transformer blocks with face adapter integration
        for block_idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

            # Face adapter integration: apply after every inject_face_latents_blocks-th block
            if block_idx % self.config.inject_face_latents_blocks == 0:
                face_adapter_block_idx = block_idx // self.config.inject_face_latents_blocks
                face_adapter_output = self.face_adapter[face_adapter_block_idx](hidden_states, motion_vec)
                face_adapter_output = face_adapter_output.to(device=hidden_states.device)
                hidden_states = face_adapter_output + hidden_states

        # 7. Sequence parallel unshard
        (hidden_states,) = sequence_parallel_unshard([hidden_states], seq_dims=[1], seq_lens=[original_seq_len])

        # 8. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
