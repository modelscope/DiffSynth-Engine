from typing import Tuple, Dict

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffsynth_engine.models.base import PreTrainedModel
from diffsynth_engine.models.basic import attention as attention_ops


# ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇ Wav2Vec2ForCTC ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇
class Wav2Vec2Config:
    def __init__(self):
        self.conv_bias = True
        self.conv_dim = [512, 512, 512, 512, 512, 512, 512]
        self.conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        self.conv_stride = [5, 2, 2, 2, 2, 2, 2]
        self.hidden_size = 1024
        self.intermediate_size = 4096
        self.layer_norm_eps = 1e-05
        self.num_attention_heads = 16
        self.num_conv_pos_embedding_groups = 16
        self.num_conv_pos_embeddings = 128
        self.num_feat_extract_layers = 7
        self.num_hidden_layers = 24


class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config, layer_id=0, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
            device=device,
            dtype=dtype,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings: int):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config: Wav2Vec2Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
            device=device,
            dtype=dtype,
        )

        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config: Wav2Vec2Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return hidden_states


class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads).contiguous()
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads).contiguous()
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads).contiguous()
        attn_output = attention_ops.attention(q=q, k=k, v=v)
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)").contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config: Wav2Vec2Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            device=device,
            dtype=dtype,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype)
        self.feed_forward = Wav2Vec2FeedForward(config, device=device, dtype=dtype)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        return hidden_states


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config: Wav2Vec2Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config, device=device, dtype=dtype)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Wav2Vec2EncoderLayerStableLayerNorm(config, device=device, dtype=dtype)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        all_hidden_states = all_hidden_states + (hidden_states,)
        return all_hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(self, config: Wav2Vec2Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i, device=device, dtype=dtype)
                for i in range(config.num_feat_extract_layers)
            ]
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config: Wav2Vec2Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps, device=device, dtype=dtype)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        return hidden_states, norm_hidden_states


class Wav2Vec2StateDictConverter:
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("wav2vec2.") and "masked_spec_embed" not in k:
                new_state_dict[k[len("wav2vec2.") :]] = v
        return new_state_dict


class Wav2Vec2Model(PreTrainedModel):
    converter = Wav2Vec2StateDictConverter()

    def __init__(self, config: Wav2Vec2Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureEncoder(config, device=device, dtype=dtype)
        self.feature_projection = Wav2Vec2FeatureProjection(config, device=device, dtype=dtype)
        self.encoder = Wav2Vec2EncoderStableLayerNorm(config, device=device, dtype=dtype)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        extract_features = self.feature_extractor(input_values).transpose(1, 2)
        hidden_states, _ = self.feature_projection(extract_features)
        return self.encoder(hidden_states)


# ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆ Wav2Vec2ForCTC ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆


def get_sample_indices(original_fps: int, target_fps: int, total_frames: int, num_samples: int) -> np.ndarray:
    required_duration = num_samples / target_fps
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    time_points = np.linspace(0, required_duration, num_samples, endpoint=False)
    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def linear_interpolation(features: torch.Tensor, input_fps: int, output_fps: int) -> torch.Tensor:
    """
    features: shape=[1, T, 512]
    input_fps: fps for audio, f_a
    output_fps: fps for video, f_m
    output_len: video length
    """
    features = features.transpose(1, 2)  # [1, 512, T]
    seq_len = features.shape[2] / float(input_fps)  # T/f_a
    output_len = int(seq_len * output_fps)  # f_m*T/f_a
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )  # [1, 512, output_len]
    return output_features.transpose(1, 2)  # [1, output_len, 512]


def extract_audio_feat(
    audio_input: torch.Tensor, model: Wav2Vec2Model, dtype=torch.float32, device="cuda:0"
) -> torch.Tensor:
    video_rate = 30
    input_values = (audio_input - audio_input.mean(dim=1, keepdim=True)) / torch.sqrt(
        audio_input.var(dim=1, keepdim=True) + 1e-7
    )
    feat = torch.cat(model(input_values.to(device)))
    feat = linear_interpolation(feat, input_fps=50, output_fps=video_rate)
    return feat.to(dtype)  # Encoding for the motion


def get_audio_embed_bucket_fps(
    audio_embed: torch.Tensor, num_frames_per_batch: int, fps: int = 16
) -> Tuple[torch.Tensor, int]:
    video_rate = 30
    scale = video_rate / fps
    num_layers, num_audio_frames, audio_dim = audio_embed.shape
    max_num_batches = int(num_audio_frames / (num_frames_per_batch * scale)) + 1
    num_buckets = max_num_batches * num_frames_per_batch
    num_audio_padding = math.ceil(max_num_batches * num_frames_per_batch / fps * video_rate) - num_audio_frames
    batch_indices = get_sample_indices(
        original_fps=video_rate,
        target_fps=fps,
        total_frames=num_audio_frames + num_audio_padding,
        num_samples=num_buckets,
    )
    batch_audio_embed = []
    audio_sample_stride = int(video_rate / fps)
    for batch_idx in batch_indices:
        if batch_idx < num_audio_frames:
            chosen_idx = list(range(batch_idx, batch_idx + audio_sample_stride, audio_sample_stride))
            chosen_idx = [0 if c < 0 else c for c in chosen_idx]
            chosen_idx = [num_audio_frames - 1 if c >= num_audio_frames else c for c in chosen_idx]
            frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
        else:
            frame_audio_embed = torch.zeros([num_layers, audio_dim], device=audio_embed.device)
        batch_audio_embed.append(frame_audio_embed)
    batch_audio_embed = torch.stack(batch_audio_embed, dim=0)

    return batch_audio_embed, max_num_batches
