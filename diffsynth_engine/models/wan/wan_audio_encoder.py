from typing import Tuple, Optional, Dict

import librosa
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

    def forward(self, hidden_states: torch.Tensor):
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

    def forward(self, hidden_states: torch.Tensor):
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

        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)

    def forward(self, hidden_states: torch.Tensor):
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

    def forward(self, hidden_states):
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
        dtype: torch.dtype = torch.bfloat16
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

    def forward(self, hidden_states: torch.Tensor):
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
            [Wav2Vec2EncoderLayerStableLayerNorm(config, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor):
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
            [Wav2Vec2LayerNormConvLayer(config, layer_id=i, device=device, dtype=dtype) for i in range(config.num_feat_extract_layers)]
        )

    def forward(self, input_values: torch.Tensor):
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config: Wav2Vec2Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps, device=device, dtype=dtype)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size, device=device, dtype=dtype)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        return hidden_states, norm_hidden_states


class Wav2Vec2StateDictConverter:
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("wav2vec2.") and "masked_spec_embed" not in k:
                new_state_dict[k[len("wav2vec2."):]] = v
        return new_state_dict


class Wav2Vec2Model(PreTrainedModel):
    converter = Wav2Vec2StateDictConverter()
    _supports_parallelization = False

    def __init__(
        self,
        config: Wav2Vec2Config,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureEncoder(config, device=device, dtype=dtype)
        self.feature_projection = Wav2Vec2FeatureProjection(config, device=device, dtype=dtype)
        self.encoder = Wav2Vec2EncoderStableLayerNorm(config, device=device, dtype=dtype)

    def forward(self, input_values: torch.Tensor):
        extract_features = self.feature_extractor(input_values).transpose(1, 2)
        hidden_states, _ = self.feature_projection(extract_features)
        return self.encoder(hidden_states)

# ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆ Wav2Vec2ForCTC ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆

def get_sample_indices(original_fps, total_frames, target_fps, num_sample, fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps

    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    features: shape=[1, T, 512]
    input_fps: fps for audio, f_a
    output_fps: fps for video, f_m
    output_len: video length
    """
    features = features.transpose(1, 2)  # [1, 512, T]
    seq_len = features.shape[2] / float(input_fps)  # T/f_a
    if output_len is None:
        output_len = int(seq_len * output_fps)  # f_m*T/f_a
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )  # [1, 512, output_len]
    return output_features.transpose(1, 2)  # [1, output_len, 512]


def extract_audio_feat(audio_path: str, model: Wav2Vec2Model, dtype=torch.float32, device="cuda:0"):
    video_rate = 30
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_values = torch.stack([torch.from_numpy(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in audio_input[None]])
    feat = torch.cat(model(input_values.to(device)))
    feat = linear_interpolation(feat, input_fps=50, output_fps=video_rate)
    z = feat.to(dtype)  # Encoding for the motion
    return z

def get_audio_embed_bucket_fps(audio_embed: torch.Tensor, fps=16, batch_frames=81, m=0):
    video_rate = 30
    num_layers, audio_frame_num, audio_dim = audio_embed.shape

    if num_layers > 1:
        return_all_layers = True
    else:
        return_all_layers = False

    scale = video_rate / fps

    min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1

    bucket_num = min_batch_num * batch_frames
    padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * video_rate) - audio_frame_num
    batch_idx = get_sample_indices(
        original_fps=video_rate,
        total_frames=audio_frame_num + padd_audio_num,
        target_fps=fps,
        num_sample=bucket_num,
        fixed_start=0,
    )
    batch_audio_eb = []
    audio_sample_stride = int(video_rate / fps)
    for bi in batch_idx:
        if bi < audio_frame_num:
            chosen_idx = list(
                range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride)
            )
            chosen_idx = [0 if c < 0 else c for c in chosen_idx]
            chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]

            if return_all_layers:
                frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
            else:
                frame_audio_embed = audio_embed[0][chosen_idx].flatten()
        else:
            frame_audio_embed = (
                torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device)
                if not return_all_layers
                else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
            )
        batch_audio_eb.append(frame_audio_embed)
    batch_audio_eb = torch.stack(batch_audio_eb, dim=0)

    return batch_audio_eb, min_batch_num
