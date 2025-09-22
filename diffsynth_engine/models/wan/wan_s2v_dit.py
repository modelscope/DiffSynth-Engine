import json
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffsynth_engine.models.basic.transformer_helper import AdaLayerNorm
from diffsynth_engine.models.wan.wan_dit import (
    WanDiT,
    DiTBlock,
    CrossAttention,
    sinusoidal_embedding_1d,
    precompute_freqs_cis_3d,
    modulate,
)
from diffsynth_engine.utils.constants import WAN2_2_DIT_S2V_14B_CONFIG_FILE
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.parallel import (
    cfg_parallel,
    cfg_parallel_unshard,
    sequence_parallel,
    sequence_parallel_unshard,
)


def rope_precompute(x: torch.Tensor, grid_sizes: List[List[torch.Tensor]], freqs: torch.Tensor):
    # roughly speaking, this function is to combine ropes, but it is written in a very strange way.
    # I try to make it align better with normal implementation
    b, s, n, c = x.shape
    c = c // 2
    output = torch.view_as_complex(x.reshape(b, s, n, c, 2).to(torch.float64))
    prev_seq = 0
    for grid_size in grid_sizes:
        f_o, h_o, w_o = grid_size[0]
        f, h, w = grid_size[1]
        t_f, t_h, t_w = grid_size[2]
        seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
        seq_len = int(seq_f * seq_h * seq_w)
        # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
        if f_o >= 0:
            f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
        else:
            f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
        h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
        w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()
        freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
        freqs_i = torch.cat(
            [
                freqs_0.view(seq_f, 1, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        # apply rotary embedding
        output[:, prev_seq : prev_seq + seq_len] = freqs_i
        prev_seq += seq_len
    return output


class FramePackMotioner(nn.Module):
    def __init__(
        self,
        inner_dim: int = 1024,
        num_heads: int = 16,
        zip_frame_buckets: List[int] = [
            1,
            2,
            16,
        ],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2), device=device, dtype=dtype)
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4), device=device, dtype=dtype)
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8), device=device, dtype=dtype)
        self.zip_frame_buckets = zip_frame_buckets

        self.inner_dim = inner_dim
        self.num_heads = num_heads

        assert (inner_dim % num_heads) == 0 and (inner_dim // num_heads) % 2 == 0
        head_dim = inner_dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

    def forward(self, motion_latents: torch.Tensor):
        b, _, f, h, w = motion_latents.shape
        padd_latents = torch.zeros(
            (b, 16, sum(self.zip_frame_buckets), h, w), device=motion_latents.device, dtype=motion_latents.dtype
        )
        overlap_frame = min(padd_latents.shape[2], f)
        if overlap_frame > 0:
            padd_latents[:, :, -overlap_frame:] = motion_latents[:, :, -overlap_frame:]

        clean_latents_4x, clean_latents_2x, clean_latents_post = padd_latents[
            :, :, -sum(self.zip_frame_buckets) :
        ].split(self.zip_frame_buckets[::-1], dim=2)  # 16, 2 ,1

        clean_latents_post = rearrange(self.proj(clean_latents_post), "b c f h w -> b (f h w) c").contiguous()
        clean_latents_2x = rearrange(self.proj_2x(clean_latents_2x), "b c f h w -> b (f h w) c").contiguous()
        clean_latents_4x = rearrange(self.proj_4x(clean_latents_4x), "b c f h w -> b (f h w) c").contiguous()
        motion_latents = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

        def get_grid_sizes(i: int):  # rope, 0: post, 1: 2x, 2: 4x
            start_time_id = -sum(self.zip_frame_buckets[: (i + 1)])
            end_time_id = start_time_id + self.zip_frame_buckets[i] // (2**i)
            return [
                [
                    torch.tensor([start_time_id, 0, 0]),
                    torch.tensor([end_time_id, h // (2 ** (i + 1)), w // (2 ** (i + 1))]),
                    torch.tensor([self.zip_frame_buckets[i], h // 2, w // 2]),
                ]
            ]

        motion_rope_emb = rope_precompute(
            x=rearrange(motion_latents, "b s (n d) -> b s n d", n=self.num_heads),
            grid_sizes=get_grid_sizes(0) + get_grid_sizes(1) + get_grid_sizes(2),
            freqs=self.freqs,
        )

        return motion_latents, motion_rope_emb


class CausalConv1d(nn.Module):
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "replicate",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1, device=device, dtype=dtype)
        self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1, device=device, dtype=dtype)
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2, device=device, dtype=dtype)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.act = nn.SiLU()
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.final_linear = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, "b t c -> b c t")
        x_original = x
        b = x.shape[0]
        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        x = self.act(self.norm1(x))
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.act(self.norm2(x))
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.act(self.norm3(x))
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x

        x = self.conv1_global(x_original)
        x = rearrange(x, "b c t -> b t c")
        x = self.act(self.norm1(x))
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.act(self.norm2(x))
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.act(self.norm3(x))
        x = self.final_linear(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        return x, x_local


class CausalAudioEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        num_layers: int = 25,
        out_dim: int = 2048,
        num_token: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.encoder = MotionEncoder(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, device=device, dtype=dtype)
        self.weights = nn.Parameter(torch.ones((1, num_layers, 1, 1), device=device, dtype=dtype) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor):
        # features: b num_layers dim video_length
        weights = self.act(self.weights)
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        return self.encoder(weighted_feat)  # b f n dim


class AudioInjector(nn.Module):
    def __init__(
        self,
        dim=5120,
        num_heads=40,
        inject_layers=[0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
        adain_dim=5120,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.injected_block_id = {}
        for i, id in enumerate(inject_layers):
            self.injected_block_id[id] = i

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(len(inject_layers))
            ]
        )
        self.injector_adain_layers = nn.ModuleList(
            [AdaLayerNorm(dim=adain_dim, device=device, dtype=dtype) for _ in range(len(inject_layers))]
        )


class DiTBlockS2V(nn.Module):
    def __init__(self, dit_block: DiTBlock):
        super().__init__()
        self.dim = dit_block.dim
        self.num_heads = dit_block.num_heads
        self.ffn_dim = dit_block.ffn_dim
        self.self_attn = dit_block.self_attn
        self.cross_attn = dit_block.cross_attn
        self.norm1 = dit_block.norm1
        self.norm2 = dit_block.norm2
        self.norm3 = dit_block.norm3
        self.ffn = dit_block.ffn
        self.modulation = dit_block.modulation

    def forward(self, x, x_seq_len, context, t_mod, t_mod_0, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t for t in (self.modulation + t_mod).chunk(6, dim=1)
        ]
        shift_msa_0, scale_msa_0, gate_msa_0, shift_mlp_0, scale_mlp_0, gate_mlp_0 = [
            t for t in (self.modulation + t_mod_0).chunk(6, dim=1)
        ]
        norm1_x = self.norm1(x)
        input_x = torch.cat(
            [
                modulate(norm1_x[:, :x_seq_len], shift_msa, scale_msa),
                modulate(norm1_x[:, x_seq_len:], shift_msa_0, scale_msa_0),
            ],
            dim=1,
        )
        self_attn_x = self.self_attn(input_x, freqs)
        x += torch.cat([self_attn_x[:, :x_seq_len] * gate_msa, self_attn_x[:, x_seq_len:] * gate_msa_0], dim=1)
        x += self.cross_attn(self.norm3(x), context)
        norm2_x = self.norm2(x)
        input_x = torch.cat(
            [
                modulate(norm2_x[:, :x_seq_len], shift_mlp, scale_mlp),
                modulate(norm2_x[:, x_seq_len:], shift_mlp_0, scale_mlp_0),
            ],
            dim=1,
        )
        ffn_x = self.ffn(input_x)
        x += torch.cat([ffn_x[:, :x_seq_len] * gate_mlp, ffn_x[:, x_seq_len:] * gate_mlp_0], dim=1)
        return x


class WanS2VDiT(WanDiT):
    def __init__(
        self,
        cond_dim: int = 16,
        audio_dim: int = 1024,
        num_audio_token: int = 4,
        audio_inject_layers: List[int] = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
        num_heads: int = 40,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        *args,
        **kwargs,
    ):
        super().__init__(num_heads=num_heads, device=device, dtype=dtype, *args, **kwargs)
        self.num_heads = num_heads
        self.cond_encoder = nn.Conv3d(
            cond_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size, device=device, dtype=dtype
        )
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim, out_dim=self.dim, num_token=num_audio_token, device=device, dtype=dtype
        )
        self.audio_injector = AudioInjector(
            dim=self.dim,
            num_heads=num_heads,
            inject_layers=audio_inject_layers,
            adain_dim=self.dim,
            device=device,
            dtype=dtype,
        )
        self.trainable_cond_mask = nn.Embedding(3, self.dim, device=device, dtype=dtype)
        self.frame_packer = FramePackMotioner(
            inner_dim=self.dim,
            num_heads=num_heads,
            zip_frame_buckets=[1, 2, 16],
            device=device,
            dtype=dtype,
        )
        dit_blocks_s2v: nn.ModuleList[DiTBlockS2V] = nn.ModuleList()
        for block in self.blocks:
            dit_blocks_s2v.append(DiTBlockS2V(block))
        self.blocks = dit_blocks_s2v

    @staticmethod
    def get_model_config(model_type: str):
        MODEL_CONFIG_FILES = {
            "wan2.2-s2v-14b": WAN2_2_DIT_S2V_14B_CONFIG_FILE,
        }
        if model_type not in MODEL_CONFIG_FILES:
            raise ValueError(f"Unsupported model type: {model_type}")

        config_file = MODEL_CONFIG_FILES[model_type]
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    def inject_motion(
        self,
        x: torch.Tensor,
        x_seq_len: int,
        rope_embs: torch.Tensor,
        motion_latents: torch.Tensor,
        drop_motion_frames: bool = False,
    ):
        # Initialize masks to indicate noisy latent, ref latent, and motion latent.
        b, s, _ = x.shape
        mask_input = torch.zeros([b, s], dtype=torch.long, device=x.device)
        mask_input[:, x_seq_len:] = 1

        if not drop_motion_frames:
            motion, motion_rope_emb = self.frame_packer(motion_latents)
            x = torch.cat([x, motion], dim=1)
            rope_embs = torch.cat([rope_embs, motion_rope_emb], dim=1)
            mask_input = torch.cat(
                [
                    mask_input,
                    2 * torch.ones([b, motion.shape[1]], device=mask_input.device, dtype=mask_input.dtype),
                ],
                dim=1,
            )
        x += self.trainable_cond_mask(mask_input).to(x.dtype)
        return x, rope_embs

    def patchify_x_with_pose(self, x: torch.Tensor, pose: torch.Tensor):
        x = self.patch_embedding(x) + self.cond_encoder(pose)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def forward(
        self,
        x: torch.Tensor,  # b c tx h w
        context: torch.Tensor,  # b s c
        timestep: torch.Tensor,  # b
        ref_latents: torch.Tensor,  # b c 1 h w
        motion_latents: torch.Tensor,  # b c tm h w
        pose_cond: torch.Tensor,  # b c tx h w
        audio_input: torch.Tensor,  # b c d tx
        num_motion_frames: int = 73,
        num_motion_latents: int = 19,
        drop_motion_frames: bool = False,  # !(ref_as_first_frame || clip_idx)
        audio_mask: Optional[torch.Tensor] = None,  # b c tx h w
        void_audio_input: Optional[torch.Tensor] = None,
    ):
        fp8_linear_enabled = getattr(self, "fp8_linear_enabled", False)
        use_cfg = x.shape[0] > 1
        with (
            fp8_inference(fp8_linear_enabled),
            gguf_inference(),
            cfg_parallel((x, context, audio_input), use_cfg=use_cfg),
        ):
            audio_emb_global, merged_audio_emb, void_audio_emb_global, void_merged_audio_emb, audio_mask = (
                self.get_audio_emb(audio_input, num_motion_frames, num_motion_latents, audio_mask, void_audio_input)
            )
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))  # (s, d)
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
            t_mod_0 = self.time_projection(
                self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, torch.zeros([1]).to(t)))
            ).unflatten(1, (6, self.dim))
            context = self.text_embedding(context)
            x, (f, h, w) = self.patchify_x_with_pose(x, pose_cond)
            ref, _ = self.patchify(ref_latents)
            x = torch.cat([x, ref], dim=1)
            freqs = rope_precompute(
                x=rearrange(x, "b s (n d) -> b s n d", n=self.num_heads),
                grid_sizes=[
                    [
                        torch.tensor([0, 0, 0]),
                        torch.tensor([f, h, w]),
                        torch.tensor([f, h, w]),
                    ],  # grid size of x
                    [
                        torch.tensor([30, 0, 0]),
                        torch.tensor([31, h, w]),
                        torch.tensor([1, h, w]),
                    ],  # grid size of ref
                ],
                freqs=self.freqs,
            )
            # why do they fix 30?
            # seems that they just want self.freqs[0][30]

            x_seq_len = f * h * w
            x, freqs = self.inject_motion(
                x=x,
                x_seq_len=x_seq_len,
                rope_embs=freqs,
                motion_latents=motion_latents,
                drop_motion_frames=drop_motion_frames,
            )

            # f must be divisible by ulysses world size
            x_img, freqs_img = x[:, :x_seq_len], freqs[:, :x_seq_len]
            x_ref_motion, freqs_ref_motion = x[:, x_seq_len:], freqs[:, x_seq_len:]
            with sequence_parallel(
                tensors=(
                    x_img,
                    freqs_img,
                    audio_emb_global,
                    merged_audio_emb,
                    audio_mask,
                    void_audio_emb_global,
                    void_merged_audio_emb,
                ),
                seq_dims=(1, 1, 1, 1, 1, 1, 1),
            ):
                x_seq_len_local = x_img.shape[1]
                x = torch.concat([x_img, x_ref_motion], dim=1)
                freqs = torch.concat([freqs_img, freqs_ref_motion], dim=1)
                for idx, block in enumerate(self.blocks):
                    x = block(
                        x=x, x_seq_len=x_seq_len_local, context=context, t_mod=t_mod, t_mod_0=t_mod_0, freqs=freqs
                    )
                    if idx in self.audio_injector.injected_block_id.keys():
                        x = self.inject_audio(
                            x=x,
                            x_seq_len=x_seq_len_local,
                            block_idx=idx,
                            audio_emb_global=audio_emb_global,
                            merged_audio_emb=merged_audio_emb,
                            audio_mask=audio_mask,
                            void_audio_emb_global=void_audio_emb_global,
                            void_merged_audio_emb=void_merged_audio_emb,
                        )

                x = x[:, :x_seq_len_local]
                x = self.head(x, t)
                (x,) = sequence_parallel_unshard((x,), seq_dims=(1,), seq_lens=(x_seq_len,))
            x = self.unpatchify(x, (f, h, w))
            (x,) = cfg_parallel_unshard((x,), use_cfg=use_cfg)
            return x

    def get_audio_emb(
        self,
        audio_input: torch.Tensor,
        num_motion_frames: int = 73,
        num_motion_latents: int = 19,
        audio_mask: Optional[torch.Tensor] = None,
        void_audio_input: Optional[torch.Tensor] = None,
    ):
        void_audio_emb_global, void_merged_audio_emb = None, None
        if audio_mask is not None:
            audio_mask = rearrange(audio_mask, "b c f h w -> b (f h w) c").contiguous()
            void_audio_input = torch.cat(
                [void_audio_input[..., 0:1].repeat(1, 1, 1, num_motion_frames), void_audio_input], dim=-1
            )
            void_audio_emb_global, void_audio_emb = self.casual_audio_encoder(void_audio_input)
            void_audio_emb_global = void_audio_emb_global[:, num_motion_latents:]
            void_merged_audio_emb = void_audio_emb[:, num_motion_latents:, :]

        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, num_motion_frames), audio_input], dim=-1)
        audio_emb_global, audio_emb = self.casual_audio_encoder(audio_input)
        audio_emb_global = audio_emb_global[:, num_motion_latents:]
        merged_audio_emb = audio_emb[:, num_motion_latents:, :]
        return audio_emb_global, merged_audio_emb, void_audio_emb_global, void_merged_audio_emb, audio_mask

    def inject_audio(
        self,
        x: torch.Tensor,
        x_seq_len: int,
        block_idx: int,
        audio_emb_global: torch.Tensor,
        merged_audio_emb: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        void_audio_emb_global: Optional[torch.Tensor] = None,
        void_merged_audio_emb: Optional[torch.Tensor] = None,
    ):
        audio_attn_id = self.audio_injector.injected_block_id[block_idx]
        num_latents_per_clip = merged_audio_emb.shape[1]

        x_input = x[:, :x_seq_len]  # b (f h w) c
        x_input = rearrange(x_input, "b (t n) c -> (b t) n c", t=num_latents_per_clip)

        def calc_x_adain(x_input: torch.Tensor, audio_emb_global: torch.Tensor):
            audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c")
            return self.audio_injector.injector_adain_layers[audio_attn_id](x_input, emb=audio_emb_global[:, 0])

        x_adain = calc_x_adain(x_input, audio_emb_global)
        if void_audio_emb_global is not None:
            x_void_adain = calc_x_adain(x_input, void_audio_emb_global)

        def calc_x_residual(x_adain: torch.Tensor, merged_audio_emb: torch.Tensor):
            merged_audio_emb = rearrange(merged_audio_emb, "b t n c -> (b t) n c", t=num_latents_per_clip)
            x_cond_residual = self.audio_injector.injector[audio_attn_id](
                x=x_adain,
                y=merged_audio_emb,
            )
            return rearrange(x_cond_residual, "(b t) n c -> b (t n) c", t=num_latents_per_clip)

        x_cond_residual = calc_x_residual(x_adain, merged_audio_emb)
        if audio_mask is not None:
            x_uncond_residual = calc_x_residual(x_void_adain, void_merged_audio_emb)
            x[:, :x_seq_len] += x_cond_residual * audio_mask + x_uncond_residual * (1 - audio_mask)
        else:
            x[:, :x_seq_len] += x_cond_residual

        return x
