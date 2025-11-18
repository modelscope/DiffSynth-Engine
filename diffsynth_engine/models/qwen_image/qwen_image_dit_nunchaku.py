import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Optional
from einops import rearrange

from diffsynth_engine.models.basic import attention as attention_ops
from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.basic.transformer_helper import AdaLayerNorm, RMSNorm
from diffsynth_engine.models.qwen_image.qwen_image_dit import (
    QwenFeedForward,
    apply_rotary_emb_qwen,
    QwenDoubleStreamAttention,
    QwenImageTransformerBlock,
    QwenImageDiT,
    QwenEmbedRope,
)

from nunchaku.models.utils import fuse_linears
from nunchaku.ops.fused import fused_gelu_mlp
from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear
from diffsynth_engine.models.basic.lora import LoRALinear, LoRAConv2d
from diffsynth_engine.models.basic.lora_nunchaku import LoRASVDQW4A4Linear, LoRAAWQW4A16Linear


class QwenDoubleStreamAttentionNunchaku(QwenDoubleStreamAttention):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        nunchaku_rank: int = 32,
    ):
        super().__init__(dim_a, dim_b, num_heads, head_dim, device=device, dtype=dtype)

        to_qkv = fuse_linears([self.to_q, self.to_k, self.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, rank=nunchaku_rank)
        self.to_out = SVDQW4A4Linear.from_linear(self.to_out, rank=nunchaku_rank)

        del self.to_q, self.to_k, self.to_v

        add_qkv_proj = fuse_linears([self.add_q_proj, self.add_k_proj, self.add_v_proj])
        self.add_qkv_proj = SVDQW4A4Linear.from_linear(add_qkv_proj, rank=nunchaku_rank)
        self.to_add_out = SVDQW4A4Linear.from_linear(self.to_add_out, rank=nunchaku_rank)

        del self.add_q_proj, self.add_k_proj, self.add_v_proj

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_qkv(image).chunk(3, dim=-1)
        txt_q, txt_k, txt_v = self.add_qkv_proj(text).chunk(3, dim=-1)

        img_q = rearrange(img_q, "b s (h d) -> b s h d", h=self.num_heads)
        img_k = rearrange(img_k, "b s (h d) -> b s h d", h=self.num_heads)
        img_v = rearrange(img_v, "b s (h d) -> b s h d", h=self.num_heads)

        txt_q = rearrange(txt_q, "b s (h d) -> b s h d", h=self.num_heads)
        txt_k = rearrange(txt_k, "b s (h d) -> b s h d", h=self.num_heads)
        txt_v = rearrange(txt_v, "b s (h d) -> b s h d", h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)

        if rotary_emb is not None:
            img_freqs, txt_freqs = rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=1)
        joint_k = torch.cat([txt_k, img_k], dim=1)
        joint_v = torch.cat([txt_v, img_v], dim=1)

        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        joint_attn_out = attention_ops.attention(joint_q, joint_k, joint_v, attn_mask=attn_mask, **attn_kwargs)

        joint_attn_out = rearrange(joint_attn_out, "b s h d -> b s (h d)").to(joint_q.dtype)

        txt_attn_output = joint_attn_out[:, : text.shape[1], :]
        img_attn_output = joint_attn_out[:, text.shape[1] :, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenFeedForwardNunchaku(QwenFeedForward):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        rank: int = 32,
    ):
        super().__init__(dim, dim_out, dropout, device=device, dtype=dtype)
        self.net[0].proj = SVDQW4A4Linear.from_linear(self.net[0].proj, rank=rank)
        self.net[2] = SVDQW4A4Linear.from_linear(self.net[2], rank=rank)
        self.net[2].act_unsigned = self.net[2].precision != "nvfp4"

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])


class QwenImageTransformerBlockNunchaku(QwenImageTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        scale_shift: float = 1.0,
        use_nunchaku_awq: bool = True,
        use_nunchaku_attn: bool = True,
        nunchaku_rank: int = 32,
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, eps, device=device, dtype=dtype)

        self.use_nunchaku_awq = use_nunchaku_awq
        if use_nunchaku_awq:
            self.img_mod[1] = AWQW4A16Linear.from_linear(self.img_mod[1], rank=nunchaku_rank)

        if use_nunchaku_attn:
            self.attn = QwenDoubleStreamAttentionNunchaku(
                dim_a=dim,
                dim_b=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                device=device,
                dtype=dtype,
                nunchaku_rank=nunchaku_rank,
            )
        else:
            self.attn = QwenDoubleStreamAttention(
                dim_a=dim,
                dim_b=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                device=device,
                dtype=dtype,
            )

        self.img_mlp = QwenFeedForwardNunchaku(dim=dim, dim_out=dim, device=device, dtype=dtype, rank=nunchaku_rank)

        if use_nunchaku_awq:
            self.txt_mod[1] = AWQW4A16Linear.from_linear(self.txt_mod[1], rank=nunchaku_rank)

        self.txt_mlp = QwenFeedForwardNunchaku(dim=dim, dim_out=dim, device=device, dtype=dtype, rank=nunchaku_rank)

        self.scale_shift = scale_shift

    def _modulate(self, x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.use_nunchaku_awq:
            if self.scale_shift != 0:
                scale.add_(self.scale_shift)
            return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)
        else:
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_nunchaku_awq:
            img_mod_params = self.img_mod(temb)  # [B, 6*dim]
            txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

            # nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
            img_mod_params = (
                img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
            )
            txt_mod_params = (
                txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
            )

            img_mod_attn, img_mod_mlp = img_mod_params.chunk(2, dim=-1)  # [B, 3*dim] each
            txt_mod_attn, txt_mod_mlp = txt_mod_params.chunk(2, dim=-1)  # [B, 3*dim] each
        else:
            img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each
            txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            rotary_emb=rotary_emb,
            attn_mask=attn_mask,
            attn_kwargs=attn_kwargs,
        )

        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        return text, image


class QwenImageDiTNunchaku(QwenImageDiT):
    def __init__(
        self,
        num_layers: int = 60,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        use_nunchaku_awq: bool = True,
        use_nunchaku_attn: bool = True,
        nunchaku_rank: int = 32,
    ):
        super().__init__(num_layers, device=device, dtype=dtype)

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True, device=device)

        self.time_text_embed = TimestepEmbeddings(256, 3072, device=device, dtype=dtype)

        self.txt_norm = RMSNorm(3584, eps=1e-6, device=device, dtype=dtype)

        self.img_in = nn.Linear(64, 3072, device=device, dtype=dtype)
        self.txt_in = nn.Linear(3584, 3072, device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlockNunchaku(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                    device=device,
                    dtype=dtype,
                    scale_shift=0,
                    use_nunchaku_awq=use_nunchaku_awq,
                    use_nunchaku_attn=use_nunchaku_attn,
                    nunchaku_rank=nunchaku_rank,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, device=device, dtype=dtype)
        self.proj_out = nn.Linear(3072, 64, device=device, dtype=dtype)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        num_layers: int = 60,
        use_nunchaku_awq: bool = True,
        use_nunchaku_attn: bool = True,
        nunchaku_rank: int = 32,
    ):
        model = cls(
            device="meta",
            dtype=dtype,
            num_layers=num_layers,
            use_nunchaku_awq=use_nunchaku_awq,
            use_nunchaku_attn=use_nunchaku_attn,
            nunchaku_rank=nunchaku_rank,
        )
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, non_blocking=True)
        return model

    def load_loras(self, lora_args: List[Dict[str, Any]], fused: bool = False):
        fuse_dict = {}
        for args in lora_args:
            key = args["key"]
            if any(suffix in key for suffix in {"add_q_proj", "add_k_proj", "add_v_proj"}):
                fuse_key = f"{key.rsplit('.', 1)[0]}.add_qkv_proj"
                type = key.rsplit(".", 1)[-1].split("_")[1]
                fuse_dict[fuse_key] = fuse_dict.get(fuse_key, {})
                fuse_dict[fuse_key][type] = args
                continue

            if any(suffix in key for suffix in {"to_q", "to_k", "to_v"}):
                fuse_key = f"{key.rsplit('.', 1)[0]}.to_qkv"
                type = key.rsplit(".", 1)[-1].split("_")[1]
                fuse_dict[fuse_key] = fuse_dict.get(fuse_key, {})
                fuse_dict[fuse_key][type] = args
                continue

            module = self.get_submodule(key)
            if not isinstance(module, (LoRALinear, LoRAConv2d, LoRASVDQW4A4Linear, LoRAAWQW4A16Linear)):
                raise ValueError(f"Unsupported lora key: {key}")

            if fused and not isinstance(module, LoRAAWQW4A16Linear):
                module.add_frozen_lora(**args)
            else:
                module.add_lora(**args)

        for key in fuse_dict.keys():
            module = self.get_submodule(key)
            if not isinstance(module, LoRASVDQW4A4Linear):
                raise ValueError(f"Unsupported lora key: {key}")
            module.add_qkv_lora(
                name=args["name"],
                scale=fuse_dict[key]["q"]["scale"],
                rank=fuse_dict[key]["q"]["rank"],
                alpha=fuse_dict[key]["q"]["alpha"],
                q_up=fuse_dict[key]["q"]["up"],
                q_down=fuse_dict[key]["q"]["down"],
                k_up=fuse_dict[key]["k"]["up"],
                k_down=fuse_dict[key]["k"]["down"],
                v_up=fuse_dict[key]["v"]["up"],
                v_down=fuse_dict[key]["v"]["down"],
                device=fuse_dict[key]["q"]["device"],
                dtype=fuse_dict[key]["q"]["dtype"],
            )
