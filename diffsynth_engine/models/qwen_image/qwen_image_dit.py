import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Union, Optional
from einops import rearrange

from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.models.basic import attention as attention_ops
from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.basic.transformer_helper import AdaLayerNorm, GELU, RMSNorm
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.parallel import (
    cfg_parallel,
    cfg_parallel_unshard,
    sequence_parallel,
    sequence_parallel_unshard,
)


class QwenImageDiTStateDictConverter(StateDictConverter):
    def __init__(self):
        pass

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict_ = {}
        dim = 3072
        for name, param in state_dict.items():
            name_ = name
            if name.startswith("transformer") and "attn.to_out.0" in name:
                name_ = name.replace("attn.to_out.0", "attn.to_out")
            if "timestep_embedder.linear_1" in name:
                name_ = name.replace("timestep_embedder.linear_1", "timestep_embedder.0")
            if "timestep_embedder.linear_2" in name:
                name_ = name.replace("timestep_embedder.linear_2", "timestep_embedder.2")
            if "norm_out.linear" in name:
                param = torch.concat([param[dim:], param[:dim]], dim=0)
            state_dict_[name_] = param
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict = self._from_diffusers(state_dict)
        return state_dict


class QwenEmbedRope(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        scale_rope=False,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        with torch.device("cpu" if device == "meta" else device):
            pos_index = torch.arange(10000)
            neg_index = torch.arange(10000).flip(0) * -1 - 1
            self.pos_freqs = torch.cat(
                [
                    self.rope_params(pos_index, self.axes_dim[0], self.theta),
                    self.rope_params(pos_index, self.axes_dim[1], self.theta),
                    self.rope_params(pos_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            )
            self.neg_freqs = torch.cat(
                [
                    self.rope_params(neg_index, self.axes_dim[0], self.theta),
                    self.rope_params(neg_index, self.axes_dim[1], self.theta),
                    self.rope_params(neg_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            )
        self.rope_cache = {}
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_length, device):
        """
        Args:
            video_fhw (List[Tuple[int, int, int]]): A list of (frame, height, width) tuples for each video/image
            txt_length (int): The maximum length of the text sequences
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + txt_length, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


class QwenFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(GELU(dim, inner_dim, approximate="tanh", device=device, dtype=dtype))
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def apply_rotary_emb_qwen(x: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]):
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (b, s, h, d) -> (b, s, h, d/2, 2)
    x_out = torch.view_as_real(x_rotated * freqs_cis.unsqueeze(1)).flatten(3)  # (b, s, h, d/2, 2) -> (b, s, h, d)
    return x_out.type_as(x)


class QwenDoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.to_k = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.to_v = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.norm_q = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_k = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.add_q_proj = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)
        self.add_k_proj = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)
        self.add_v_proj = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.to_out = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.to_add_out = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)

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


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True, device=device, dtype=dtype),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            device=device,
            dtype=dtype,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.img_mlp = QwenFeedForward(dim=dim, dim_out=dim, device=device, dtype=dtype)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True, device=device, dtype=dtype),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim, device=device, dtype=dtype)

    def _modulate(self, x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
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


class QwenImageDiT(PreTrainedModel):
    converter = QwenImageDiTStateDictConverter()
    _supports_parallelization = True

    def __init__(
        self,
        num_layers: int = 60,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True, device=device)

        self.time_text_embed = TimestepEmbeddings(256, 3072, device=device, dtype=dtype)

        self.txt_norm = RMSNorm(3584, eps=1e-6, device=device, dtype=dtype)

        self.img_in = nn.Linear(64, 3072, device=device, dtype=dtype)
        self.txt_in = nn.Linear(3584, 3072, device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, device=device, dtype=dtype)
        self.proj_out = nn.Linear(3072, 64, device=device, dtype=dtype)

    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states

    def unpatchify(self, hidden_states, height, width):
        hidden_states = rearrange(
            hidden_states, "B (H W) (C P Q) -> B C (H P) (W Q)", P=2, Q=2, H=height // 2, W=width // 2
        )
        return hidden_states

    def process_entity_masks(
        self,
        text: torch.Tensor,
        text_seq_lens: torch.LongTensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        video_fhw: List[Tuple[int, int, int]],
        entity_text: List[torch.Tensor],
        entity_seq_lens: List[torch.LongTensor],
        entity_masks: List[torch.Tensor],
        device: str,
        dtype: torch.dtype,
    ):
        entity_seq_lens = [seq_lens.max().item() for seq_lens in entity_seq_lens]
        text_seq_lens = entity_seq_lens + [text_seq_lens.max().item()]
        entity_text = [
            self.txt_in(self.txt_norm(text[:, :seq_len])) for text, seq_len in zip(entity_text, entity_seq_lens)
        ]
        text = torch.cat(entity_text + [text], dim=1)

        entity_txt_freqs = [self.pos_embed(video_fhw, seq_len, device)[1] for seq_len in entity_seq_lens]
        img_freqs, txt_freqs = rotary_emb
        txt_freqs = torch.cat(entity_txt_freqs + [txt_freqs], dim=0)
        rotary_emb = (img_freqs, txt_freqs)

        global_mask = torch.ones_like(entity_masks[0], device=device, dtype=dtype)
        patched_masks = [self.patchify(mask) for mask in entity_masks + [global_mask]]
        batch_size, image_seq_len = patched_masks[0].shape[:2]
        total_seq_len = sum(text_seq_lens) + image_seq_len
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), device=device, dtype=torch.bool)

        # text-image attention mask
        img_start, img_end = sum(text_seq_lens), total_seq_len
        cumsum = [0]
        for seq_len in text_seq_lens:
            cumsum.append(cumsum[-1] + seq_len)
        for i, patched_mask in enumerate(patched_masks):
            txt_start, txt_end = cumsum[i], cumsum[i + 1]
            mask = torch.sum(patched_mask, dim=-1) > 0
            mask = mask.unsqueeze(1).repeat(1, text_seq_lens[i], 1)
            # text-to-image attention
            attention_mask[:, txt_start:txt_end, img_start:img_end] = mask
            # image-to-text attention
            attention_mask[:, img_start:img_end, txt_start:txt_end] = mask.transpose(1, 2)
        # entity text tokens should not attend to each other
        for i in range(len(text_seq_lens)):
            for j in range(len(text_seq_lens)):
                if i == j:
                    continue
                i_start, i_end = cumsum[i], cumsum[i + 1]
                j_start, j_end = cumsum[j], cumsum[j + 1]
                attention_mask[:, i_start:i_end, j_start:j_end] = False

        attn_mask = torch.zeros_like(attention_mask, device=device, dtype=dtype)
        attn_mask[~attention_mask] = -torch.inf
        attn_mask = attn_mask.unsqueeze(1)
        return text, rotary_emb, attn_mask

    def forward(
        self,
        image: torch.Tensor,
        edit: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        text: torch.Tensor = None,
        text_seq_lens: torch.LongTensor = None,
        context_latents: Optional[torch.Tensor] = None,
        entity_text: Optional[List[torch.Tensor]] = None,
        entity_seq_lens: Optional[List[torch.LongTensor]] = None,
        entity_masks: Optional[List[torch.Tensor]] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        h, w = image.shape[-2:]
        fp8_linear_enabled = getattr(self, "fp8_linear_enabled", False)
        use_cfg = image.shape[0] > 1
        with (
            fp8_inference(fp8_linear_enabled),
            gguf_inference(),
            cfg_parallel(
                (
                    image,
                    *(edit if edit is not None else ()),
                    timestep,
                    text,
                    text_seq_lens,
                    *(entity_text if entity_text is not None else ()),
                    *(entity_seq_lens if entity_seq_lens is not None else ()),
                    *(entity_masks if entity_masks is not None else ()),
                    context_latents,
                ),
                use_cfg=use_cfg,
            ),
        ):
            conditioning = self.time_text_embed(timestep, image.dtype)
            video_fhw = [(1, h // 2, w // 2)]  # frame, height, width
            text_seq_len = text_seq_lens.max().item()
            image = self.patchify(image)
            image_seq_len = image.shape[1]
            if context_latents is not None:
                context_latents = context_latents.to(dtype=image.dtype)
                context_latents = self.patchify(context_latents)
                image = torch.cat([image, context_latents], dim=1)
                video_fhw += [(1, h // 2, w // 2)]
            if edit is not None:
                for img in edit:
                    img = img.to(dtype=image.dtype)
                    edit_h, edit_w = img.shape[-2:]
                    img = self.patchify(img)
                    image = torch.cat([image, img], dim=1)
                    video_fhw += [(1, edit_h // 2, edit_w // 2)]

            rotary_emb = self.pos_embed(video_fhw, text_seq_len, image.device)

            image = self.img_in(image)
            text = self.txt_in(self.txt_norm(text[:, :text_seq_len]))

            attn_mask = None
            if entity_text is not None:
                text, rotary_emb, attn_mask = self.process_entity_masks(
                    text,
                    text_seq_lens,
                    rotary_emb,
                    video_fhw,
                    entity_text,
                    entity_seq_lens,
                    entity_masks,
                    image.device,
                    image.dtype,
                )

            # warning: Eligen does not work with sequence parallel because long context attention does not support attention masks
            img_freqs, txt_freqs = rotary_emb
            with sequence_parallel((image, text, img_freqs, txt_freqs), seq_dims=(1, 1, 0, 0)):
                rotary_emb = (img_freqs, txt_freqs)
                for block in self.transformer_blocks:
                    text, image = block(
                        image=image,
                        text=text,
                        temb=conditioning,
                        rotary_emb=rotary_emb,
                        attn_mask=attn_mask,
                        attn_kwargs=attn_kwargs,
                    )
                image = self.norm_out(image, conditioning)
                image = self.proj_out(image)
                (image,) = sequence_parallel_unshard((image,), seq_dims=(1,), seq_lens=(image_seq_len,))
            image = image[:, :image_seq_len]
            image = self.unpatchify(image, h, w)

        (image,) = cfg_parallel_unshard((image,), use_cfg=use_cfg)
        return image

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        num_layers: int = 60,
    ):
        model = cls(device="meta", dtype=dtype, num_layers=num_layers)
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    def compile_repeated_blocks(self, *args, **kwargs):
        for block in self.transformer_blocks:
            block.compile(*args, **kwargs)

    def get_fsdp_module_cls(self):
        return {QwenImageTransformerBlock}
