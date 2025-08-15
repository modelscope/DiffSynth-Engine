import torch
import torch.nn as nn
from typing import Dict, Optional

from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.models.basic.relative_position_emb import RelativePositionEmbedding
from diffsynth_engine.models.basic.transformer_helper import RMSNorm
from diffsynth_engine.models.basic.attention import Attention
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class T5FeedForward(nn.Module):
    # T5DenseGatedActDense
    def __init__(self, d_model, d_ff, dropout_rate, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.wo = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_gelu * hidden_linear)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        d_ff: int,
        eps: float,
        dropout_rate: float = 0.0,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.attn = Attention(
            q_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, scale=1.0, device=device, dtype=dtype
        )
        self.attn_norm = RMSNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.feed_forward = T5FeedForward(embed_dim, d_ff, dropout_rate, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        # Self Attention
        attn_output = self.attn(
            self.attn_norm(hidden_states),
            attn_mask=attention_mask,
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        # Feed Forward
        hidden_states = hidden_states + self.dropout(self.feed_forward(self.ffn_norm(hidden_states)))
        return hidden_states


class T5EncoderModelStateDictConverter(StateDictConverter):
    def __init__(self, num_encoder_layers: int = 24):
        self.num_encoder_layers = num_encoder_layers

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = {
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": "relative_position_embedding.relative_attention_bias.weight",
            "encoder.final_layer_norm.weight": "final_layer_norm.weight",
            "encoder.embed_tokens.weight": "token_embedding.weight",
            "shared.weight": "token_embedding.weight",
        }

        for i in range(self.num_encoder_layers):
            rename_dict.update(
                {
                    f"encoder.block.{i}.layer.0.SelfAttention.q.weight": f"encoders.{i}.attn.to_q.weight",
                    f"encoder.block.{i}.layer.0.SelfAttention.k.weight": f"encoders.{i}.attn.to_k.weight",
                    f"encoder.block.{i}.layer.0.SelfAttention.v.weight": f"encoders.{i}.attn.to_v.weight",
                    f"encoder.block.{i}.layer.0.SelfAttention.o.weight": f"encoders.{i}.attn.to_out.weight",
                    f"encoder.block.{i}.layer.0.layer_norm.weight": f"encoders.{i}.attn_norm.weight",
                    f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight": f"encoders.{i}.feed_forward.wi_0.weight",
                    f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight": f"encoders.{i}.feed_forward.wi_1.weight",
                    f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight": f"encoders.{i}.feed_forward.wo.weight",
                    f"encoder.block.{i}.layer.1.layer_norm.weight": f"encoders.{i}.ffn_norm.weight",
                }
            )

        new_state_dict = {}
        for key, param in state_dict.items():
            if key in rename_dict:
                new_state_dict[rename_dict[key]] = param
        return new_state_dict

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = {
            "enc.blk.0.attn_rel_b.weight": "relative_position_embedding.relative_attention_bias.weight",
            "enc.output_norm.weight": "final_layer_norm.weight",
            "token_embd.weight": "token_embedding.weight",
        }

        for i in range(self.num_encoder_layers):
            rename_dict.update(
                {
                    f"enc.blk.{i}.attn_q.weight": f"encoders.{i}.attn.to_q.weight",
                    f"enc.blk.{i}.attn_k.weight": f"encoders.{i}.attn.to_k.weight",
                    f"enc.blk.{i}.attn_v.weight": f"encoders.{i}.attn.to_v.weight",
                    f"enc.blk.{i}.attn_o.weight": f"encoders.{i}.attn.to_out.weight",
                    f"enc.blk.{i}.attn_norm.weight": f"encoders.{i}.attn_norm.weight",
                    f"enc.blk.{i}.ffn_gate.weight": f"encoders.{i}.feed_forward.wi_0.weight",
                    f"enc.blk.{i}.ffn_up.weight": f"encoders.{i}.feed_forward.wi_1.weight",
                    f"enc.blk.{i}.ffn_down.weight": f"encoders.{i}.feed_forward.wo.weight",
                    f"enc.blk.{i}.ffn_norm.weight": f"encoders.{i}.ffn_norm.weight",
                }
            )

        new_state_dict = {}
        for key, param in state_dict.items():
            if key in rename_dict:
                new_state_dict[rename_dict[key]] = param
        return new_state_dict

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "encoder.block.0.layer.0.SelfAttention.v.weight" in state_dict:
            state_dict = self._from_diffusers(state_dict)
            logger.info("use diffusers format state dict")
        elif "enc.blk.0.attn_v.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return state_dict


class T5EncoderModel(PreTrainedModel):
    converter = T5EncoderModelStateDictConverter()

    def __init__(
        self,
        embed_dim: int = 4096,
        vocab_size: int = 32128,
        num_encoder_layers: int = 24,
        d_ff: int = 10240,
        num_heads: int = 64,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.0,
        eps: float = 1e-6,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        # token_embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)

        # relative position embedding
        self.relative_position_embedding = RelativePositionEmbedding(
            relative_attention_num_buckets, relative_attention_max_distance, num_heads, device=device, dtype=dtype
        )

        # encoders
        self.encoders = nn.ModuleList(
            [
                T5EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    head_dim=embed_dim // num_heads,
                    d_ff=d_ff,
                    eps=eps,
                    dropout_rate=dropout_rate,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_encoder_layers)
            ]
        )

        # final_layer_norm
        self.final_layer_norm = RMSNorm(embed_dim, eps=eps, device=device, dtype=dtype)

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.FloatTensor] = None):
        with gguf_inference():
            inputs_embeds = self.token_embedding(input_ids)
            hidden_states = self.dropout(inputs_embeds)
            seq_len = hidden_states.shape[1]
            position_bias = self.relative_position_embedding(seq_len, seq_len)
            if attention_mask is not None:
                causal_mask = attention_mask[:, None, None, :]
                causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
                causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
                attention_mask = causal_mask + position_bias
            else:
                attention_mask = position_bias
            for layer_module in self.encoders:
                hidden_states = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)
            return hidden_states

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype, **kwargs):
        model = cls(device="meta", dtype=dtype, **kwargs)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
