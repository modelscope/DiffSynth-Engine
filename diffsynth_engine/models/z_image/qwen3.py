# modified from transformers.models.qwen3.modeling_qwen3
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.utils.cache import Cache, DynamicCache
from diffsynth_engine.utils import logging

from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.masking_utils import create_causal_mask

logger = logging.get_logger(__name__)


class Qwen3ModelStateDictConverter(StateDictConverter):
    def __init__(self):
        super().__init__()

    def _from_diffusers(self, state_dict):
        new_state_dict = {}
        for key, param in state_dict.items():
            if key.startswith("model."):
                key = key[len("model.") :]
            new_state_dict[key] = param
        return new_state_dict

    def convert(self, state_dict):
        return self._from_diffusers(state_dict)


class Qwen3Model(PreTrainedModel):
    converter = Qwen3ModelStateDictConverter()

    def __init__(self, config: Qwen3Config, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        # for causal_mask
        config._attn_implementation = "sdpa"
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(layer_idx=layer_idx, config=config).to(device=device, dtype=dtype)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps).to(device=device, dtype=dtype)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        config: Qwen3Config,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        model = cls(config=config, device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        all_hidden_states = []
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            seq_len = inputs_embeds.size(1)
            cache_position = torch.arange(seq_len, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states.append(hidden_states)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": past_key_values,
            "hidden_states": all_hidden_states,
        }
