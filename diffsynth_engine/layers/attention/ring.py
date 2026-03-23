import torch

from diffsynth_engine.layers.attention.backends.abstract import AttentionImpl


def ring_flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_impl: AttentionImpl,
):
    # TODO: implement ring flash attention
    raise NotImplementedError("Ring Flash Attention is not supported yet")
