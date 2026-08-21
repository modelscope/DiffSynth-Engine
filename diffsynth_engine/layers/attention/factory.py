"""Attention module factory for platform-aware parallel attention creation."""

import torch.nn as nn


def create_parallel_attention(
    num_heads: int,
    head_size: int,
    attn_type: str | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    num_kv_heads: int | None = None,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    **extra_impl_args,
) -> nn.Module:
    """
    根据平台能力和并行配置创建合适的序列并行 attention 模块。

    - NPU + SP initialized: AscendLongContextAttention
    - 其他: USPAttention

    Args:
        num_heads: attention head 数量
        head_size: 每个 head 的维度
        attn_type: attention backend 类型 (如 "mindie", "sdpa", "fa2" 等)
        softmax_scale: softmax 缩放系数
        causal: 是否使用因果 attention
        num_kv_heads: KV head 数量 (GQA)
        scatter_idx: Ulysses scatter 维度索引
        gather_idx: Ulysses gather 维度索引
        **extra_impl_args: 传递给底层 attention 实现的额外参数

    Returns:
        nn.Module: 配置好的 attention 模块
    """
    # Lazy imports to avoid circular dependencies
    from diffsynth_engine.distributed.parallel_state import is_sp_group_initialized
    from diffsynth_engine.utils.platform import is_mindie_sd_available

    common_kwargs = dict(
        num_heads=num_heads,
        head_size=head_size,
        softmax_scale=softmax_scale,
        causal=causal,
        num_kv_heads=num_kv_heads,
        attn_type=attn_type,
        scatter_idx=scatter_idx,
        gather_idx=gather_idx,
        **extra_impl_args,
    )

    if is_mindie_sd_available() and is_sp_group_initialized():
        from diffsynth_engine.layers.attention.ascend_long_context import AscendLongContextAttention

        return AscendLongContextAttention(**common_kwargs)
    else:
        from diffsynth_engine.layers.attention.layer import USPAttention

        return USPAttention(**common_kwargs)
