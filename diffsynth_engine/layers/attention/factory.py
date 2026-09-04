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

    - NPU MindIE (attn_type == "mindie") + SP initialized + Ulysses-only (ring degree == 1):
      AscendLongContextAttention
    - 其他: USPAttention

    AscendLongContextAttention 目前仅支持 MindIE backend 且仅支持 Ulysses 序列并行
    (sp_ring_degree == 1)。因此只有在调用方显式请求 "mindie" 且未启用 ring 并行时才路由到
    NPU 长上下文实现，其余情况一律 fallback 到 USPAttention。

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
    from diffsynth_engine.distributed.parallel_state import (
        get_ring_parallel_world_size,
        is_sp_group_initialized,
    )
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

    # AscendLongContextAttention 只服务 MindIE backend，且只支持 Ulysses（ring degree == 1）。
    # 因此必须校验 attn_type，并确认 ring 配置未启用，否则 fallback 到 USPAttention。
    # 注意短路顺序：get_ring_parallel_world_size() 依赖 SP 已初始化，必须放在 is_sp_group_initialized() 之后。
    if (
        is_mindie_sd_available()
        and is_sp_group_initialized()
        and str(attn_type) == "mindie"
        and get_ring_parallel_world_size() == 1
    ):
        from diffsynth_engine.layers.attention.ascend_long_context import AscendLongContextAttention

        return AscendLongContextAttention(**common_kwargs)
    else:
        from diffsynth_engine.layers.attention.layer import USPAttention

        return USPAttention(**common_kwargs)
