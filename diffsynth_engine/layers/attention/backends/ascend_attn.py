"""Ascend NPU attention backend built on torch_npu's fused attention kernel.

Why this backend exists
-----------------------
``attn_type="sdpa"`` does work on Ascend for the plain forward path, because
torch_npu maps ``F.scaled_dot_product_attention`` onto its own fused kernel.
Its ``forward_with_lse`` path does not: it calls
``torch.ops.aten._scaled_dot_product_efficient_attention``, a CUDA-only ATen
operator that has no PrivateUse1 kernel. PyTorch then falls back to the CPU
implementation and dies with::

    NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention'
    with arguments from the 'CPU' backend

Ring attention merges per-shard softmax statistics, so it needs
``forward_with_lse`` -- which means ``sp_ring_degree > 1`` cannot run on Ascend
without a backend that provides it.

``torch_npu.npu_fusion_attention`` returns ``softmax_max`` and ``softmax_sum``
next to the output, and ``log(sum) + max`` is exactly the log-sum-exp that ring
attention needs. Measured against a float32 reference on Ascend950PR: outputs
match SDPA exactly, LSE matches to ~1e-6. No extra kernel required.
"""

import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)


def _require_torch_npu():
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise ImportError(
            "attn_type='npu' requires torch_npu, which is not installed. "
            "Install the Ascend build of PyTorch or pick another attention backend."
        ) from exc
    return torch_npu


class AscendAttentionBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        _require_torch_npu()

    @staticmethod
    def get_type() -> str:
        return str(AttentionType.NPU)

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return AscendAttentionImpl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # npu_fusion_attention accepts any head size that is a multiple of 8
        # (16 for fp16 on some SoCs). Returning an empty list disables the
        # pre-check, same as the sdpa backend.
        return []

    @classmethod
    def supports_ring_attention(cls) -> bool:
        return True


class AscendAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_groups = num_heads // num_kv_heads
        self.causal = causal
        # Resolve the default here instead of passing None down: the fused
        # kernel wants a concrete float and does not apply the 1/sqrt(d) itself.
        self.softmax_scale = softmax_scale if softmax_scale is not None else head_size**-0.5

    def _expand_kv(self, key: torch.Tensor, value: torch.Tensor):
        """GQA/MQA: repeat K/V heads up to Q heads.

        The fused kernel does accept ``num_key_value_heads``, but repeating is
        what the sdpa backend does and it keeps a single code path for every
        head configuration. Qwen-Image's DiT is plain MHA, so this is a no-op
        there.
        """
        if self.num_kv_groups == 1:
            return key, value
        # BSND layout -> head dim is 2 (sdpa transposes to BNSD first, hence dim 1)
        return (
            torch.repeat_interleave(key, self.num_kv_groups, dim=2),
            torch.repeat_interleave(value, self.num_kv_groups, dim=2),
        )

    @staticmethod
    def _to_bool_mask(attn_mask: torch.Tensor | None) -> torch.Tensor | None:
        """aten_mask wants a boolean mask: True = keep, False = mask out.

        Additive float masks (0 for keep, -inf for masked) are the common form
        elsewhere, so convert rather than reject.
        """
        if attn_mask is None:
            return None
        if attn_mask.dtype == torch.bool:
            return attn_mask
        return attn_mask >= 0

    def _fused_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ):
        torch_npu = _require_torch_npu()
        bool_mask = self._to_bool_mask(attn_mask)
        # sparse_mode=2 is the right-down causal triangle, which the kernel
        # generates internally; it must not be combined with an explicit mask.
        sparse_mode = 2 if (self.causal and bool_mask is None) else 0
        return torch_npu.npu_fusion_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            head_num=query.shape[2],
            input_layout="BSND",
            scale=self.softmax_scale,
            keep_prob=1.0,
            sparse_mode=sparse_mode,
            atten_mask=bool_mask,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        key, value = self._expand_kv(key, value)
        return self._fused_attention(query, key, value, attn_mask)[0]

    def forward_with_lse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention plus log-sum-exp, for ring attention to merge shards.

        Returns:
            output: [B, S, H, D]
            lse:    [B, H, S] -- the layout ring.py expects (it transposes to
                    [B, S, H, 1] internally).
        """
        key, value = self._expand_kv(key, value)
        result = self._fused_attention(query, key, value, attn_mask)

        output, softmax_max, softmax_sum = result[0], result[1], result[2]
        if not torch.is_tensor(softmax_sum) or softmax_sum.numel() == 0:
            raise RuntimeError(
                "npu_fusion_attention returned no softmax statistics, so ring attention "
                "cannot merge shards. This happens on CANN builds that only fill "
                "softmax_max/softmax_sum when gradients are required."
            )

        # Shape is [B, H, S, 8]: the trailing dim is alignment padding that
        # repeats the same value, so index 0 instead of summing over it.
        lse = torch.log(softmax_sum[..., 0].float().clamp(min=1e-30)) + softmax_max[..., 0].float()
        return output, lse
