"""W8A8 FP8 quantization for Ascend NPU linear layers.

Why a separate path is needed
-----------------------------
On CUDA, FP8 matmul goes through ``torch._scaled_mm``. torch_npu does not
implement it, so on Ascend the cube unit has to be driven through
``npu_quant_matmul`` instead, with the operands quantized beforehand:

    activations -> npu_dynamic_quant(dst_type=FP8_DST_TYPE)   per-token, online
    weights     -> per-channel float8_e4m3fn, transposed to [K, N] once
    matmul      -> npu_quant_matmul(fp8, fp8 -> bf16), scaled by both operands

Three parameter rules were learned the hard way and are easy to get wrong:

1. ``dst_type=292``. The acl enum ``ACL_FLOAT8_E4M3FN`` is 36, and torch_npu
   offsets its internal dtype ids by 256, so 36 + 256 = 292. Omitting it
   silently quantizes to int8.
2. The weight scale must be a **1-D float32** tensor. Passing a 2-D scale makes
   the operator demand an int64 ``scale`` instead, with an error message that
   points nowhere near the real cause.
3. The activation scale must be float32 as well.

Design notes
------------
The per-channel scale is stored as an attribute on the weight parameter rather
than in a dict keyed by ``id(weight)``. A fused LoRA rewrites ``weight.data`` in
place, and object ids get recycled -- both make an id-keyed table go stale and
produce silently wrong results.

Quantization is opt-in: a linear layer is quantized only after
``enable_fp8_linear()`` has visited it, and the forward path checks both the
scale attribute and the weight dtype, so a layer whose weight was restored to
bf16 (fused LoRA, GGUF load) falls back to the dense path automatically.
"""

import torch
import torch.nn as nn

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

# aclDataType ACL_FLOAT8_E4M3FN = 36; torch_npu offsets dtype ids by 256.
FP8_DST_TYPE = 256 + 36
FP8_MAX = 448.0

# Layers matched by leaf name keep bf16.
#   proj_out      the output projection feeds the latent directly, so its
#                 quantization error is not smoothed out by any later block
#   img_mod/txt_mod  the adaLN modulation MLPs (6 * dim), tiny and sensitive
DEFAULT_SKIP = ("proj_out", "img_mod", "txt_mod")

# Tensor-parallel linears are the only ones quantized by default. They carry
# essentially all of the block FLOPs, while the remaining nn.Linear layers sit
# at the edges of the stack (img_in / txt_in / modulation / proj_out). There is
# also a mechanical reason: nn.Linear.forward is PyTorch's own, so it knows
# nothing about the transposed [K, N] weight layout and would fail on shapes.
_PARALLEL_LINEAR_NAMES = ("ColumnParallelLinear", "RowParallelLinear")

_QUANT_DTYPE = torch.float8_e4m3fn


def is_quantized(weight: torch.Tensor) -> bool:
    """Whether this weight has been through ``quantize_linear_weight``.

    Both conditions matter: a fused LoRA can rewrite ``weight.data`` back to
    bf16 while leaving ``npu_quant_scale`` in place.
    """
    return getattr(weight, "npu_quant_scale", None) is not None and weight.dtype == _QUANT_DTYPE


def quantize_linear_weight(module: nn.Linear) -> None:
    """Quantize one layer's weight to per-channel float8_e4m3fn, stored as [K, N]."""
    weight = module.weight.data
    if not torch.is_floating_point(weight):
        return

    # Per-output-channel range. clamp keeps an all-zero row from dividing by zero.
    w_max = weight.abs().amax(dim=1).clamp(min=1e-8).float()
    scale = w_max / FP8_MAX
    quantized = (weight.float() / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    # [N, K] -> [K, N]: npu_quant_matmul wants [K, N]. Transposing once here
    # avoids a large copy on every forward.
    module.weight.data = quantized.t().contiguous()
    module.weight.npu_quant_scale = scale


def npu_quant_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W8A8 online-quantized linear: activations per-token, weights per-channel."""
    import torch_npu

    scale_b = weight.npu_quant_scale
    origin_dtype = input.dtype
    origin_shape = input.shape
    x2d = input.reshape(-1, origin_shape[-1])

    if weight.dtype == torch.float8_e4m3fn:
        # dst_type is what selects e4m3; without it the activation is quantized to int8
        x_quant, x_scale = torch_npu.npu_dynamic_quant(x2d, dst_type=FP8_DST_TYPE)
    else:
        x_quant, x_scale = torch_npu.npu_dynamic_quant(x2d)

    out = torch_npu.npu_quant_matmul(
        x_quant,
        weight,
        scale=scale_b,
        pertoken_scale=x_scale,
        output_dtype=origin_dtype,
    )
    if bias is not None:
        out = out + bias
    return out.reshape(origin_shape[:-1] + (out.shape[-1],))


def enable_fp8_linear(
    module: nn.Module,
    skip: tuple[str, ...] = DEFAULT_SKIP,
    include_plain_linear: bool = False,
) -> int:
    """Quantize every eligible linear layer under ``module``. Returns the count.

    ``include_plain_linear`` additionally quantizes plain ``nn.Linear`` layers.
    That only works if the caller also patches their forward; it is off by
    default because ``nn.Linear.forward`` would otherwise receive a ``[K, N]``
    weight and compute the wrong thing.
    """
    count = 0

    def _walk(mod: nn.Module, prefix: str) -> None:
        nonlocal count
        eligible = type(mod).__name__ in _PARALLEL_LINEAR_NAMES or (include_plain_linear and type(mod) is nn.Linear)
        if eligible and torch.is_floating_point(mod.weight.data):
            leaf = prefix.rsplit(".", 1)[-1]
            # match the leaf name only, so e.g. "proj_out" does not also catch
            # an unrelated "xx_proj_output"
            if leaf not in skip:
                quantize_linear_weight(mod)
                count += 1
        for name, child in mod.named_children():
            _walk(child, f"{prefix}.{name}" if prefix else name)

    _walk(module, "")
    logger.info(f"fp8: quantized {count} linear layers to W8A8 fp8")
    return count
