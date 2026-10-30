import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from diffsynth_engine.utils.platform import DTYPE_FP8, IS_NPU


def enable_fp8_autocast(module: nn.Module, compute_dtype: torch.dtype = torch.bfloat16, use_fp8_linear: bool = False):
    if len(list(module.children())) == 0:
        if len(list(module.parameters())) > 0:
            add_fp8_autocast_hook(module, compute_dtype)
        return
    if len(list(module.parameters(recurse=False))) > 0:
        add_fp8_autocast_hook(module, compute_dtype)
    for submodule in module.children():
        if isinstance(submodule, nn.Linear) and use_fp8_linear:
            continue

        enable_fp8_autocast(submodule, compute_dtype, use_fp8_linear)


def add_fp8_autocast_hook(module: nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
    if IS_NPU:
        # NPU keeps the quantized weight as a plain tensor with a scale attribute and
        # casts inside the custom linear, so no round trip to the compute dtype is needed.
        return

    def _fp8_autocast_pre_hook(module: nn.Module, input_):
        for name, param in module.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                param.data = param.data.to(compute_dtype)
        new_inputs = []
        for x in input_:
            if isinstance(x, torch.Tensor) and x.dtype in [torch.float8_e4m3fn, torch.float16, torch.bfloat16]:
                new_inputs.append(x.to(compute_dtype))
            else:
                new_inputs.append(x)
        return tuple(new_inputs)

    def _fp8_autocast_hook(module: nn.Module, input_, output_):
        for name, param in module.named_parameters():
            if param.dtype == compute_dtype:
                param.data = param.data.to(torch.float8_e4m3fn)

    if getattr(module, "_fp8_autocast_enabled", False):
        return
    module.register_forward_pre_hook(_fp8_autocast_pre_hook)
    module.register_forward_hook(_fp8_autocast_hook)
    setattr(module, "_fp8_autocast_enabled", True)


# ---------------------------------------------------------------------------
# Ascend NPU: W8A8 online quantization
# ---------------------------------------------------------------------------
# torch_npu does not implement torch._scaled_mm, so the FP8 path above cannot run on
# NPU. The cube unit is quantized instead, through npu_quant_matmul:
#
#   fp8 (default)
#     activations -> npu_dynamic_quant(dst_type=FP8_DST_TYPE)  per-token scale, online
#     weights     -> per-channel float8_e4m3fn, transposed to [K, N] once at enable time
#     matmul      -> npu_quant_matmul   (fp8 x fp8 -> bf16, scaled by both operands)
#
#   int8 (NPU_QUANT_DTYPE=int8)
#     activations -> npu_dynamic_quant                          per-token scale, online
#     weights     -> per-channel int8, [K, N]
#     matmul      -> npu_quant_matmul   (int8 x int8 -> int32 accumulate)
#
# Both variants keep `scale` on the weight parameter itself: a dict keyed by id(weight)
# goes stale as soon as a fused LoRA rewrites weight.data, and object ids are recycled.
#
# Layers whose name contains one of NPU_QUANT_SKIP keep bf16 weights: the output
# projections inject quantization error most directly into the latent.
_NPU_QUANT_SKIP = tuple(
    name.strip() for name in os.environ.get("NPU_QUANT_SKIP", "proj_out,to_out").split(",") if name.strip()
)
_NPU_QUANT_DTYPE = os.environ.get("NPU_QUANT_DTYPE", "fp8").strip().lower()

# aclDataType ACL_FLOAT8_E4M3FN is 36, and torch_npu offsets its internal dtype ids by 256.
FP8_DST_TYPE = 256 + 36
FP8_MAX = 448.0

_npu_origin_linear = None

# aclDataType ACL_FLOAT8_E4M3FN is 36, and torch_npu offsets its internal dtype ids by 256.
FP8_DST_TYPE = 256 + 36
FP8_MAX = 448.0

_npu_origin_linear = None


def _npu_quantize_linear_weight(module: nn.Linear) -> None:
    """Quantize one Linear weight to [K, N] and record its per-channel scale."""
    weight = module.weight.data
    if not torch.is_floating_point(weight):
        return

    w_max = weight.abs().amax(dim=1).clamp(min=1e-8).float()
    if _NPU_QUANT_DTYPE == "int8":
        scale = w_max / 127.0
        quantized = torch.round(weight.float() / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    else:
        scale = w_max / FP8_MAX
        quantized = (weight.float() / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    # [N, K] -> [K, N]: npu_quant_matmul expects [K, N], and transposing here avoids
    # a large copy on every forward.
    module.weight.data = quantized.t().contiguous()
    module.weight.npu_quant_scale = scale


def _npu_enable_quant_linear(module: nn.Module, prefix: str = "") -> None:
    if isinstance(module, nn.Linear) and torch.is_floating_point(module.weight.data):
        # match the leaf module name only, so e.g. "to_out" does not also hit
        # an unrelated "xxx_to_output"
        if prefix.rsplit(".", 1)[-1] not in _NPU_QUANT_SKIP:
            _npu_quantize_linear_weight(module)
    for name, child in module.named_children():
        _npu_enable_quant_linear(child, f"{prefix}.{name}" if prefix else name)


def npu_quant_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W8A8 online-quantized linear layer (activations per-token, weights per-channel)."""
    scale_b = getattr(weight, "npu_quant_scale", None)
    if scale_b is None or weight.dtype not in (torch.int8, torch.float8_e4m3fn):
        # not quantized: kept in bf16 by NPU_QUANT_SKIP, an int GGUF weight, or a
        # fused LoRA that has rewritten the weight back to bf16 after quantization
        return _npu_origin_linear(input, weight, bias)

    import torch_npu

    origin_dtype = input.dtype
    origin_shape = input.shape
    x2d = input.reshape(-1, origin_shape[-1])

    if weight.dtype == torch.float8_e4m3fn:
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


def enable_fp8_linear(module: nn.Module):
    if IS_NPU:
        _npu_enable_quant_linear(module)
    else:
        _enable_fp8_linear(module)
    setattr(module, "fp8_linear_enabled", True)


def _enable_fp8_linear(module: nn.Module):
    if isinstance(module, nn.Linear) and torch.is_floating_point(module.weight.data):
        # avoid conversion for int weights like GGUF
        module.weight.data = module.weight.data.to(DTYPE_FP8)
    for submodule in module.children():
        _enable_fp8_linear(submodule)


@contextmanager
def fp8_inference(enabled=True):
    if not enabled:
        yield
        return

    global _npu_origin_linear

    origin_linear = F.linear

    def fp8_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = input.device
        origin_dtype = input.dtype
        origin_shape = input.shape
        input = input.reshape(-1, origin_shape[-1])

        x_max = torch.max(torch.abs(input), dim=-1, keepdim=True).values
        fp8_max = 448.0
        # For float8_e4m3fnuz, the maximum representable value is half of that of e4m3fn.
        # To avoid overflow and ensure numerical compatibility during FP8 computation,
        # we scale down the input by 2.0 in advance.
        # This scaling will be compensated later during the final result scaling.
        if DTYPE_FP8 == torch.float8_e4m3fnuz:
            fp8_max = fp8_max / 2.0
        scale_a = torch.clamp(x_max / fp8_max, min=1.0).float().to(device=device)
        scale_b = torch.ones((weight.shape[0], 1)).float().to(device=device)
        input = input / scale_a
        input = input.to(DTYPE_FP8)
        weight = weight.to(DTYPE_FP8)

        result = torch._scaled_mm(
            input,
            weight.T,
            scale_a=scale_a,
            scale_b=scale_b.T,
            bias=bias,
            out_dtype=origin_dtype,
        )
        new_shape = origin_shape[:-1] + result.shape[-1:]
        result = result.reshape(new_shape)
        return result

    if IS_NPU:
        _npu_origin_linear = origin_linear
        F.linear = npu_quant_linear
        try:
            yield
        finally:
            F.linear = origin_linear
        return

    F.linear = fp8_linear
    try:
        yield
    finally:
        F.linear = origin_linear
