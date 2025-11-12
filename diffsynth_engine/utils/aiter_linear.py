import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from aiter import hipb_mm, hipb_create_extension, per_tensor_quant_hip
from aiter.tuned_gemm import tgemm
from aiter.ops.shuffle import shuffle_weight
from diffsynth_engine.utils.platform import DTYPE_FP8
from contextlib import contextmanager


@lru_cache(maxsize=1)
def init_hipblas():
    hipb_create_extension()


@contextmanager
def use_swizzle_hipblaslt(swizzle=True, use_fp8_linear=True, use_scale_for_fp8=False):
    if not swizzle:
        yield
        return

    # Preserve original F.linear
    _original_linear = F.linear
    
    def optimized_linear(input, weight, bias=None, otype=torch.bfloat16, 
                        scaleA=None, scaleB=None, device="cuda"):

        input_flat = input.reshape(-1, input.shape[-1])
        
        init_hipblas()
    
        weight_preshuffle = shuffle_weight(weight.contiguous(), layout=(16, 16), use_int4=False).to(device)
        output_flat = hipb_mm(
            input_flat,
            weight_preshuffle.t(),
            bias=bias,
            solution_index=-1,
            out_dtype=otype,
            scaleA=scaleA,
            scaleB=scaleB,
            scaleOut=None,
            bpreshuffle=True
        )
        
        # Reshape output to match input dimensions
        new_shape = input.shape[:-1] + (weight.shape[0],)
        output = output_flat.view(new_shape)
        return output
    
    
    def optimized_linear_fp8(input, weight, bias=None, otype=torch.bfloat16,
                        scaleA=None, scaleB=None, device="cuda"):
        
        input_flat = input.reshape(-1, input.shape[-1])

        if use_scale_for_fp8:

            input_flat, a_scale = per_tensor_quant_hip(input_flat, quant_dtype=DTYPE_FP8)
            weight = weight.to(DTYPE_FP8)
    
            init_hipblas()
    
            weight_preshuffle = shuffle_weight(weight.contiguous(), layout=(16, 16)).to(device)
            output_flat = hipb_mm(
                input_flat,
                weight_preshuffle.t(),
                bias=bias,
                solution_index=-1,
                out_dtype=otype,
                scaleA=a_scale,
                scaleB=scaleB,
                scaleOut=None,
                bpreshuffle=True
            )

        else:
            input_flat = input_flat.to(DTYPE_FP8)
            weight = weight.to(DTYPE_FP8)

            init_hipblas()

            weight_preshuffle = shuffle_weight(weight.contiguous(), layout=(16, 16)).to(device)
            output_flat = hipb_mm(
                input_flat,
                weight_preshuffle.t(),
                bias=bias,
                solution_index=-1,
                out_dtype=otype,
                scaleA=scaleA,
                scaleB=scaleB,
                scaleOut=None,
                bpreshuffle=True
            )

    
        # Reshape output to match input dimensions
        new_shape = input.shape[:-1] + (weight.shape[0],)
        output = output_flat.view(new_shape)
        return output
    
    if use_fp8_linear:
        F.linear = optimized_linear_fp8
    else:
        F.linear = optimized_linear

    yield
    F.linear = _original_linear


