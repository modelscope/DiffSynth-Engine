"""
Precision test: torch_npu.npu_rms_norm vs DiffusersRMSNorm.

Usage (on NPU):
    PYTHONPATH=. python tests/test_precision_rmsnorm.py
"""

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffsynth_engine.layers.norm import RMSNorm


def rmsnorm_reference(hidden_states, weight, eps):
    """v1 reference: diffusers RMSNorm implementation."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def run_test(hidden_size, batch_size, seq_len, dtype):
    """Compare NPU fused vs v1 reference for a given config."""
    eps = 1e-6
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="npu")

    # Use same weight for both paths
    weight = torch.randn(hidden_size, dtype=dtype, device="npu")

    # NPU fused path — move to NPU so weight is on same device as input
    norm = RMSNorm(hidden_size, eps).to("npu")
    with torch.no_grad():
        norm.weight.copy_(weight)
    npu_out = norm(x)

    # v1 reference path
    ref = DiffusersRMSNorm(hidden_size, eps).to("npu")
    with torch.no_grad():
        ref.weight.copy_(weight)
    ref_out = ref(x)

    # Metrics
    abs_diff = (npu_out.float() - ref_out.float()).abs()
    mean_abs = abs_diff.mean().item()
    max_abs = abs_diff.max().item()
    rel_err = (abs_diff / (ref_out.float().abs() + 1e-8)).mean().item()

    return mean_abs, max_abs, rel_err


def main():
    print("=" * 80)
    print("RMSNorm Precision Test: npu_rms_norm vs DiffusersRMSNorm")
    print("=" * 80)

    configs = [
        # (hidden_size, batch_size, seq_len)
        (64, 1, 256),
        (128, 1, 256),
        (3584, 1, 256),    # Qwen-Image inner_dim, txt2img
        (3584, 1, 512),
        (3584, 1, 1024),
        (3584, 1, 4096),    # multi-image edit long seq
        (3584, 2, 256),
        (3584, 4, 256),
        (128, 4, 1024),     # attention head_dim * num_heads
    ]

    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    print(f"\n{'dim':>8} {'B':>4} {'S':>6} {'dtype':>12} {'mean_abs':>14} {'max_abs':>14} {'mean_rel':>14}")
    print("-" * 80)

    for hidden_size, batch_size, seq_len in configs:
        for dtype in dtypes:
            mean_abs, max_abs, rel_err = run_test(hidden_size, batch_size, seq_len, dtype)
            dtype_str = str(dtype).split(".")[-1]
            print(f"{hidden_size:>8} {batch_size:>4} {seq_len:>6} {dtype_str:>12} {mean_abs:>14.6e} {max_abs:>14.6e} {rel_err:>14.6e}")


if __name__ == "__main__":
    main()
