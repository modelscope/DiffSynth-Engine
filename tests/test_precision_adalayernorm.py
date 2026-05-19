"""
Precision test: layernorm_scale_shift fused vs nn.LayerNorm + manual modulation.

Usage (on NPU):
    PYTHONPATH=. python tests/test_precision_adalayernorm.py
"""

import torch
import torch.nn as nn
from diffsynth_engine.layers.norm import AdaLayerNorm


def adalayernorm_reference(hidden_states, scale, shift, layernorm):
    """v1 reference: nn.LayerNorm + _modulate (scale, shift are [B, dim])."""
    normed = layernorm(hidden_states)
    # _modulate: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    return normed * (1 + scale) + shift


def run_test(hidden_size, batch_size, seq_len, dtype, scale_shift_data):
    """Compare NPU fused vs v1 reference for a given config.

    Args:
        scale_shift_data: one of "2d", "3d_unsqueeze", "3d_full"
            - "2d": [B, dim] (the actual call pattern in transformer block)
            - "3d_unsqueeze": [B, 1, dim] (pre-broadcast)
            - "3d_full": [B, S, dim] (per-token different)
    """
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="npu")

    layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps).to("npu")

    # Generate scale/shift based on data pattern
    if scale_shift_data == "2d":
        scale = torch.randn(batch_size, hidden_size, dtype=dtype, device="npu") * 0.1
        shift = torch.randn(batch_size, hidden_size, dtype=dtype, device="npu") * 0.1
    elif scale_shift_data == "3d_unsqueeze":
        scale = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device="npu") * 0.1
        shift = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device="npu") * 0.1
    else:  # "3d_full"
        scale = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="npu") * 0.1
        shift = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="npu") * 0.1

    # NPU fused path
    ada_norm = AdaLayerNorm(layernorm)
    npu_out = ada_norm(x, scale, shift)

    # v1 reference path
    ref_out = adalayernorm_reference(x, scale, shift, layernorm)

    # Metrics
    abs_diff = (npu_out.float() - ref_out.float()).abs()
    mean_abs = abs_diff.mean().item()
    max_abs = abs_diff.max().item()
    rel_err = (abs_diff / (ref_out.float().abs() + 1e-8)).mean().item()

    return mean_abs, max_abs, rel_err


def main():
    print("=" * 80)
    print("AdaLayerNorm Precision Test: layernorm_scale_shift vs LayerNorm + modulate")
    print("=" * 80)

    configs = [
        # (hidden_size, batch_size, seq_len)
        (64, 1, 256),
        (128, 1, 256),
        (3584, 1, 256),     # Qwen-Image, txt2img
        (3584, 1, 512),
        (3584, 1, 1024),
        (3584, 1, 4096),    # multi-image edit long seq
        (3584, 2, 256),
        (3584, 4, 256),
        (128, 4, 1024),
    ]

    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    scale_patterns = ["2d", "3d_unsqueeze"]

    for pattern in scale_patterns:
        print(f"\n--- scale/shift pattern: {pattern} ---")
        print(f"{'dim':>8} {'B':>4} {'S':>6} {'dtype':>12} {'mean_abs':>14} {'max_abs':>14} {'mean_rel':>14}")
        print("-" * 80)

        for hidden_size, batch_size, seq_len in configs:
            for dtype in dtypes:
                mean_abs, max_abs, rel_err = run_test(hidden_size, batch_size, seq_len, dtype, pattern)
                dtype_str = str(dtype).split(".")[-1]
                print(f"{hidden_size:>8} {batch_size:>4} {seq_len:>6} {dtype_str:>12} {mean_abs:>14.6e} {max_abs:>14.6e} {rel_err:>14.6e}")


if __name__ == "__main__":
    main()
