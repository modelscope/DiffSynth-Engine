"""
Precision test: MINDIE attention_forward(fused=True) vs F.scaled_dot_product_attention.

Usage (on NPU):
    PYTHONPATH=. python tests/test_precision_mindie_attn.py
"""

import torch
import torch.nn.functional as F


def sdpa_reference(query, key, value, scale):
    """v1 reference: F.scaled_dot_product_attention."""
    return F.scaled_dot_product_attention(
        query, key, value,
        scale=scale,
    )


def mindie_fused(query, key, value, scale):
    """NPU path: mindiesd attention_forward."""
    from mindiesd.layers.flash_attn.attention_forward import attention_forward

    return attention_forward(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
        scale=scale,
        fused=True,
        head_first=False,
    )


def run_test(num_heads, head_size, batch_size, seq_len, kv_len, dtype):
    """Compare NPU fused vs v1 reference for a given config."""
    hidden_size = num_heads * head_size

    # [B, S, H, D] format (head_first=False)
    query = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype, device="npu")
    key = torch.randn(batch_size, kv_len, num_heads, head_size, dtype=dtype, device="npu")
    value = torch.randn(batch_size, kv_len, num_heads, head_size, dtype=dtype, device="npu")

    scale = head_size ** -0.5

    npu_out = mindie_fused(query, key, value, scale)
    ref_out = sdpa_reference(query, key, value, scale)

    abs_diff = (npu_out.float() - ref_out.float()).abs()
    mean_abs = abs_diff.mean().item()
    max_abs = abs_diff.max().item()
    rel_err = (abs_diff / (ref_out.float().abs() + 1e-8)).mean().item()

    return mean_abs, max_abs, rel_err


def main():
    print("=" * 80)
    print("MINDIE Attention Precision Test: attention_forward vs SDPA")
    print("=" * 80)

    configs = [
        # (num_heads, head_size, batch_size, seq_len, kv_len)
        (8, 64, 1, 256, 256),
        (8, 128, 1, 256, 256),
        (24, 128, 1, 256, 256),    # Qwen-Image: 24 heads, head_dim=128
        (24, 128, 1, 512, 512),
        (24, 128, 1, 1024, 1024),
        (24, 128, 1, 4096, 4096),   # multi-image edit long seq
        (24, 128, 2, 256, 256),
        (24, 128, 4, 256, 256),
        # asymmetric kv_len (text + image in joint attention)
        (24, 128, 1, 256, 512),
        (24, 128, 1, 512, 1024),
    ]

    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    print(f"\n{'heads':>6} {'hsz':>5} {'B':>4} {'S':>6} {'kv_len':>6} {'dtype':>12} {'mean_abs':>14} {'max_abs':>14} {'mean_rel':>14}")
    print("-" * 90)

    for num_heads, head_size, batch_size, seq_len, kv_len in configs:
        for dtype in dtypes:
            mean_abs, max_abs, rel_err = run_test(num_heads, head_size, batch_size, seq_len, kv_len, dtype)
            dtype_str = str(dtype).split(".")[-1]
            print(f"{num_heads:>6} {head_size:>5} {batch_size:>4} {seq_len:>6} {kv_len:>6} {dtype_str:>12} {mean_abs:>14.6e} {max_abs:>14.6e} {rel_err:>14.6e}")


if __name__ == "__main__":
    main()
