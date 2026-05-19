"""
Precision test: torch_npu.npu_fast_gelu vs F.gelu(approximate='tanh').

Usage (on NPU):
    PYTHONPATH=. python tests/test_precision_fast_gelu.py
"""

import torch
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:
    torch_npu = None


def run_test(hidden_size, batch_size, seq_len, dtype):
    """Compare NPU fused vs v1 reference for a given config."""
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="npu")

    # NPU fused path
    npu_out = torch_npu.npu_fast_gelu(x)

    # v1 reference path
    ref_out = F.gelu(x, approximate="tanh")

    # Metrics
    abs_diff = (npu_out.float() - ref_out.float()).abs()
    mean_abs = abs_diff.mean().item()
    max_abs = abs_diff.max().item()
    rel_err = (abs_diff / (ref_out.float().abs() + 1e-8)).mean().item()

    return mean_abs, max_abs, rel_err


def main():
    print("=" * 80)
    print("FastGELU Precision Test: npu_fast_gelu vs F.gelu(approximate='tanh')")
    print("=" * 80)

    configs = [
        # (hidden_size, batch_size, seq_len)
        (64, 1, 256),
        (128, 1, 256),
        (3584, 1, 256),
        (3584, 1, 1024),
        (3584, 1, 4096),
        (3584, 2, 256),
        (3584, 4, 256),
        # FastGELU inner_dim = dim * 4
        (3584 * 4, 1, 256),
        (3584 * 4, 1, 4096),
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
