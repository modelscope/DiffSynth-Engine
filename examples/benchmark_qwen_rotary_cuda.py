#!/usr/bin/env python3
import argparse
import time

import torch

from diffsynth_engine.models.qwen_image.qwen_image_cuda_ext import rotary_emb_forward as rotary_emb_forward_cuda


def rotary_emb_pytorch(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis.unsqueeze(1)).flatten(3)
    return x_out.type_as(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen rotary CUDA kernel against PyTorch implementation.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--dim", type=int, default=128, help="Head dim (must be even).")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_str(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def benchmark(fn, x: torch.Tensor, freqs: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn(x, freqs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(x, freqs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main():
    args = parse_args()
    if args.dim % 2 != 0:
        raise ValueError("--dim must be even")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = dtype_from_str(args.dtype)

    x = torch.randn(args.batch, args.seq, args.heads, args.dim, device=device, dtype=dtype).contiguous()
    phase = torch.randn(args.seq, args.dim // 2, device=device, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(phase), phase).contiguous()

    y_cuda = rotary_emb_forward_cuda(x, freqs)
    if y_cuda is None:
        raise RuntimeError(
            "CUDA extension failed to load/compile. "
            "Set QWEN_IMAGE_CUDA_EXT_WARN=1 to see full build errors."
        )

    atol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    max_abs = 0.0
    max_rel = 0.0
    for _ in range(args.check_iters):
        x_check = torch.randn_like(x)
        y_pt = rotary_emb_pytorch(x_check, freqs)
        y_cuda = rotary_emb_forward_cuda(x_check, freqs)
        if y_cuda is None:
            raise RuntimeError("CUDA extension became unavailable during correctness check.")
        diff = (y_cuda - y_pt).abs()
        denom = y_pt.abs().clamp_min(1e-6)
        max_abs = max(max_abs, diff.max().item())
        max_rel = max(max_rel, (diff / denom).max().item())
        if not torch.allclose(y_cuda, y_pt, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Correctness check failed: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
                f"atol={atol}, rtol={rtol}"
            )

    ms_pt = benchmark(rotary_emb_pytorch, x, freqs, args.warmup, args.iters)
    ms_cuda = benchmark(rotary_emb_forward_cuda, x, freqs, args.warmup, args.iters)
    speedup = ms_pt / ms_cuda

    print("=== Qwen Rotary Comparison ===")
    print(f"shape: B={args.batch}, S={args.seq}, H={args.heads}, D={args.dim}, dtype={args.dtype}")
    print(f"correctness: PASS (max_abs={max_abs:.6e}, max_rel={max_rel:.6e})")
    print(f"pytorch: {ms_pt:.4f} ms/iter")
    print(f"cuda   : {ms_cuda:.4f} ms/iter")
    print(f"speedup: {speedup:.3f}x ({(1.0 - ms_cuda / ms_pt) * 100.0:+.2f}% vs pytorch)")


if __name__ == "__main__":
    main()
