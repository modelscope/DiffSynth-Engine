#!/usr/bin/env python3
import argparse
import time

import torch

from diffsynth_engine.models.qwen_image.qwen_image_cuda_ext import (
    modulate_forward as modulate_forward_cuda,
    modulate_indexed_forward as modulate_indexed_forward_cuda,
)


def modulate_pytorch(x: torch.Tensor, mod_params: torch.Tensor):
    shift, scale, gate = mod_params.chunk(3, dim=-1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)


def modulate_indexed_pytorch(x: torch.Tensor, mod_params: torch.Tensor, index: torch.Tensor):
    shift, scale, gate = mod_params.chunk(3, dim=-1)
    actual_batch = shift.size(0) // 2
    shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
    scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
    gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
    shift_result = torch.where(index == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
    scale_result = torch.where(index == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
    gate_result = torch.where(index == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
    return x * (1 + scale_result) + shift_result, gate_result


def benchmark(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Qwen modulate CUDA kernel vs PyTorch.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--check-iters", type=int, default=3)
    parser.add_argument("--indexed", action="store_true", help="Benchmark indexed modulation path.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_str(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    torch.manual_seed(args.seed)
    dtype = dtype_from_str(args.dtype)
    device = "cuda"

    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=dtype).contiguous()

    if args.indexed:
        mod_params = torch.randn(args.batch * 2, args.dim * 3, device=device, dtype=dtype).contiguous()
        # Shared index pattern across batch, matching model behavior.
        index = torch.randint(0, 2, (1, args.seq, 1), device=device, dtype=torch.int32).contiguous()
        ref_fn = lambda: modulate_indexed_pytorch(x, mod_params, index)
        cuda_fn = lambda: modulate_indexed_forward_cuda(x, mod_params, index)
        mode_name = "indexed"
    else:
        mod_params = torch.randn(args.batch, args.dim * 3, device=device, dtype=dtype).contiguous()
        ref_fn = lambda: modulate_pytorch(x, mod_params)
        cuda_fn = lambda: modulate_forward_cuda(x, mod_params)
        mode_name = "plain"

    out_cuda = cuda_fn()
    if out_cuda is None:
        raise RuntimeError("CUDA extension unavailable. Set QWEN_IMAGE_CUDA_EXT_WARN=1 for build errors.")

    atol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    max_abs = 0.0
    max_rel = 0.0
    for _ in range(args.check_iters):
        y_ref, g_ref = ref_fn()
        y_cuda, g_cuda = cuda_fn()
        if y_cuda is None or g_cuda is None:
            raise RuntimeError("CUDA extension became unavailable during correctness check.")
        for a, b in ((y_ref, y_cuda), (g_ref, g_cuda)):
            diff = (a - b).abs()
            denom = a.abs().clamp_min(1e-6)
            max_abs = max(max_abs, diff.max().item())
            max_rel = max(max_rel, (diff / denom).max().item())
            if not torch.allclose(a, b, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"Correctness check failed: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
                    f"atol={atol}, rtol={rtol}"
                )

    ms_ref = benchmark(ref_fn, args.warmup, args.iters)
    ms_cuda = benchmark(cuda_fn, args.warmup, args.iters)
    speedup = ms_ref / ms_cuda

    print("=== Qwen Modulate Comparison ===")
    print(f"mode: {mode_name}")
    print(f"shape: B={args.batch}, S={args.seq}, D={args.dim}, dtype={args.dtype}")
    print(f"correctness: PASS (max_abs={max_abs:.6e}, max_rel={max_rel:.6e})")
    print(f"pytorch: {ms_ref:.4f} ms/iter")
    print(f"cuda   : {ms_cuda:.4f} ms/iter")
    print(f"speedup: {speedup:.3f}x ({(1.0 - ms_cuda / ms_ref) * 100.0:+.2f}% vs pytorch)")


if __name__ == "__main__":
    main()
