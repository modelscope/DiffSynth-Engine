"""
FFN torch.compile A/B Benchmark
================================
对比 FFN block 编译 vs 不编译对 NPU 推理性能的影响。

方案:
  A) Baseline: 不启用 compile, 运行 5 步 text-to-image
  B) Compiled: compile_ffn=True, 运行 5 步 text-to-image (排除编译预热步)

输出:
  - results/compile_ffn_results.json
  - 精度对比 (SSIM)
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("[ERROR] torch_npu not available. This benchmark requires NPU.")
    sys.exit(1)

from PIL import Image

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model

# ==================== 配置 ====================
SEED = 42
NUM_INFERENCE_STEPS = 5  # 使用少量步数加速测试
WARMUP_RUNS = 2
TIMED_RUNS = 3
COMPILE_WARMUP_RUNS = 3  # 编译版本需要更多预热（首次编译开销大）
DEVICE = "npu"
ATTN_TYPE = "mindie"
MODEL_DTYPE = torch.bfloat16
WIDTH = 1024
HEIGHT = 1024

# 路径
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results"
RESULT_JSON = OUTPUT_DIR / "compile_ffn_results.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 环境变量
os.environ["USE_MINDIESD_FUSE"] = "true"

# 防止 core dump 占满磁盘
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def make_generator():
    return torch.Generator(device="cpu").manual_seed(SEED)


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """计算两张 PIL 图片之间的 SSIM。"""
    try:
        from skimage.metrics import structural_similarity as ssim
        arr1 = np.array(img1).astype(np.float64)
        arr2 = np.array(img2).astype(np.float64)
        if arr1.shape != arr2.shape:
            return 0.0
        # multichannel SSIM
        return ssim(arr1, arr2, channel_axis=2, data_range=255.0)
    except ImportError:
        # Fallback: simple pixel-level correlation
        arr1 = np.array(img1).astype(np.float64).flatten()
        arr2 = np.array(img2).astype(np.float64).flatten()
        if arr1.shape != arr2.shape:
            return 0.0
        # Normalized correlation as rough approximation
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(arr1, arr2) / (norm1 * norm2))


def run_benchmark(name: str, compile_ffn: bool) -> dict:
    """运行一组 benchmark，返回结果字典。"""
    result = {
        "variant": name,
        "compile_ffn": compile_ffn,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "avg_time_s": None,
        "per_step_avg_ms": None,
        "peak_memory_mb": None,
        "status": "failed",
        "error": None,
        "output_image_path": None,
        "compile_errors": [],
    }

    warmup_runs = COMPILE_WARMUP_RUNS if compile_ffn else WARMUP_RUNS

    print(f"\n{'='*60}")
    print(f"  Variant: {name} (compile_ffn={compile_ffn})")
    print(f"{'='*60}")

    try:
        # 创建 engine
        print(f"  [1/4] Loading model...")
        model_path = fetch_model("Qwen/Qwen-Image")
        config = QwenImagePipelineConfig(
            model_path=model_path,
            device=DEVICE,
            attn_type=ATTN_TYPE,
            model_dtype=MODEL_DTYPE,
            compile_ffn=compile_ffn,
        )
        engine = DiffSynthEngine.from_pretrained(config)
        print(f"  [1/4] Model loaded (compile_ffn={compile_ffn}).")

        generate_kwargs = dict(
            prompt="A painting of a cat in a zen garden",
            negative_prompt="ugly, blurry, low quality",
            true_cfg_scale=4.0,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )

        # Warmup
        print(f"  [2/4] Warmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            torch.npu.empty_cache()
            try:
                _ = engine.generate(**generate_kwargs, generator=make_generator())
                print(f"         warmup {i+1}/{warmup_runs} done")
            except Exception as e:
                error_msg = f"Warmup run {i+1} failed: {e}"
                print(f"         [WARN] {error_msg}")
                result["compile_errors"].append(error_msg)
                if compile_ffn and i == 0:
                    # First compile attempt failed - try fallback modes
                    raise

        # Timed runs
        print(f"  [3/4] Timed runs ({TIMED_RUNS} runs)...")
        times = []
        output = None
        for i in range(TIMED_RUNS):
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()

            torch.npu.synchronize()
            t0 = time.perf_counter()
            output = engine.generate(**generate_kwargs, generator=make_generator())
            torch.npu.synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            times.append(elapsed)
            peak_mem = torch.npu.max_memory_allocated() / (1024 * 1024)
            print(f"         run {i+1}/{TIMED_RUNS}: {elapsed:.3f}s, peak_mem={peak_mem:.0f}MB")

        avg_time = sum(times) / len(times)
        per_step_avg_ms = (avg_time / NUM_INFERENCE_STEPS) * 1000
        peak_memory_mb = torch.npu.max_memory_allocated() / (1024 * 1024)

        result["avg_time_s"] = round(avg_time, 4)
        result["per_step_avg_ms"] = round(per_step_avg_ms, 2)
        result["peak_memory_mb"] = round(peak_memory_mb, 1)
        result["status"] = "success"

        # 保存输出图片
        print(f"  [4/4] Saving output...")
        img = output.images[0]
        suffix = "compiled" if compile_ffn else "baseline"
        img_path = OUTPUT_DIR / f"compile_ffn_{suffix}.png"
        img.save(str(img_path))
        result["output_image_path"] = str(img_path)

        # 清理
        engine.shutdown()
        del engine
        torch.npu.empty_cache()

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["compile_errors"].append(traceback.format_exc())
        print(f"  [ERROR] {e}")
        traceback.print_exc()

    print(f"  Result: {result['status']} | avg={result['avg_time_s']}s | "
          f"per_step={result['per_step_avg_ms']}ms | peak_mem={result['peak_memory_mb']}MB")
    return result


def try_compile_with_fallbacks() -> dict:
    """尝试多种 compile 配置，如果默认方式失败则尝试 fallback。"""
    # 1. 首先尝试默认 compile (MindIE backend if available)
    print("\n" + "="*60)
    print("  Attempting compile_ffn with default backend...")
    print("="*60)
    result = run_benchmark("compiled_default", compile_ffn=True)

    if result["status"] == "success":
        return result

    # 2. 尝试 reduce-overhead mode
    print("\n" + "="*60)
    print("  Default compile failed. Trying mode='reduce-overhead'...")
    print("="*60)
    try:
        # Patch compile_kwargs temporarily
        from diffsynth_engine.utils import platform as plat_mod
        original_fn = plat_mod.get_compile_kwargs

        def patched_kwargs():
            kwargs = original_fn()
            kwargs["mode"] = "reduce-overhead"
            return kwargs

        plat_mod.get_compile_kwargs = patched_kwargs
        result = run_benchmark("compiled_reduce_overhead", compile_ffn=True)
        plat_mod.get_compile_kwargs = original_fn

        if result["status"] == "success":
            return result
    except Exception as e:
        print(f"  [ERROR] reduce-overhead attempt failed: {e}")

    # 3. 尝试 fullgraph=False + default backend (no MindIE)
    print("\n" + "="*60)
    print("  Trying fullgraph=False with inductor backend...")
    print("="*60)
    try:
        from diffsynth_engine.utils import platform as plat_mod

        def patched_kwargs_inductor():
            return {"fullgraph": False}

        plat_mod.get_compile_kwargs = patched_kwargs_inductor
        result = run_benchmark("compiled_inductor_nofullgraph", compile_ffn=True)
        plat_mod.get_compile_kwargs = original_fn

        if result["status"] == "success":
            return result
    except Exception as e:
        print(f"  [ERROR] inductor attempt failed: {e}")

    return result


def main():
    print("=" * 60)
    print("  FFN torch.compile A/B Benchmark")
    print(f"  Device: {DEVICE} | Dtype: {MODEL_DTYPE} | Attn: {ATTN_TYPE}")
    print(f"  Steps: {NUM_INFERENCE_STEPS} | Size: {WIDTH}x{HEIGHT}")
    print(f"  Seed: {SEED}")
    print("=" * 60)

    results = {}

    # ==================== A) Baseline (no compile) ====================
    baseline_result = run_benchmark("baseline", compile_ffn=False)
    results["baseline"] = baseline_result

    # ==================== B) Compiled FFN ====================
    compiled_result = try_compile_with_fallbacks()
    results["compiled"] = compiled_result

    # ==================== 精度对比 ====================
    ssim_value = None
    if baseline_result["status"] == "success" and compiled_result["status"] == "success":
        print("\n" + "="*60)
        print("  Computing SSIM between baseline and compiled outputs...")
        print("="*60)
        try:
            img_baseline = Image.open(baseline_result["output_image_path"])
            img_compiled = Image.open(compiled_result["output_image_path"])
            ssim_value = compute_ssim(img_baseline, img_compiled)
            print(f"  SSIM: {ssim_value:.6f}")
            if ssim_value >= 0.95:
                print(f"  [PASS] SSIM >= 0.95 threshold")
            else:
                print(f"  [WARN] SSIM < 0.95 threshold")
        except Exception as e:
            print(f"  [ERROR] SSIM computation failed: {e}")

    # ==================== 性能对比 ====================
    speedup = None
    if (baseline_result["status"] == "success" and compiled_result["status"] == "success"
            and baseline_result["per_step_avg_ms"] and compiled_result["per_step_avg_ms"]):
        speedup = (baseline_result["per_step_avg_ms"] - compiled_result["per_step_avg_ms"]) / baseline_result["per_step_avg_ms"] * 100
        print(f"\n  Performance delta: {speedup:+.2f}% "
              f"({'faster' if speedup > 0 else 'slower'} with compile)")

    # ==================== 汇总 ====================
    summary = {
        "metadata": {
            "device": DEVICE,
            "attn_type": ATTN_TYPE,
            "model_dtype": str(MODEL_DTYPE),
            "seed": SEED,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "resolution": f"{WIDTH}x{HEIGHT}",
            "warmup_runs_baseline": WARMUP_RUNS,
            "warmup_runs_compiled": COMPILE_WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
            "torch_version": torch.__version__,
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
        },
        "baseline": baseline_result,
        "compiled": compiled_result,
        "comparison": {
            "ssim": ssim_value,
            "ssim_pass": ssim_value >= 0.95 if ssim_value is not None else None,
            "speedup_percent": round(speedup, 2) if speedup is not None else None,
            "conclusion": _derive_conclusion(baseline_result, compiled_result, ssim_value, speedup),
        },
    }

    with open(str(RESULT_JSON), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Benchmark complete!")
    print(f"  Results saved to: {RESULT_JSON}")
    print(f"{'='*60}")

    # Final summary table
    print(f"\n{'Variant':<30} {'Status':<10} {'Avg(s)':<10} {'Per Step(ms)':<14} {'Peak Mem(MB)':<14}")
    print("-" * 80)
    for variant_name, r in results.items():
        avg = f"{r.get('avg_time_s', '-')}" if r.get('avg_time_s') else "-"
        step = f"{r.get('per_step_avg_ms', '-')}" if r.get('per_step_avg_ms') else "-"
        mem = f"{r.get('peak_memory_mb', '-')}" if r.get('peak_memory_mb') else "-"
        print(f"{variant_name:<30} {r['status']:<10} {avg:<10} {step:<14} {mem:<14}")

    if ssim_value is not None:
        print(f"\n  SSIM: {ssim_value:.6f} ({'PASS' if ssim_value >= 0.95 else 'FAIL'})")
    if speedup is not None:
        print(f"  Speedup: {speedup:+.2f}%")


def _derive_conclusion(baseline, compiled, ssim, speedup) -> str:
    """根据结果推导结论。"""
    if compiled["status"] != "success":
        errors = compiled.get("compile_errors", [])
        error_summary = errors[0][:200] if errors else compiled.get("error", "unknown error")
        return f"torch.compile failed on NPU FFN blocks: {error_summary}"

    if ssim is not None and ssim < 0.95:
        return f"torch.compile produces inaccurate results (SSIM={ssim:.4f} < 0.95)"

    if speedup is None:
        return "Unable to compute speedup"

    if speedup > 1.0:
        return f"torch.compile FFN provides {speedup:.1f}% speedup with acceptable accuracy"
    elif speedup > -1.0:
        return f"torch.compile FFN has negligible effect ({speedup:+.1f}%)"
    else:
        return f"torch.compile FFN causes {abs(speedup):.1f}% regression - not recommended"


if __name__ == "__main__":
    main()
