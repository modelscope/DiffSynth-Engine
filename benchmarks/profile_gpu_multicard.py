"""
GPU Multi-Card Profiling on 134 (8x H20)
=========================================
Measures scaling efficiency across parallelism configurations.
Note: callback_on_step_end cannot be used with multi-card (not picklable),
      so per-step time is derived from total_time / num_steps.

Usage:
    TMPDIR=/data1/tmp_bench QWEN_IMAGE_PATH=/path/to/model PYTHONPATH=/tmp/pylibs:$PWD \
    /opt/conda310/bin/python benchmarks/profile_gpu_multicard.py
"""
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model

SEED = 42
DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
NUM_STEPS = 5
WARMUP = 2
TIMED = 3

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

GEN_KWARGS = dict(
    prompt="A painting of a cat in a zen garden",
    negative_prompt="ugly, blurry, low quality",
    true_cfg_scale=4.0,
    width=1024,
    height=1024,
    num_inference_steps=NUM_STEPS,
)

def make_gen():
    return torch.Generator(device="cpu").manual_seed(SEED)

def get_gpu_info():
    info = {
        "gpu_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if info["gpu_count"] > 0:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info

def profile_config(config_name, num_cards, attn_type, sp_ulysses_degree=None, sp_ring_degree=None, use_cfg_parallel=False):
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"  Cards: {num_cards} | Attn: {attn_type} | Ulysses: {sp_ulysses_degree} | Ring: {sp_ring_degree} | CFG: {use_cfg_parallel}")
    print(f"{'='*60}")

    model_path = os.environ.get("QWEN_IMAGE_PATH", None)
    if model_path is None:
        model_path = fetch_model("Qwen/Qwen-Image", local_files_only=True)

    kwargs = dict(
        model_path=model_path, device=DEVICE, attn_type=attn_type,
        model_dtype=MODEL_DTYPE, parallelism=num_cards,
        use_cfg_parallel=use_cfg_parallel,
    )
    if sp_ulysses_degree is not None:
        kwargs["sp_ulysses_degree"] = sp_ulysses_degree
    if sp_ring_degree is not None:
        kwargs["sp_ring_degree"] = sp_ring_degree

    try:
        config = QwenImagePipelineConfig(**kwargs)
    except Exception as e:
        print(f"  [SKIP] Config invalid: {e}")
        return {"status": "skipped", "config_name": config_name, "error": str(e)}

    try:
        engine = DiffSynthEngine.from_pretrained(config)
    except Exception as e:
        print(f"  [ERROR] Engine init: {e}")
        return {"status": "error", "config_name": config_name, "error": str(e)}

    print(f"  Engine loaded ({num_cards}-way)")

    # Warmup
    for i in range(WARMUP):
        try:
            _ = engine.generate(**GEN_KWARGS, generator=make_gen())
            print(f"    warmup {i+1}/{WARMUP}")
        except Exception as e:
            print(f"  [ERROR] Warmup: {e}")
            engine.shutdown(); del engine; gc.collect(); torch.cuda.empty_cache()
            return {"status": "error", "config_name": config_name, "error": f"warmup: {e}"}

    # Timed runs - total pipeline only (no callback for multi-card)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(TIMED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = engine.generate(**GEN_KWARGS, generator=make_gen())
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"    run {i+1}/{TIMED}: {elapsed:.1f} ms")

    avg_total = sum(times) / len(times)
    avg_step = avg_total / NUM_STEPS
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"  => total={avg_total:.1f}ms, step~={avg_step:.2f}ms, mem={peak_mem:.0f}MB")

    engine.shutdown(); del engine; gc.collect(); torch.cuda.empty_cache()
    time.sleep(2)

    return {
        "status": "success",
        "config_name": config_name,
        "num_cards": num_cards,
        "attn_type": attn_type,
        "sp_ulysses_degree": sp_ulysses_degree,
        "sp_ring_degree": sp_ring_degree,
        "use_cfg_parallel": use_cfg_parallel,
        "timing": {
            "avg_total_ms": round(avg_total, 2),
            "avg_step_ms": round(avg_step, 2),
            "run_times_ms": [round(t, 2) for t in times],
        },
        "peak_memory_mb": round(peak_mem, 1),
    }


def compute_analysis(results):
    baseline = None
    for r in results:
        if r.get("status") == "success" and r.get("num_cards") == 1:
            baseline = r
            break
    if not baseline:
        return {"error": "No baseline"}

    base_step = baseline["timing"]["avg_step_ms"]
    base_total = baseline["timing"]["avg_total_ms"]

    analysis = {"baseline": {"step_ms": base_step, "total_ms": base_total}, "scaling": [], "optimizations": []}

    for r in results:
        if r.get("status") != "success" or r.get("num_cards") == 1:
            continue
        n = r["num_cards"]
        step_ms = r["timing"]["avg_step_ms"]
        total_ms = r["timing"]["avg_total_ms"]
        speedup = base_step / step_ms if step_ms > 0 else 0
        total_speedup = base_total / total_ms if total_ms > 0 else 0
        eff = speedup / n * 100
        ideal = base_step / n
        overhead = step_ms - ideal
        overhead_pct = overhead / step_ms * 100 if step_ms > 0 else 0

        entry = {
            "config": r["config_name"], "cards": n,
            "step_ms": round(step_ms, 2), "total_ms": round(total_ms, 2),
            "speedup": round(speedup, 3), "total_speedup": round(total_speedup, 3),
            "efficiency": round(eff, 1),
            "ideal_ms": round(ideal, 2), "overhead_ms": round(overhead, 2),
            "overhead_pct": round(overhead_pct, 1),
        }
        analysis["scaling"].append(entry)

        if overhead_pct > 15:
            analysis["optimizations"].append({
                "config": r["config_name"], "type": "high_comm_overhead",
                "overhead_pct": round(overhead_pct, 1), "overhead_ms": round(overhead, 2),
                "fix": "AllToAll overlap / reduce SP degree / try Ring attention for better overlap",
            })
        if eff < 60:
            analysis["optimizations"].append({
                "config": r["config_name"], "type": "low_efficiency",
                "efficiency": round(eff, 1),
                "fix": "Reduce parallelism / use CFG parallel / increase workload size",
            })

    analysis["scaling"].sort(key=lambda x: x["speedup"], reverse=True)
    return analysis


def main():
    gpu_info = get_gpu_info()
    print("=" * 70)
    print(f"  GPU Multi-Card Profiling: {gpu_info.get('gpu_name','N/A')} x {gpu_info['gpu_count']}")
    print(f"  Torch {gpu_info['torch_version']} | CUDA {gpu_info['cuda_version']}")
    print(f"  Steps={NUM_STEPS} Warmup={WARMUP} Timed={TIMED}")
    print("=" * 70)

    configs = [
        # (name, cards, attn, ulysses, ring, cfg_parallel)
        ("1card_fa2", 1, "fa2", None, None, False),
        ("2card_ulysses_fa2", 2, "fa2", 2, 1, False),
        ("4card_ulysses_fa2", 4, "fa2", 4, 1, False),
        ("8card_ulysses_fa2", 8, "fa2", 8, 1, False),
        ("2card_ring_fa2", 2, "fa2", 1, 2, False),
        ("4card_ring_fa2", 4, "fa2", 1, 4, False),
        ("8card_ring_fa2", 8, "fa2", 1, 8, False),
        ("4card_hybrid_u2r2", 4, "fa2", 2, 2, False),
        ("8card_hybrid_u4r2", 8, "fa2", 4, 2, False),
        ("8card_hybrid_u2r4", 8, "fa2", 2, 4, False),
        ("2card_cfg", 2, "fa2", 1, 1, True),
        ("4card_cfg_u2", 4, "fa2", 2, 1, True),
        ("8card_cfg_u4", 8, "fa2", 4, 1, True),
    ]

    results = []
    for name, n, attn, u, r, cfg in configs:
        if n > gpu_info["gpu_count"]:
            print(f"\n  [SKIP] {name}: need {n}, have {gpu_info['gpu_count']}")
            continue
        res = profile_config(name, n, attn, u, r, cfg)
        results.append(res)

    analysis = compute_analysis(results)

    # Print summary table
    print("\n" + "=" * 70)
    print("  SCALING SUMMARY")
    print("=" * 70)
    if "baseline" in analysis:
        print(f"  Baseline (1 card): {analysis['baseline']['step_ms']:.2f} ms/step, {analysis['baseline']['total_ms']:.1f} ms total")
        print(f"  {'Config':<25} {'N':<4} {'Step':<9} {'Spdup':<7} {'Eff%':<7} {'OH%':<7} {'Total':<10}")
        print("  " + "-" * 70)
        for s in analysis.get("scaling", []):
            print(f"  {s['config']:<25} {s['cards']:<4} {s['step_ms']:<9.2f} {s['speedup']:<7.3f} {s['efficiency']:<7.1f} {s['overhead_pct']:<7.1f} {s['total_ms']:<10.1f}")

    if analysis.get("optimizations"):
        print(f"\n  OPTIMIZATION POINTS ({len(analysis['optimizations'])} found):")
        for i, o in enumerate(analysis["optimizations"], 1):
            print(f"  [{i}] {o['config']}: {o['type']} => {o['fix']}")

    output = {
        "metadata": {"timestamp": datetime.now().isoformat(), "hardware": gpu_info,
                     "config": {"seed": SEED, "steps": NUM_STEPS, "warmup": WARMUP, "timed": TIMED}},
        "raw_results": results,
        "analysis": analysis,
    }
    out_path = RESULT_DIR / "gpu_multicard_profiling.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
