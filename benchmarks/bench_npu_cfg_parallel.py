"""
NPU CFG Parallel Benchmark - Test use_cfg_parallel optimization
Compares: pure Ulysses vs CFG+Ulysses configurations
"""
import gc, json, os, resource, sys, time
from pathlib import Path
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model

SEED = 42; DEVICE = "npu"; ATTN_TYPE = "mindie"; MODEL_DTYPE = torch.bfloat16
NUM_STEPS = 5; WARMUP = 2; TIMED = 3
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["USE_MINDIESD_FUSE"] = "true"
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
GEN_KWARGS = dict(prompt="A painting of a cat in a zen garden", negative_prompt="ugly, blurry",
    true_cfg_scale=4.0, width=1024, height=1024, num_inference_steps=NUM_STEPS)

def make_gen(): return torch.Generator(device="cpu").manual_seed(SEED)

def run_config(name, parallelism, use_cfg_parallel, sp_ulysses_degree):
    print(f"\n{'='*60}")
    print(f"  {name}: parallelism={parallelism}, cfg={use_cfg_parallel}, ulysses={sp_ulysses_degree}")
    print(f"{'='*60}")

    model_path = fetch_model("Qwen/Qwen-Image")
    config = QwenImagePipelineConfig(
        model_path=model_path, device=DEVICE, attn_type=ATTN_TYPE,
        model_dtype=MODEL_DTYPE, parallelism=parallelism,
        use_cfg_parallel=use_cfg_parallel, sp_ulysses_degree=sp_ulysses_degree,
    )
    engine = DiffSynthEngine.from_pretrained(config)
    print(f"  Engine ready ({parallelism}-way, cfg={use_cfg_parallel})")

    for i in range(WARMUP):
        _ = engine.generate(**GEN_KWARGS, generator=make_gen())
        print(f"    warmup {i+1}/{WARMUP}")

    times = []
    for i in range(TIMED):
        torch.npu.synchronize(); t0 = time.perf_counter()
        _ = engine.generate(**GEN_KWARGS, generator=make_gen())
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"    run {i+1}/{TIMED}: {elapsed:.1f} ms")

    avg_total = sum(times) / len(times)
    avg_step = avg_total / NUM_STEPS
    print(f"  => total={avg_total:.1f}ms, step={avg_step:.1f}ms")

    engine.shutdown(); del engine; gc.collect(); torch.npu.empty_cache()
    time.sleep(3)

    return {"name": name, "parallelism": parallelism, "cfg": use_cfg_parallel,
            "ulysses": sp_ulysses_degree, "avg_total_ms": round(avg_total, 2),
            "avg_step_ms": round(avg_step, 2), "runs": [round(t, 2) for t in times]}

def main():
    print("=== NPU CFG Parallel Optimization Test ===")
    single_step = 555.5  # baseline from single-card profiling

    configs = [
        # (name, parallelism, use_cfg_parallel, sp_ulysses_degree)
        ("4card_pure_ulysses", 4, False, 4),      # baseline: already measured
        ("4card_cfg_u2", 4, True, 2),              # P0 optimization!
        ("8card_pure_ulysses", 8, False, 8),       # baseline: already measured
        ("8card_cfg_u4", 8, True, 4),              # P0 optimization!
    ]

    results = []
    for name, par, cfg, uly in configs:
        try:
            r = run_config(name, par, cfg, uly)
            r["speedup"] = round(single_step / r["avg_step_ms"], 3) if r["avg_step_ms"] > 0 else 0
            r["efficiency"] = round(r["speedup"] / par * 100, 1)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results.append({"name": name, "error": str(e)})

    # Summary
    print("\n" + "="*60)
    print("  CFG PARALLEL RESULTS")
    print("="*60)
    print(f"  Single-card baseline: {single_step} ms/step")
    print(f"  {'Config':<25} {'Cards':<6} {'Step(ms)':<10} {'Spdup':<8} {'Eff%':<8}")
    print("  " + "-"*57)
    for r in results:
        if "error" in r: 
            print(f"  {r['name']:<25} ERROR: {r['error'][:30]}")
        else:
            print(f"  {r['name']:<25} {r['parallelism']:<6} {r['avg_step_ms']:<10.2f} {r['speedup']:<8.3f} {r['efficiency']:<8.1f}")

    # Save
    out = RESULT_DIR / "npu_cfg_parallel_results.json"
    with open(out, "w") as f: json.dump({"single_step_ms": single_step, "results": results}, f, indent=2)
    print(f"\n  Saved: {out}")

if __name__ == "__main__":
    main()
