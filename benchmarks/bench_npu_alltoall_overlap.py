"""
NPU AllToAll Overlap Tuning - Test different FA_ALLTOALL_OVERLAP values
on 4-card Ulysses configuration.
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

def run_overlap_test(overlap_val):
    """Test a specific FA_ALLTOALL_OVERLAP value on 4-card Ulysses."""
    print(f"\n{'='*60}")
    print(f"  FA_ALLTOALL_OVERLAP = {overlap_val} (4-card Ulysses)")
    print(f"{'='*60}")

    # Set env before importing platform (already imported, but AscendPlatform reads at class level)
    # Need to reload or set before engine creation
    os.environ["FA_ALLTOALL_OVERLAP"] = str(overlap_val)
    os.environ["FA_ALLTOALL_CUT"] = "1"

    # Force reload of platform module to pick up new env
    import diffsynth_engine.platforms.ascend as ascend_mod
    import importlib
    importlib.reload(ascend_mod)

    model_path = fetch_model("Qwen/Qwen-Image")
    config = QwenImagePipelineConfig(
        model_path=model_path, device=DEVICE, attn_type=ATTN_TYPE,
        model_dtype=MODEL_DTYPE, parallelism=4, sp_ulysses_degree=4,
    )
    engine = DiffSynthEngine.from_pretrained(config)
    print(f"  Engine ready (4-way, overlap={overlap_val})")

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

    return {"overlap": overlap_val, "avg_total_ms": round(avg_total, 2),
            "avg_step_ms": round(avg_step, 2), "runs": [round(t, 2) for t in times]}

def main():
    print("=== NPU AllToAll Overlap Tuning (4-card Ulysses) ===")
    single_step = 555.5

    overlap_values = [1, 2, 4, 8]
    results = []

    for ov in overlap_values:
        try:
            r = run_overlap_test(ov)
            r["speedup"] = round(single_step / r["avg_step_ms"], 3) if r["avg_step_ms"] > 0 else 0
            r["efficiency"] = round(r["speedup"] / 4 * 100, 1)
            ideal = single_step / 4
            r["overhead_ms"] = round(r["avg_step_ms"] - ideal, 2)
            r["overhead_pct"] = round(r["overhead_ms"] / r["avg_step_ms"] * 100, 1) if r["avg_step_ms"] > 0 else 0
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] overlap={ov}: {e}")
            results.append({"overlap": ov, "error": str(e)})

    # Summary
    print("\n" + "="*60)
    print("  ALLTOALL OVERLAP TUNING RESULTS (4-card)")
    print("="*60)
    print(f"  Single-card: {single_step} ms/step, Ideal 4-card: {single_step/4:.1f} ms/step")
    print(f"  {'Overlap':<10} {'Step(ms)':<10} {'Spdup':<8} {'Eff%':<8} {'OH%':<8}")
    print("  " + "-"*44)
    for r in results:
        if "error" in r:
            print(f"  {r['overlap']:<10} ERROR: {r['error'][:30]}")
        else:
            print(f"  {r['overlap']:<10} {r['avg_step_ms']:<10.2f} {r['speedup']:<8.3f} {r['efficiency']:<8.1f} {r['overhead_pct']:<8.1f}")

    best = min([r for r in results if "error" not in r], key=lambda x: x["avg_step_ms"], default=None)
    if best:
        print(f"\n  BEST: overlap={best['overlap']} -> {best['avg_step_ms']} ms/step ({best['speedup']}x, {best['efficiency']}% eff)")

    out = RESULT_DIR / "npu_alltoall_overlap_results.json"
    with open(out, "w") as f: json.dump({"single_step_ms": single_step, "num_cards": 4, "results": results}, f, indent=2)
    print(f"  Saved: {out}")

if __name__ == "__main__":
    main()
