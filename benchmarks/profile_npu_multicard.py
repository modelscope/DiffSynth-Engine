"""
NPU Multi-Card Profiling - uses DiffSynthEngine internal parallelism
Usage: python3 benchmarks/profile_npu_multicard.py --num-cards 4
Note: callback_on_step_end is NOT picklable for multi-card, using total/steps.
"""
import argparse, json, os, resource, sys, time
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

def profile(num_cards):
    print(f"=== NPU {num_cards}-Card Profiling (Ulysses SP) ===")
    print(f"Steps: {NUM_STEPS}, Warmup: {WARMUP}, Timed: {TIMED}")
    model_path = fetch_model("Qwen/Qwen-Image")
    config = QwenImagePipelineConfig(model_path=model_path, device=DEVICE, attn_type=ATTN_TYPE,
        model_dtype=MODEL_DTYPE, parallelism=num_cards, sp_ulysses_degree=num_cards)
    engine = DiffSynthEngine.from_pretrained(config)
    print(f"Engine loaded with {num_cards}-way parallelism")

    # Warmup
    for i in range(WARMUP):
        _ = engine.generate(**GEN_KWARGS, generator=make_gen())
        print(f"  warmup {i+1}/{WARMUP}")

    # Timed runs (no callback - not picklable for multiprocessing)
    times = []
    for i in range(TIMED):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = engine.generate(**GEN_KWARGS, generator=make_gen())
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  run {i+1}/{TIMED}: {elapsed:.1f} ms")
    avg_total = sum(times) / len(times)
    avg_step = avg_total / NUM_STEPS
    print(f"  avg: {avg_total:.1f} ms total, {avg_step:.1f} ms/step")

    # Scaling analysis
    single_step = 555.5  # from single-card profiling
    single_attn = 263.44
    speedup = single_step / avg_step if avg_step > 0 else 0
    efficiency = speedup / num_cards * 100
    ideal_step = single_step / num_cards
    overhead = avg_step - ideal_step
    overhead_pct = overhead / avg_step * 100 if avg_step > 0 else 0

    print("\n=== SCALING ANALYSIS ===")
    print(f"  Single-card step: {single_step:.1f} ms")
    print(f"  {num_cards}-card step:   {avg_step:.1f} ms")
    print(f"  Ideal step:       {ideal_step:.1f} ms (linear {num_cards}x)")
    print(f"  Speedup:          {speedup:.2f}x (ideal {num_cards}x)")
    print(f"  Efficiency:       {efficiency:.1f}%")
    print(f"  Overhead:         {overhead:.1f} ms ({overhead_pct:.1f}% of step)")
    print(f"  Comm bottleneck:  {overhead_pct > 15}")

    # Optimization analysis
    optimizations = []
    if overhead_pct > 15:
        optimizations.append({
            "type": "high_comm_overhead",
            "overhead_pct": round(overhead_pct, 1),
            "fix": "Improve AllToAll overlap (AscendLongContextAttention fa_alltoall_overlap parameter)"
        })
    if efficiency < 60:
        optimizations.append({
            "type": "low_efficiency",
            "efficiency": round(efficiency, 1),
            "fix": "Reduce SP degree or use hybrid Ulysses+Ring"
        })
    # Check if attention dominates (comm overhead in attention AllToAll)
    attn_pct_of_step = single_attn / single_step * 100
    comm_in_attn_estimate = overhead * (attn_pct_of_step / 100)
    if comm_in_attn_estimate > 30:
        optimizations.append({
            "type": "alltoall_in_attention_dominant",
            "estimated_comm_ms": round(comm_in_attn_estimate, 1),
            "fix": "Increase fa_alltoall_overlap chunks / enable comm-compute stream overlap"
        })

    if optimizations:
        print("\n=== OPTIMIZATION POINTS ===")
        for i, o in enumerate(optimizations, 1):
            print(f"  [{i}] {o['type']}: {o['fix']}")

    results = {
        "metadata": {"num_cards": num_cards, "steps": NUM_STEPS, "torch": torch.__version__},
        "timing": {"avg_total_ms": round(avg_total, 2), "avg_step_ms": round(avg_step, 2),
                   "run_times_ms": [round(t, 2) for t in times]},
        "scaling": {"single_step_ms": single_step, "multi_step_ms": round(avg_step, 2),
                    "ideal_step_ms": round(ideal_step, 2), "speedup": round(speedup, 3),
                    "ideal_speedup": num_cards, "efficiency_pct": round(efficiency, 1),
                    "overhead_ms": round(overhead, 2), "overhead_pct": round(overhead_pct, 1),
                    "is_bottleneck": bool(overhead_pct > 15)},
        "optimizations": optimizations,
        "gate": {"do_comm_optimize": bool(overhead_pct > 15),
                 "reason": f"Overhead {overhead_pct:.1f}% {'>' if overhead_pct>15 else '<='} 15%"}
    }
    out = RESULT_DIR / f"profiling_multicard_{num_cards}.json"
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    engine.shutdown(); del engine; torch.npu.empty_cache()
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-cards", type=int, required=True, choices=[2, 4, 8])
    args = p.parse_args()
    profile(args.num_cards)
