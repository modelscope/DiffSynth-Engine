#!/usr/bin/env python3
"""
Multi-Model Performance Benchmark Script

This script benchmarks various image generation pipelines with different optimization configurations
and provides performance profiling with detailed CUDA timeline traces.

Supported models:
- FLUX
- Qwen-Image

Usage:
    python model_perf_benchmark.py --model flux --mode fp8 --trace-file trace.json
    python model_perf_benchmark.py --model qwen_image --mode basic --trace-file trace.json
    python model_perf_benchmark.py --model flux --mode all --trace-file
"""

import argparse
import time
import torch
import re
from torch.profiler import profile, record_function, ProfilerActivity
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List
from PIL import Image


class BaseModelBenchmark(ABC):
    """Base class for model benchmarking."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def create_pipeline(self, mode: str):
        """Create pipeline with specified optimization mode."""
        pass

    @abstractmethod
    def generate_sample(self, pipe: Any, prompt: str, **kwargs) -> Image.Image:
        """Generate a sample image using the pipeline."""
        pass

    def benchmark_pipeline(
        self, pipe: Any, prompt: str, num_runs: int = 3, warmup_runs: int = 1, **kwargs
    ) -> Tuple[float, float, float, Image.Image]:
        """Benchmark the pipeline with timing measurements."""
        # Warmup runs
        for _ in range(warmup_runs):
            _ = self.generate_sample(pipe, prompt, **kwargs)

        # Timed runs
        times = []

        for i in range(num_runs):
            start_time = time.time()
            image = self.generate_sample(pipe, prompt, **kwargs)
            end_time = time.time()

            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        return avg_time, min_time, max_time, image

    def profile_pipeline(
        self, pipe: Any, prompt: str, trace_file: Optional[str] = None, **kwargs
    ) -> Tuple[Image.Image, Any]:
        """Profile the pipeline with torch.profiler and save timeline trace."""
        # Configure profiler with all activities and detailed profiling
        profiler_args = {
            "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            "record_shapes": True,
            "profile_memory": False,
            "with_stack": True,
            "with_flops": True,
            "with_modules": True,
            "experimental_config": torch._C._profiler._ExperimentalConfig(verbose=True),
        }

        # First run (warmup)
        _ = self.generate_sample(pipe, prompt, num_inference_steps=30, **kwargs)

        # Profiled run - only 5 steps for smaller trace file
        with profile(**profiler_args) as prof:
            with record_function("model_inference"):
                image = self.generate_sample(pipe, prompt, num_inference_steps=5, **kwargs)

        if trace_file:
            prof.export_chrome_trace(trace_file)
            print(f"Chrome trace saved to {trace_file}")

        return image, prof


class FluxBenchmark(BaseModelBenchmark):
    """FLUX model benchmark implementation."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        from diffsynth_engine import FluxImagePipeline, FluxPipelineConfig

        self.pipeline_class = FluxImagePipeline
        self.config_class = FluxPipelineConfig

    def create_pipeline(self, mode: str):
        """Create FLUX pipeline with specified optimization mode."""
        config = None

        if mode == "fp8":
            config = self.config_class(
                model_path=self.model_path,
                device="cuda",
                model_dtype=torch.bfloat16,
                use_fp8_linear=True,
            )
        elif mode == "compile":
            config = self.config_class(
                model_path=self.model_path,
                device="cuda",
                model_dtype=torch.bfloat16,
                use_torch_compile=True,
            )
        elif mode == "fp8_compile":
            config = self.config_class(
                model_path=self.model_path,
                device="cuda",
                model_dtype=torch.bfloat16,
                use_fp8_linear=True,
                use_torch_compile=True,
            )
        elif mode == "offload":
            config = self.config_class(
                model_path=self.model_path,
                device="cuda",
                model_dtype=torch.bfloat16,
                offload_mode="sequential_cpu_offload",
            )
        else:  # basic mode
            config = self.config_class.basic_config(
                model_path=self.model_path,
                device="cuda",
            )

        return self.pipeline_class.from_pretrained(config)

    def generate_sample(self, pipe: Any, prompt: str, **kwargs) -> Image.Image:
        """Generate a sample image using the FLUX pipeline."""
        return pipe(prompt=prompt, **kwargs)


class QwenImageBenchmark(BaseModelBenchmark):
    """Qwen-Image model benchmark implementation."""

    def __init__(self, model_path: str, encoder_path: str, vae_path: str):
        super().__init__(model_path)
        from diffsynth_engine import QwenImagePipeline, QwenImagePipelineConfig

        self.pipeline_class = QwenImagePipeline
        self.config_class = QwenImagePipelineConfig
        self.encoder_path = encoder_path
        self.vae_path = vae_path

    def create_pipeline(self, mode: str):
        """Create Qwen-Image pipeline with specified optimization mode."""
        config = None

        if mode == "fp8":
            config = self.config_class(
                model_path=self.model_path,
                encoder_path=self.encoder_path,
                vae_path=self.vae_path,
                device="cuda",
                model_dtype=torch.bfloat16,
                encoder_dtype=torch.bfloat16,
                vae_dtype=torch.float32,
                use_fp8_linear=True,
            )
        elif mode == "compile":
            config = self.config_class(
                model_path=self.model_path,
                encoder_path=self.encoder_path,
                vae_path=self.vae_path,
                device="cuda",
                model_dtype=torch.bfloat16,
                encoder_dtype=torch.bfloat16,
                vae_dtype=torch.float32,
                use_torch_compile=True,
            )
        elif mode == "fp8_compile":
            config = self.config_class(
                model_path=self.model_path,
                encoder_path=self.encoder_path,
                vae_path=self.vae_path,
                device="cuda",
                model_dtype=torch.bfloat16,
                encoder_dtype=torch.bfloat16,
                vae_dtype=torch.float32,
                use_fp8_linear=True,
                use_torch_compile=True,
            )
        elif mode == "offload":
            config = self.config_class(
                model_path=self.model_path,
                encoder_path=self.encoder_path,
                vae_path=self.vae_path,
                device="cuda",
                model_dtype=torch.bfloat16,
                encoder_dtype=torch.bfloat16,
                vae_dtype=torch.float32,
                offload_mode="sequential_cpu_offload",
            )
        else:  # basic mode
            config = self.config_class.basic_config(
                model_path=self.model_path,
                encoder_path=self.encoder_path,
                vae_path=self.vae_path,
                device="cuda",
            )

        return self.pipeline_class.from_pretrained(config)

    def generate_sample(self, pipe: Any, prompt: str, **kwargs) -> Image.Image:
        """Generate a sample image using the Qwen-Image pipeline."""
        return pipe(prompt=prompt, **kwargs)


def get_gpu_info():
    """Get GPU information for trace file naming."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        # Clean up the GPU name for file naming
        gpu_name = re.sub(r"[^\w\-_\. ]", "_", gpu_name)
        gpu_name = re.sub(r"\s+", "_", gpu_name)
        return gpu_name
    return "cpu"


def generate_trace_filename(model: str, mode: str, config: Any):
    """Generate auto trace filename with optimization config and GPU info."""
    gpu_info = get_gpu_info()

    # Build config string
    config_parts = [model, mode]

    # Check for common optimization flags
    if hasattr(config, "use_fp8_linear") and config.use_fp8_linear:
        config_parts.append("fp8")

    if hasattr(config, "use_torch_compile") and config.use_torch_compile:
        config_parts.append("compile")

    if hasattr(config, "offload_mode") and config.offload_mode:
        config_parts.append(config.offload_mode.replace("_", ""))

    config_str = "_".join(config_parts)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    return f"{config_str}_{gpu_info}_{timestamp}.json"


def get_benchmark_instance(
    model: str, model_path: str, encoder_path: str = None, vae_path: str = None
) -> BaseModelBenchmark:
    """Get the appropriate benchmark instance based on the model type."""
    if model == "flux":
        return FluxBenchmark(model_path)
    elif model == "qwen_image":
        if not encoder_path or not vae_path:
            raise ValueError("Qwen-Image model requires encoder_path and vae_path")
        return QwenImageBenchmark(model_path, encoder_path, vae_path)
    else:
        raise ValueError(f"Unsupported model: {model}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Performance Benchmark")
    parser.add_argument("--model", choices=["flux", "qwen_image"], default="flux", help="Model type to benchmark")
    parser.add_argument(
        "--mode",
        choices=["basic", "fp8", "compile", "fp8_compile", "offload", "all"],
        default="basic",
        help="Optimization mode",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        nargs="?",
        const="",  # This allows --trace-file without a value
        help="Path to save Chrome trace file (optional). If no value provided, auto-generated name will be used.",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to model file (if not provided, will fetch default model)"
    )
    parser.add_argument(
        "--encoder-path", type=str, default=None, help="Path to encoder file (required for Qwen-Image model)"
    )
    parser.add_argument("--vae-path", type=str, default=None, help="Path to VAE file (required for Qwen-Image model)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful landscape with mountains and a lake",
        help="Prompt for image generation",
    )
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for benchmarking")

    args = parser.parse_args()

    # Get model paths
    if args.model_path is None:
        if args.model == "flux":
            print("Fetching default FLUX model...")
            from diffsynth_engine import fetch_model

            model_path: str | List[str] = fetch_model("muse/flux-with-vae", path="flux1-dev-with-vae.safetensors")
        elif args.model == "qwen_image":
            print("Fetching default Qwen-Image model...")
            from diffsynth_engine import fetch_model

            model_path = fetch_model(model_uri="MusePublic/Qwen-image", revision="v1", path="transformer/*.safetensors")
            if args.encoder_path is None:
                args.encoder_path = fetch_model(
                    "MusePublic/Qwen-image", revision="v1", path="text_encoder/*.safetensors"
                )
            if args.vae_path is None:
                args.vae_path = fetch_model("MusePublic/Qwen-image", revision="v1", path="vae/*.safetensors")
        else:
            raise ValueError(f"Unsupported model: {args.model}")
    else:
        model_path = args.model_path

    print(f"Using model: {model_path}")
    if args.encoder_path:
        print(f"Using encoder: {args.encoder_path}")
    if args.vae_path:
        print(f"Using VAE: {args.vae_path}")
    print(f"Using prompt: {args.prompt}")

    # Get benchmark instance
    try:
        benchmark = get_benchmark_instance(args.model, model_path, args.encoder_path, args.vae_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Modes to benchmark
    modes = ["basic", "fp8", "compile", "fp8_compile", "offload"] if args.mode == "all" else [args.mode]

    results = {}

    for mode in modes:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {args.model.upper()} with {mode.upper()} mode")
        print(f"{'=' * 50}")

        try:
            # Create pipeline
            print(f"Creating pipeline with {mode} configuration...")
            # config = None
            pipe = benchmark.create_pipeline(mode)

            if args.trace_file is not None:
                # Profile with trace
                if args.trace_file:  # Use provided filename
                    trace_file = (
                        f"{args.trace_file.replace('.json', '')}_{args.model}_{mode}.json"
                        if args.mode == "all"
                        else args.trace_file
                    )
                else:  # Auto-generate filename or use empty string for auto
                    # We need to create a temporary pipeline to get the config for filename generation
                    trace_file = generate_trace_filename(args.model, mode, pipe.config)

                # Print profiling configuration
                print(f"Profiling pipeline and saving trace to {trace_file}...")

                image, prof = benchmark.profile_pipeline(pipe, args.prompt, trace_file)

                # Print profiling summary
                print("\nProfiling Summary:")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            else:
                # Just benchmark
                print("Benchmarking pipeline...")
                avg_time, min_time, max_time, image = benchmark.benchmark_pipeline(pipe, args.prompt, args.num_runs)

                print(f"\nResults for {mode.upper()} mode:")
                print(f"  Average time: {avg_time:.2f}s")
                print(f"  Min time: {min_time:.2f}s")
                print(f"  Max time: {max_time:.2f}s")

                results[mode] = {
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                }

            # Save sample image
            output_file = f"{args.model}_benchmark_{mode}.png"
            image.save(output_file)
            print(f"Sample image saved to {output_file}")

            # Clean up
            del pipe

        except Exception as e:
            print(f"Error in {mode} mode: {e}")
            if args.mode != "all":
                raise

    # Print summary if multiple modes
    if len(modes) > 1 and not args.trace_file:
        print(f"\n{'=' * 50}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 50}")
        print(f"{'Mode':<15} {'Avg Time (s)':<15} {'Min Time (s)':<15} {'Max Time (s)':<15}")
        print("-" * 60)
        for mode in modes:
            if mode in results:
                r = results[mode]
                print(f"{mode:<15} {r['avg_time']:<15.2f} {r['min_time']:<15.2f} {r['max_time']:<15.2f}")


if __name__ == "__main__":
    main()
