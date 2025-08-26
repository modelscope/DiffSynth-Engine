# Multi-Model Performance Benchmark

This directory contains a performance benchmark script for various image generation pipelines with different optimization configurations.

## Features

- Benchmark different models:
  - FLUX
  - Qwen-Image
- Benchmark different optimization modes:
  - Basic mode (default)
  - FP8 linear optimization
  - Torch compile optimization
  - FP8 + compile combination
  - CPU offloading
- Generate detailed CUDA timeline traces using torch.profiler
- Compare performance across different configurations
- Save sample images from each benchmark run

## Usage

```bash
# Basic FLUX benchmark
python model_perf_benchmark.py --model flux --mode basic

# Qwen-Image with FP8 optimization
python model_perf_benchmark.py --model qwen_image --mode fp8

# FLUX with Torch compile optimization
python model_perf_benchmark.py --model flux --mode compile

# Qwen-Image with all optimizations and profiling
python model_perf_benchmark.py --model qwen_image --mode all --trace-file

# FLUX profiling with auto-generated filename (includes config and GPU info)
python model_perf_benchmark.py --model flux --mode fp8 --trace-file

# Qwen-Image with custom prompt and profiling
python model_perf_benchmark.py --model qwen_image --mode fp8 --prompt "a cyberpunk cityscape" --trace-file

# Benchmark with specific loop count
python model_perf_benchmark.py --model flux --mode compile --num-runs 10
```

For Qwen-Image models, you may need to specify additional paths:
```bash
# Qwen-Image with custom model paths
python model_perf_benchmark.py --model qwen_image --model-path /path/to/model --encoder-path /path/to/encoder --vae-path /path/to/vae --mode basic
```

## Output

The script will generate:
- Performance timing results
- Sample images from each run
- Chrome trace files for detailed profiling (if `--trace-file` is specified)

You can view the trace files in https://ui.perfetto.dev/