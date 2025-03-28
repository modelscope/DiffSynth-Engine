# DiffSynth-Engine

[![PyPI](https://img.shields.io/pypi/v/DiffSynth-Engine)](https://pypi.org/project/DiffSynth-Engine/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Engine.svg)](https://github.com/modelscope/DiffSynth-Engine/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Engine.svg)](https://github.com/modelscope/DiffSynth-Engine/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Engine.svg)](https://GitHub.com/modelscope/DiffSynth-Engine/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Engine)](https://GitHub.com/modelscope/DiffSynth-Engine/commit/)

DiffSynth-Engine is a high-performance engine geared towards buidling efficient inference pipelines for diffusion models.

**Key Features:**

- **Thoughtfully-Designed Implementation:** We carefully re-implemented key components in Diffusion pipelines, such as sampler and scheduler, without introducing external dependencies on libraries like k-diffusion, ldm, or sgm.

- **Extensive Model Support:** Compatible with popular formats (e.g., CivitAI) of base models and LoRA models , catering to diverse use cases.

- **Versatile Resource Management:** Comprehensive support for varous model quantization (e.g., FP8, INT8) 
and offloading strategies, enabling loading of larger diffusion models (e.g., Flux.1 Dev) on limited hardware budget of GPU memory.

- **Optimized Performance:** Carefully-crafted inference pipeline to achieve fast generation across various hardware environments.

- **Cross-Platform Support:** Runnable on Windows, macOS (Apple Silicon), and Linux, ensuring a smooth experience across different operating systems.

## Quick Start
### Requirements

- Python 3.10+
- NVIDIA GPU with compute capability 8.6+ (e.g., RTX 50 Series, RTX 40 Series, RTX 30 Series. Please see [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) for more details about your GPUs.) or Apple Silicon M-series.

### Installation

Install released version (from PyPI):
```shell
pip3 install diffsynth-engine
```

Install from source:
```shell
git clone https://github.com/modelscope/diffsynth-engine.git && cd diffsynth-engine
pip3 install -e .
```

### Usage
Text to image
```python
from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0')
image = pipe(prompt="a cat")
image.save("image.png")
```
Text to image with LoRA
```python
from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
lora_path = fetch_model("DonRat/MAJICFLUS_SuperChinesestyleheongsam", path="麦橘超国风旗袍.safetensors")

pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0')
pipe.load_lora(path=lora_path, scale=1.0)
image = pipe(prompt="a girl, qipao")
image.save("image.png")
```

For more details, please refer to our tutorials ([English](./docs/tutorial.md), [中文](./docs/tutorial_zh.md)).

## Showcase

<img src="assets/showcase.jpeg" />

## Contact

If you have any questions or feedback, please scan the QR code below, or send email to muse@alibaba-inc.com.

<div style="display: flex; justify-content: space-between;">
    <img src="assets/dingtalk.png" alt="dingtalk" width="400" />
</div>

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Citation

If you use this codebase, or otherwise found our work helpful, please cite:

```bibtex
@misc{diffsynth-engine2025,
      title={DiffSynth-Engine: a high-performance diffusion inference engine},
      author={Zhipeng Di, Guoxuan Zhu, Zhongjie Duan, Zihao Chu, Yingda Chen, Weiyi Lu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/modelscope/diffsynth-engine}},
}
```
