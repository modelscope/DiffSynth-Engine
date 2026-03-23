# DiffSynth-Engine

[![PyPI](https://img.shields.io/pypi/v/DiffSynth-Engine)](https://pypi.org/project/DiffSynth-Engine/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Engine.svg)](https://github.com/modelscope/DiffSynth-Engine/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Engine.svg)](https://github.com/modelscope/DiffSynth-Engine/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Engine.svg)](https://GitHub.com/modelscope/DiffSynth-Engine/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Engine)](https://GitHub.com/modelscope/DiffSynth-Engine/commit/)

DiffSynth-Engine is a light-weighted and high-performance engine for diffusion model inference.

**Key Features:**

- **More Features:** To be filled.

## News

- **[v0.7.0](https://github.com/modelscope/DiffSynth-Engine/releases/tag/v0.7.0)** | **January 19, 2026**: ![Image](assets/tongyi.svg) Supports [Qwen-Image-Edit-2511](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2511), [Qwen-Image-2512](https://modelscope.cn/models/Qwen/Qwen-Image-2512), [Z-Image-Turbo](https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) and Z-Image-Omni-Base
- **[v0.6.0](https://github.com/modelscope/DiffSynth-Engine/releases/tag/v0.6.0)** | **September 9, 2025**: ![Image](assets/tongyi.svg) Supports [Wan2.2-S2V](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B), a video generation model designed for audio-driven cinematic video generation
- **[v0.5.0](https://github.com/modelscope/DiffSynth-Engine/releases/tag/v0.5.0)** | **August 27, 2025**: ![Image](assets/tongyi.svg) Supports [Qwen-Image-Edit](https://modelscope.cn/models/Qwen/Qwen-Image-Edit), the image editing version of Qwen-Image, enabling semantic/appearance visual editing, and precise text editing
- **[v0.4.1](https://github.com/modelscope/DiffSynth-Engine/releases/tag/v0.4.1)** | **August 4, 2025**: ![Image](assets/tongyi.svg) Supports [Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image), an image generation model excels at complex text rendering and creating images in a wide range of artistic styles
- **[v0.4.0](https://github.com/modelscope/DiffSynth-Engine/releases/tag/v0.4.0)** | **August 1, 2025**:
  - ![Image](assets/tongyi.svg) Supports [Wan2.2](https://modelscope.cn/collections/tongyiwanxiang-22--shipinshengcheng-2bb5b1adef2840) video generation model
  - ⚠️[**Breaking Change**] Improved `from_pretrained` method pipeline initialization

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
```

Text to image with LoRA
```python
```

For more details, please refer to our tutorials ([English](./docs/tutorial.md), [中文](./docs/tutorial_zh.md)).

## Showcase

<img src="assets/showcase.jpeg" />

## Contact

If you have any questions or feedback, please scan the QR code below, or send email to muse@alibaba-inc.com.

<div style="display: flex; justify-content: space-between;">
    <img src="assets/dingtalk.png" alt="dingtalk" width="400" />
</div>

## Contributing
We welcome contributions to DiffSynth-Engine. After Install from source, we recommand developers install this project using following command to setup the development environment.
```bash
pip install -e '.[dev]'
pre-commit install
```
TODO: Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

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

## Acknowledgment

We inspired by the design and leveraging code from following projects: [diffusers](https://github.com/huggingface/diffusers), [SGLang Diffusion](https://github.com/sgl-project/sglang), [vLLM-Omni](https://github.com/vllm-project/vllm-omni) and [LightX2V](https://github.com/ModelTC/LightX2V).