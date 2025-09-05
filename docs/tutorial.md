# DiffSynth-Engine User Guide

## Installation

Before using DiffSynth-Engine, please ensure your device meets the following requirements:

* NVIDIA GPU with CUDA Compute Capability 8.6+ (e.g., RTX 50 Series, RTX 40 Series, RTX 30 Series, see [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) for details) or Apple Silicon M-series chips.

Python environment requirements: Python 3.10+.

Use `pip3` to install DiffSynth-Engine from PyPI:

```shell
pip3 install diffsynth-engine
```

DiffSynth-Engine also supports installation from source, which provides access to the latest features but might come with stability issues. We recommend installing the stable version via `pip3`.

```shell
git clone https://github.com/modelscope/diffsynth-engine.git && cd diffsynth-engine
pip3 install -e .
```

## Model Download

DiffSynth-Engine supports loading models from the [ModelScope Model Hub](https://www.modelscope.cn/aigc/models) by model ID. For example, on the [MajicFlus model page](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0), we can find the model ID and the corresponding model filename in the image below.

![Image](https://github.com/user-attachments/assets/a6f71768-487d-4376-8974-fe6563f2896c)

Next, download the MajicFlus model with the following code.

```python
from diffsynth_engine import fetch_model

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")
```

![Image](https://github.com/user-attachments/assets/596c3383-23b3-4372-a7ce-3c4e1c1ba81a)

For sharded models, specify multiple files using the `path` parameter.

```python
from diffsynth_engine import fetch_model

model_path = fetch_model("Wan-AI/Wan2.1-T2V-14B", path=[
    "diffusion_pytorch_model-00001-of-00006.safetensors",
    "diffusion_pytorch_model-00002-of-00006.safetensors",
    "diffusion_pytorch_model-00003-of-00006.safetensors",
    "diffusion_pytorch_model-00004-of-00006.safetensors",
    "diffusion_pytorch_model-00005-of-00006.safetensors",
    "diffusion_pytorch_model-00006-of-00006.safetensors",
])
```

It also supports using wildcards to match multiple files.

```python
from diffsynth_engine import fetch_model

model_path = fetch_model("Wan-AI/Wan2.1-T2V-14B", path="diffusion_pytorch_model*.safetensors")
```

The file path `model_path` returned by the `fetch_model` function is the path to the downloaded file(s).

## Model Types

Diffusion models come in a wide variety of architectures. Each model is loaded and run for inference by a corresponding pipeline. The model types we currently support include:

| Model Architecture | Example                                                      | Pipeline              |
| :----------------- | :----------------------------------------------------------- | :-------------------- |
| SD1.5              | [DreamShaper](https://www.modelscope.cn/models/MusePublic/DreamShaper_SD_1_5) | `SDImagePipeline`     |
| SDXL               | [RealVisXL](https://www.modelscope.cn/models/MusePublic/42_ckpt_SD_XL) | `SDXLImagePipeline`   |
| FLUX               | [MajicFlus](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0) | `FluxImagePipeline`   |
| Qwen-Image         | [Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) | `QwenImagePipeline` |
| Wan2.1             | [Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | `WanVideoPipeline`    |
| Wan2.2             | [Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) | `WanVideoPipeline` |
| Wan2.2 S2V      | [Wan2.2-S2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B) | `WanSpeech2VideoPipeline` |
| SD1.5 LoRA         | [Detail Tweaker](https://www.modelscope.cn/models/MusePublic/Detail_Tweaker_LoRA_xijietiaozheng_LoRA_SD_1_5) | `SDImagePipeline`     |
| SDXL LoRA          | [Aesthetic Anime](https://www.modelscope.cn/models/MusePublic/100_lora_SD_XL) | `SDXLImagePipeline`   |
| FLUX LoRA          | [ArtAug](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1) | `FluxImagePipeline`   |
| Qwen-Image LoRA    | [QwenCapybara](https://www.modelscope.cn/models/MusePublic/QwenCapybara) | `QwenImagePipeline` |
| Wan2.1 LoRA        | [Highres-fix](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1) | `WanVideoPipeline`    |

Among these, SD1.5, SDXL, FLUX, and Qwen-Image are base models for image generation, while Wan2.x is a base model for video generation. Base models can generate content independently. SD1.5 LoRA, SDXL LoRA, FLUX LoRA, Qwen-Image LoRA and Wan2.1 LoRA are [LoRA](https://arxiv.org/abs/2106.09685) models. LoRA models are trained as "additional branches" on top of base models to enhance specific capabilities. They must be combined with a base model to be used for generation.

We will continuously update DiffSynth-Engine to support more models. (Wan2.2 LoRA is coming soon❗)

## Model Inference

After the model is downloaded, load the model with the corresponding pipeline and perform inference.

### Image Generation(Qwen-Image)

The following code calls `QwenImagePipeline` to load the [Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) model and generate an image. Recommended resolutions are 928×1664, 1104×1472, 1328×1328, 1472×1104, and 1664×928, with a suggested cfg_scale of 4. If no negative_prompt is provided, it defaults to a single space character (not an empty string). For multi-GPU parallelism, currently only cfg parallelism is supported (parallelism=2), with other optimization efforts underway.

```python
from diffsynth_engine import fetch_model, QwenImagePipeline, QwenImagePipelineConfig

config = QwenImagePipelineConfig.basic_config(
    model_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="transformer/*.safetensors"),
    encoder_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="text_encoder/*.safetensors"),
    vae_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="vae/*.safetensors"),
    parallelism=2,
)
pipe = QwenImagePipeline.from_pretrained(config)

prompt = """
    一副典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联上左书“思涌如泉万类灵感皆可触”，右书“智启于问千机代码自天成”，横批“AI脑洞力”，字体飘逸灵动，兼具传统笔意与未来感。中间挂着一幅中国风的画作，内容是岳阳楼，云雾缭绕间似有数据流光隐现，古今交融，意境深远。
    """
negative_prompt = " "
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=4.0,
    width=1104,
    height=1472,
    num_inference_steps=30,
    seed=42,
)
image.save("image.png")
```

Please note that if some necessary modules, like text encoders, are missing from a model repository, the pipeline will automatically download the required files.

### Detailed Parameters(Qwen-Image)

In the image generation pipeline `pipe`, we can use the following parameters for fine-grained control:

* `prompt`: The prompt, used to describe the content of the generated image, It supports multiple languages (Chinese, English, Japanese, etc.), e.g., “一只猫” (Chinese), "a cat" (English), or "庭を走る猫" (Japanese).
* `negative_prompt`: The negative prompt, used to describe content you do not want in the image, it defaults to a single space character (not an empty string), e.g., "ugly".
* `cfg_scale`: The guidance scale for [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598). A larger value usually results in stronger correlation between the text and the image but reduces the diversity of the generated content.
* `height`: Image height.
* `width`: Image width.
* `num_inference_steps`: The number of inference steps. Generally, more steps lead to longer computation time but higher image quality.
* `seed`: The random seed. A fixed seed ensures reproducible results.

### Image Generation

The following code calls `FluxImagePipeline` to load the [MajicFlus](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0) model and generate an image. To load other types of models, replace `FluxImagePipeline` and `FluxPipelineConfig` in the code with the corresponding pipeline and config.

```python
from diffsynth_engine import fetch_model, FluxImagePipeline, FluxPipelineConfig

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")

config = FluxPipelineConfig.basic_config(model_path=model_path, device='cuda:0')
pipe = FluxImagePipeline.from_pretrained(config)
image = pipe(prompt="a cat")
image.save("image.png")
```

Please note that if some necessary modules, like text encoders, are missing from a model repository, the pipeline will automatically download the required files.

#### Detailed Parameters

In the image generation pipeline `pipe`, we can use the following parameters for fine-grained control:

* `prompt`: The prompt, used to describe the content of the generated image, e.g., "a cat".
* `negative_prompt`: The negative prompt, used to describe content you do not want in the image, e.g., "ugly".
* `cfg_scale`: The guidance scale for [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598). A larger value usually results in stronger correlation between the text and the image but reduces the diversity of the generated content.
* `clip_skip`: The number of layers to skip in the [CLIP](https://arxiv.org/abs/2103.00020) text encoder. The more layers skipped, the lower the text-image correlation, but this can lead to interesting variations in the generated content.
* `input_image`: Input image, used for image-to-image generation.
* `denoising_strength`: The denoising strength. When set to 1, a full generation process is performed. When set to a value between 0 and 1, some information from the input image is preserved.
* `height`: Image height.
* `width`: Image width.
* `num_inference_steps`: The number of inference steps. Generally, more steps lead to longer computation time but higher image quality.
* `seed`: The random seed. A fixed seed ensures reproducible results.

#### Loading LoRA

We supports loading LoRA on top of the base model. For example, the following code loads a [Cheongsam LoRA](https://www.modelscope.cn/models/DonRat/MAJICFLUS_SuperChinesestyleheongsam) based on the [MajicFlus](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0) model to generate images of cheongsams, which the base model might struggle to create.

```python
from diffsynth_engine import fetch_model, FluxImagePipelin, FluxPipelineConfige

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")
lora_path = fetch_model("DonRat/MAJICFLUS_SuperChinesestyleheongsam", path="麦橘超国风旗袍.safetensors")

config = FluxPipelineConfig.basic_config(model_path=model_path, device="cuda:0")
pipe = FluxImagePipeline.from_pretrained(config)
pipe.load_lora(path=lora_path, scale=1.0)
image = pipe(prompt="a girl, qipao")
image.save("image.png")
```

The `scale` parameter in the code controls the degree of influence the LoRA model has on the base model. A value of 1.0 is usually sufficient. When set to a value greater than 1, the LoRA's effect will be stronger, but this may cause artifacts or degradation in the image content. Please adjust this parameter with caution.

#### VRAM Optimization

DiffSynth-Engine supports various levels of VRAM optimization, allowing models to run on GPUs with low VRAM. For example, at `bfloat16` precision and with no optimization options enabled, the FLUX model requires 35.84GB of VRAM for inference. By adding the parameter `offload_mode="cpu_offload"`, the VRAM requirement drops to 22.83GB. Furthermore, using `offload_mode="sequential_cpu_offload"` reduces the requirement to just 4.30GB, although this comes with an increase of inference time.

```python
from diffsynth_engine import fetch_model, FluxImagePipeline, FluxPipelineConfig

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")

config = FluxPipelineConfig.basic_config(model_path=model_path, device="cuda:0", offload_mode="sequential_cpu_offload")
pipe = FluxImagePipeline.from_pretrained(config)
image = pipe(prompt="a cat")
image.save("image.png")
```

### Video Generation

DiffSynth-Engine also supports video generation. The following code loads the [Wan Video Generation Model](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) and generates a video.

```python
from diffsynth_engine import fetch_model, WanVideoPipeline, WanPipelineConfig
from diffsynth_engine.utils.video import save_video

model_path = fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors")

config = WanPipelineConfig.basic_config(model_path=model_path, device="cuda:0")
pipe = WanVideoPipeline.from_pretrained(config)
# The prompt translates to: "A lively puppy runs quickly on a green lawn. The puppy has brownish-yellow fur,
# its two ears are perked up, and it looks focused and cheerful. Sunlight shines on it,
# making its fur look especially soft and shiny."
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```

#### Detailed Parameters

In the video generation pipeline `pipe`, we can use the following parameters for fine-grained control:

* `prompt`: The prompt, used to describe the content of the generated video, e.g., "a cat".
* `negative_prompt`: The negative prompt, used to describe content you do not want in the video, e.g., "ugly".
* `cfg_scale`: The guidance scale for [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598). A larger value usually results in stronger correlation between the text and the video but reduces the diversity of the generated content.
* `input_image`: Input image, only effective in image-to-video models, such as [Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P).
* `input_video`: Input video, used for video-to-video generation.
* `denoising_strength`: The denoising strength. When set to 1, a full generation process is performed. When set to a value between 0 and 1, some information from the input video is preserved.
* `height`: Video frame height.
* `width`: Video frame width.
* `num_frames`: Number of video frames.
* `num_inference_steps`: The number of inference steps. Generally, more steps lead to longer computation time but higher video quality.
* `seed`: The random seed. A fixed seed ensures reproducible results.

#### Loading LoRA

We supports loading LoRA on top of the base model. For example, the following code loads a [High-Resolution Fix LoRA](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1) on top of the [Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) model to improve the generation quality at high resolutions.

```python
from diffsynth_engine import fetch_model, WanVideoPipeline, WanPipelineConfig
from diffsynth_engine.utils.video import save_video

model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors")
lora_path = fetch_model("DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1", path="model.safetensors")

config = WanPipelineConfig.basic_config(model_path=model_path, device="cuda:0")
pipe = WanVideoPipeline.from_pretrained(config)
pipe.load_lora(path=lora_path, scale=1.0)
# The prompt translates to: "A lively puppy runs quickly on a green lawn. The puppy has brownish-yellow fur,
# its two ears are perked up, and it looks focused and cheerful. Sunlight shines on it,
# making its fur look especially soft and shiny."
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```

The `scale` parameter in the code controls the degree of influence the LoRA model has on the base model. A value of 1.0 is usually sufficient. When set to a value greater than 1, the LoRA's effect will be stronger, but this may cause artifacts or degradation in the image content. Please adjust this parameter with caution.

#### Multi-GPU Parallelism

We supports multi-GPU parallel inference of the Wan2.1 model for faster video generation. Add the parameters `parallelism=4` (the number of GPUs to use) into the code to enable parallelism.

```python
from diffsynth_engine import fetch_model, WanVideoPipeline, WanPipelineConfig
from diffsynth_engine.utils.video import save_video

model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors")

config = WanPipelineConfig.basic_config(model_path=model_path, device="cuda", parallelism=4)
pipe = WanVideoPipeline.from_pretrained(config)
# The prompt translates to: "A lively puppy runs quickly on a green lawn. The puppy has brownish-yellow fur,
# its two ears are perked up, and it looks focused and cheerful. Sunlight shines on it,
# making its fur look especially soft and shiny."
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```
