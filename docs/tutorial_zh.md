# DiffSynth-Engine 使用指南

## 安装

在使用 DiffSynth-Engine 前，请先确保您的硬件设备满足以下要求:

* NVIDIA GPU CUDA 计算能力 8.6+（例如 RTX 50 Series、RTX 40 Series、RTX 30 Series 等，详见 [NVidia 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)）或 Apple Silicon M 系列芯片

以及 Python 环境需求: Python 3.10+。

使用 `pip3` 工具从 PyPI 安装 DiffSynth-Engine:

```shell
pip3 install diffsynth-engine
```

DiffSynth-Engine 也支持通过源码安装，这种方式可体验最新的特性，但可能存在稳定性问题，我们推荐您通过 `pip3` 安装稳定版本。

```shell
git clone https://github.com/modelscope/diffsynth-engine.git && cd diffsynth-engine
pip3 install -e .
```

## 模型下载

DiffSynth-Engine 可以直接加载[魔搭社区模型库](https://www.modelscope.cn/aigc/models)中的模型，这些模型通过模型 ID 进行检索。例如，在[麦橘超然的模型页面](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0)，我们可以在下图中找到模型 ID 以及对应的模型文件名。

![Image](https://github.com/user-attachments/assets/a6f71768-487d-4376-8974-fe6563f2896c)

接下来，通过以下代码即可自动下载麦橘超然模型。

```python
from diffsynth_engine import fetch_model

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")
```

![Image](https://github.com/user-attachments/assets/596c3383-23b3-4372-a7ce-3c4e1c1ba81a)

对于模型分片的情况，可以通过 `path` 参数指定多个文件。

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

也支持使用通配符来匹配多个文件。

```python
from diffsynth_engine import fetch_model

model_path = fetch_model("Wan-AI/Wan2.1-T2V-14B", path="diffusion_pytorch_model*.safetensors")
```

`fetch_model` 函数返回的文件路径 `model_path` 即为下载后的文件路径。

## 模型类型

Diffusion 模型包含多种多样的模型结构，每种模型由对应的流水线进行加载和推理，目前我们支持的模型类型包括:

| 模型结构         | 样例                                                         | 流水线              |
| --------------- | ------------------------------------------------------------ | ------------------- |
| SD1.5           | [DreamShaper](https://www.modelscope.cn/models/MusePublic/DreamShaper_SD_1_5) | `SDImagePipeline`   |
| SDXL            | [RealVisXL](https://www.modelscope.cn/models/MusePublic/42_ckpt_SD_XL) | `SDXLImagePipeline` |
| FLUX            | [麦橘超然](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0) | `FluxImagePipeline` |
| Qwen-Image      | [Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) | `QwenImagePipeline` |
| Wan2.1          | [Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | `WanVideoPipeline` |
| Wan2.2          | [Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) | `WanVideoPipeline` |
| Wan2.2 S2V      | [Wan2.2-S2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B) | `WanSpeech2VideoPipeline` |
| SD1.5 LoRA      | [Detail Tweaker](https://www.modelscope.cn/models/MusePublic/Detail_Tweaker_LoRA_xijietiaozheng_LoRA_SD_1_5) | `SDImagePipeline`   |
| SDXL LoRA       | [Aesthetic Anime](https://www.modelscope.cn/models/MusePublic/100_lora_SD_XL) | `SDXLImagePipeline` |
| FLUX LoRA       | [ArtAug](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1) | `FluxImagePipeline` |
| Qwen-Image LoRA | [QwenCapybara](https://www.modelscope.cn/models/MusePublic/QwenCapybara) | `QwenImagePipeline` |
| Wan2.1 LoRA     | [Highres-fix](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1) | `WanVideoPipeline` |

其中 SD1.5、SDXL、FLUX、Qwen-Image 为图像生成的基础模型，Wan2.x 是视频生成的基础模型，基础模型可以独立进行内容生成；SD1.5 LoRA、SDXL LoRA、FLUX LoRA、Qwen-Image LoRA、Wan2.1 LoRA 为 [LoRA](https://arxiv.org/abs/2106.09685) 模型，LoRA 模型是在基础模型上以“额外分支”的形式训练的，能够增强模型某方面的能力，需要与基础模型结合后才可用于图像生成。

我们会持续更新 DiffSynth-Engine 以支持更多模型。(即将支持Wan2.2 LoRA❗)

## 模型推理

模型下载完毕后，我们可以根据对应的模型类型选择流水线加载模型并进行推理。

### 图像生成(Qwen-Image)

以下代码可以调用 `QwenImagePipeline` 加载[Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image)模型生成一张图。推荐分辨率为928×1664, 1104×1472, 1328×1328, 1472×1104, 1664×928，cfg_scale为4，如果没有negative_prompt默认为一个空格而不是空字符串。多卡并行目前支持cfg并行(parallelism=2)，其他优化工作正在进行中。

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

请注意，某些模型库中缺乏必要的文本编码器等模块，我们的代码会自动补充下载所需的模型文件。

#### 详细参数(Qwen-Image)

在图像生成流水线 `pipe` 中，我们可以通过以下参数进行精细的控制:

* `prompt`: 提示词，用于描述生成图像的内容，支持多种语言(中文/英文/日文等)，例如“一只猫”/"a cat"/"庭を走る猫"。
* `negative_prompt`: 负面提示词，用于描述不希望图像中出现的内容，例如“ugly”，默认为一个空格而不是空字符串， " "。
* `cfg_scale`:[Classifier-free guidance](https://arxiv.org/abs/2207.12598) 的引导系数，通常更大的引导系数可以达到更强的文图相关性，但会降低生成内容的多样性，推荐值为4。
* `height`: 图像高度。
* `width`: 图像宽度。
* `num_inference_steps`: 推理步数，通常推理步数越多，计算时间越长，图像质量越高。
* `seed`: 随机种子，固定的随机种子可以使生成的内容固定。


### 图像生成

以下代码可以调用 `FluxImagePipeline` 加载[麦橘超然](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0)模型生成一张图。如果要加载其他结构的模型，请将代码中的 `FluxImagePipeline` 和 `FluxPipelineConfig` 替换成对应的流水线模块及配置。

```python
from diffsynth_engine import fetch_model, FluxImagePipeline, FluxPipelineConfig

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")

config = FluxPipelineConfig.basic_config(model_path=model_path, device='cuda:0')
pipe = FluxImagePipeline.from_pretrained(config)
image = pipe(prompt="a cat")
image.save("image.png")
```

请注意，某些模型库中缺乏必要的文本编码器等模块，我们的代码会自动补充下载所需的模型文件。

#### 详细参数

在图像生成流水线 `pipe` 中，我们可以通过以下参数进行精细的控制:

* `prompt`: 提示词，用于描述生成图像的内容，例如“a cat”。
* `negative_prompt`: 负面提示词，用于描述不希望图像中出现的内容，例如“ugly”。
* `cfg_scale`: [Classifier-free guidance](https://arxiv.org/abs/2207.12598) 的引导系数，通常更大的引导系数可以达到更强的文图相关性，但会降低生成内容的多样性。
* `clip_skip`: 跳过 [CLIP](https://arxiv.org/abs/2103.00020) 文本编码器的层数，跳过的层数越多，生成的图像与文本的相关性越低，但生成的图像内容可能会出现奇妙的变化。
* `input_image`: 输入图像，用于图生图。
* `denoising_strength`: 去噪力度，当设置为 1 时，执行完整的生成过程，当设置为 0 到 1 之间的值时，会保留输入图像中的部分信息。
* `height`: 图像高度。
* `width`: 图像宽度。
* `num_inference_steps`: 推理步数，通常推理步数越多，计算时间越长，图像质量越高。
* `seed`: 随机种子，固定的随机种子可以使生成的内容固定。

#### LoRA 加载

对于 LoRA 模型，请在加载模型后，进一步加载 LoRA 模型。例如，以下代码可以在[麦橘超然](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0)的基础上加载[旗袍 LoRA](https://www.modelscope.cn/models/DonRat/MAJICFLUS_SuperChinesestyleheongsam)，进而生成基础模型难以生成的旗袍图片。

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

代码中的 `scale` 可以控制 LoRA 模型对基础模型的影响程度，通常将其设置为 1 即可，当将其设置为大于 1 的值时，LoRA 的效果会更加明显，但画面内容可能会产生崩坏，请谨慎地调整这个参数。

#### 显存优化

DiffSynth-Engine 支持不同粒度的显存优化，让模型能够在低显存GPU上运行。例如，在 `bfloat16` 精度且不开启任何显存优化选项的情况下，FLUX 模型需要 35.84GB 显存才能进行推理。添加参数 `offload_mode="cpu_offload"` 后，显存需求降低到 22.83GB；进一步使用参数 `offload_mode="sequential_cpu_offload"` 后，只需要 4.30GB 显存即可进行推理，但推理时间有一定的延长。

```python
from diffsynth_engine import fetch_model, FluxImagePipeline, FluxPipelineConfig

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")

config = FluxPipelineConfig.basic_config(model_path=model_path, device="cuda:0", offload_mode="sequential_cpu_offload")
pipe = FluxImagePipeline.from_pretrained(config)
image = pipe(prompt="a cat")
image.save("image.png")
```

### 视频生成

DiffSynth-Engine 也支持视频生成，以下代码可以加载[通义万相视频生成模型](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)并生成视频。

```python
from diffsynth_engine import fetch_model, WanVideoPipeline, WanPipelineConfig
from diffsynth_engine.utils.video import save_video

model_path = fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors")

config = WanPipelineConfig.basic_config(model_path=model_path, device="cuda:0")
pipe = WanVideoPipeline.from_pretrained(config)
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```

#### 详细参数

在视频生成流水线 `pipe` 中，我们可以通过以下参数进行精细的控制:

* `prompt`: 提示词，用于描述生成图像的内容，例如“a cat”。
* `negative_prompt`: 负面提示词，用于描述不希望图像中出现的内容，例如“ugly”。
* `cfg_scale`: [Classifier-free guidance](https://arxiv.org/abs/2207.12598) 的引导系数，通常更大的引导系数可以达到更强的文图相关性，但会降低生成内容的多样性。
* `input_image`: 输入图像，只在图生视频模型中有效，例如 [Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)。
* `input_video`: 输入视频，用于视频生视频。
* `denoising_strength`: 去噪力度，当设置为 1 时，执行完整的生成过程，当设置为 0 到 1 之间的值时，会保留输入视频中的部分信息。
* `height`: 视频帧高度。
* `width`: 视频帧宽度。
* `num_frames`: 视频帧数。
* `num_inference_steps`: 推理步数，通常推理步数越多，计算时间越长，图像质量越高。
* `seed`: 随机种子，固定的随机种子可以使生成的内容固定。

#### LoRA 加载

对于 LoRA 模型，请在加载模型后，进一步加载 LoRA 模型。例如，以下代码可以在[Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)的基础上加载[高分辨率修复 LoRA](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1)，进而改善模型在高分辨率下的生成效果。

```python
from diffsynth_engine import fetch_model, WanVideoPipeline, WanPipelineConfig
from diffsynth_engine.utils.video import save_video

model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors")
lora_path = fetch_model("DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1", path="model.safetensors")

config = WanPipelineConfig.basic_config(model_path=model_path, device="cuda:0")
pipe = WanVideoPipeline.from_pretrained(config)
pipe.load_lora(path=lora_path, scale=1.0)
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```

代码中的 `scale` 可以控制 LoRA 模型对基础模型的影响程度，通常将其设置为 1 即可，当将其设置为大于 1 的值时，LoRA 的效果会更加明显，但画面内容可能会产生崩坏，请谨慎地调整这个参数。

#### 多卡并行

考虑到视频生成模型庞大的计算量，我们为 Wan2.1 模型提供了多卡并行的支持，只需要在代码中增加参数 `parallelism=4`（使用的GPU数量）即可。

```python
from diffsynth_engine import fetch_model, WanVideoPipeline, WanPipelineConfig
from diffsynth_engine.utils.video import save_video

model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors")

config = WanPipelineConfig.basic_config(model_path=model_path, device="cuda", parallelism=4)
pipe = WanVideoPipeline.from_pretrained(config)
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```
