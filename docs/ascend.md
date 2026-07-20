# Ascend A5 / Ascend 950 推理指南

DiffSynth-Engine 首批支持在单张 Ascend A5 / Ascend 950 上运行 Qwen-Image、
Qwen-Image-Edit-2509 和 Qwen-Image-Edit-2511。NPU 不会自动替换默认 CUDA 路径，
必须显式设置 `device="npu:0"`。

## 环境

请使用同一发布组合中的 CANN、PyTorch、torch-npu 和 MindIE-SD 3.x。torch-npu 与
MindIE-SD 不属于 DiffSynth-Engine 的核心 PyPI 依赖，需要根据华为发布矩阵单独安装。
硬件 CI 镜像应记录实际 wheel 版本；项目不在 `pyproject.toml` 中固定这些平台依赖。

| 组件 | 要求 |
| --- | --- |
| 硬件 | Ascend A5 / Ascend 950 |
| CANN | 与 torch-npu wheel 配套的版本 |
| torch-npu | 与当前 PyTorch 和 CANN 配套的版本 |
| MindIE-SD | 3.x，与当前 torch-npu/CANN 配套 |

普通 BF16 NPU 推理只要求 torch-npu。MindIE-SD 不存在时，`AttnImpl.AUTO` 会从
MindIE attention 回退到 PyTorch SDPA，再回退到 eager attention。显式选择 MindIE
attention、MindIE compile 或原生量化时不会静默降级，而是在模型分配前报告缺失能力。

安装后可检查能力：

```python
from diffsynth_engine.platforms import probe_ascend_capabilities

print(probe_ascend_capabilities())
```

该调用会为基础设备、MindIE attention、compile、MXFP8、W4A4 和 FP8 attention
分别执行一次最小算子并缓存结果，不只检查 Python 模块能否导入。正常推理则按需探测所使用的
功能；例如 BF16 `AUTO` 配置不会在初始化阶段触发量化或 compile 探测。

## BF16 推理

完整示例见 `examples/qwen_image_ascend.py`，核心配置如下：

```python
from diffsynth_engine import AttnImpl, QwenImagePipeline, QwenImagePipelineConfig

config = QwenImagePipelineConfig.basic_config(
    model_path=model_path,
    encoder_path=encoder_path,
    vae_path=vae_path,
    device="npu:0",
    parallelism=1,
)
config.dit_attn_impl = AttnImpl.AUTO
pipe = QwenImagePipeline.from_pretrained(config)
```

如需强制使用 MindIE-SD attention，可设置：

```python
config.dit_attn_impl = AttnImpl.MINDIE
```

该模式缺少 MindIE-SD 或 `attention_forward` API 不兼容时会立即失败。首批版本不支持
NPU 长上下文 attention，也不支持 `parallelism > 1`；多 NPU 配置会 fail-fast。

## MindIE compile

compile 保持显式开启，只编译 Qwen DiT 中重复的 Transformer blocks，不编译 encoder
或 VAE：

```python
config.use_torch_compile = True
```

NPU 使用 `MindieSDBackend()`，CUDA/ROCm 的空参数 `torch.compile` 行为不变。MindIE
compile 需要静态权重，不能与 CPU/model/sequential offload、动态 LoRA 或 ControlNet、
原生量化组合。FB-cache 的控制流位于已编译 block 外，可以与 compile 组合验证。

## 原生量化

原生量化从未量化的 Qwen 权重在线构建，不接受 Nunchaku/SVDQ/AWQ packed checkpoint。

```python
from diffsynth_engine import QuantizationConfig

# W8A8_MXFP8 linear
config.quantization = QuantizationConfig(backend="mindie", linear="fp8")

# W4A4_MXFP4_DYNAMIC linear
config.quantization = QuantizationConfig(backend="mindie", linear="int4")

# FP8_DYNAMIC attention，可与上述任一 linear 配置合并
config.quantization = QuantizationConfig(
    backend="mindie",
    linear="fp8",
    attention="fp8",
)
```

Qwen denoise 循环会在每一步更新 MindIE `TimestepManager`。如果运行时没有对应的
MXFP8、W4A4 或 FP8 attention 算子，初始化直接报错，不会退回 BF16。原生量化不能与
offload、compile、动态 LoRA 或 ControlNet 组合。

旧配置 `use_fp8_linear=True` 仍然保留：CUDA/ROCm 继续走现有 `_scaled_mm` 路径，NPU
映射为 MindIE `W8A8_MXFP8`。推荐新代码使用 `QuantizationConfig`。

## 支持矩阵

| 场景 | 状态 | 约束 |
| --- | --- | --- |
| Qwen-Image / Edit-2509 / Edit-2511 BF16 | 支持 | 单 NPU，AUTO 或 MindIE attention |
| LoRA / ControlNet | 支持 | BF16；可与 offload 或 FB-cache 组合 |
| FB-cache | 支持 | 可分别与 BF16、compile 或原生量化组合验证 |
| CPU/model/sequential offload | 支持 | 不与 compile 或原生量化组合 |
| MindIE compile | 支持 | 静态权重；不含动态 LoRA/ControlNet |
| MXFP8 / W4A4 linear、FP8 attention | 支持 | A5 算子能力探测必须通过 |
| Nunchaku/SVDQ/AWQ checkpoint | NPU 不支持 | 使用未量化权重和 MindIE 原生量化 |
| 多 NPU | 暂不支持 | `parallelism > 1` 明确报错 |

## 硬件测试

PR [#259](https://github.com/modelscope/DiffSynth-Engine/pull/259) 的 v1 实现已经在单张
Ascend 950 上完成 Qwen-Image-Edit-2511 实测，可作为主线融合的硬件基线。测试使用
`device="npu:0"`、MindIE attention、`parallelism=1`、预热后 10 个推理步：

| 输入 | H20 端到端 | Ascend 950 端到端 | H20 / Ascend |
| --- | ---: | ---: | ---: |
| 1 image, 1024x1024 | 29.10s | 16.97s | 1.71x |
| 2 images, 1024x1024 | 49.12s | 30.65s | 1.60x |
| 4 images, 1280x720 | 98.35s | 54.68s | 1.80x |

三组设备 kernel 总时间比分别为 1.97x、1.99x、1.97x；主要热点是 GEMM/Linear 和
FlashAttention。该结果证明 v1 硬件路径可用，但主线重构后的最终发布仍需在同一镜像中
重新执行下列回归，不能用 v1 结果替代主线 golden、compile 和量化验收。

先运行最小 NPU 与 MindIE attention 探测：

```bash
RUN_ASCEND_TESTS=1 python -m unittest tests.test_platforms.test_ascend_integration
```

量化发布验收还需启用完整能力断言，并执行 Qwen golden tensor、文生图、Edit、
ControlNet、LoRA、FB-cache、offload、compile 和三种量化图像用例：

```bash
RUN_ASCEND_TESTS=1 RUN_ASCEND_QUANT_TESTS=1 \
  python -m unittest tests.test_platforms.test_ascend_integration
```

图像阈值沿用现有测试：基础 Qwen 0.99、Edit/ControlNet 0.95、量化 0.90。compile
输出还需与同机 eager 对齐，并确认 MindIE 融合 graph 实际生成。
