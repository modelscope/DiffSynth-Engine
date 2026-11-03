# Qwen Image NPU 适配技术文档

## 1. 概述

本文档描述 Qwen Image 模型在华为 Atlas 950 ProR (Ascend 910B) NPU 上的完整适配方案。

### 支持场景

- **text-to-image**: 文本生成图像
- **image-edit**: 图像编辑
- **image-edit-plus**: 增强图像编辑
- **layered-generation**: 分层生成

## 2. 环境要求

### 硬件

| 项目 | 规格 |
|------|------|
| 加速卡 | Atlas 950 ProR (Ascend 910B) |
| 卡数 | 8 卡 |

### 软件栈

| 组件 | 版本 |
|------|------|
| CANN | 9.1.0 |
| PyTorch | 2.10.0 (with torch_npu) |
| MindIE SDK | mindiesd |
| Python | 3.11+ |

### 环境变量

```bash
# 启用 MindIE 融合算子
export USE_MINDIESD_FUSE=true

# 指向项目根目录
export PYTHONPATH=/path/to/DiffSynth-Engine:$PYTHONPATH
```

## 3. 快速启动

### 单卡推理

```python
from diffsynth_engine import DiffSynthEngine, QwenImagePipelineConfig

config = QwenImagePipelineConfig(
    model_path="Qwen/Qwen-Image",
    device="npu",
    attn_type="mindie",
)
engine = DiffSynthEngine(config)
image = engine("A cat sitting on a windowsill", num_inference_steps=28)
image.save("output.png")
```

### 多卡并行 (Ulysses SP)

```bash
torchrun --nproc_per_node=4 examples/qwen_image/run_text_to_image.py \
    --model-path Qwen/Qwen-Image \
    --device npu \
    --attn-type mindie \
    --parallelism 4 \
    --sp-ulysses-degree 4
```

## 4. 架构设计

### 4.1 统一平台 ops 接口

**文件**: `diffsynth_engine/platforms/ops.py`

提供统一的算子接口，根据运行平台自动选择最优实现:

| 接口 | NPU 实现 | GPU 路径 |
|------|----------|----------|
| `fused_rotary_embedding()` | `mindiesd.rotary_position_embedding` | 零开销直通原始实现 |
| `fused_layernorm_scale_shift()` | `mindiesd.layernorm_scale_shift` | 零开销直通原始实现 |
| `fused_rms_norm()` | `torch_npu.npu_rms_norm` | 零开销直通原始实现 |

设计原则:
- NPU 路径利用 MindIE SDK 融合算子获取加速
- GPU 路径保持零开销直通，不引入额外调度延迟
- 通过环境变量 `USE_MINDIESD_FUSE` 控制是否启用融合

### 4.2 Attention 实现选择

**文件**: `diffsynth_engine/layers/attention/ascend_long_context.py`

`use_ascend_long_ctx_attn(attn_type)` 根据 MindIE 可用性和并行配置判断是否适用：

- `attn_type="mindie"`，`is_mindie_sd_available()` 返回 `True`，SP 已初始化且 ring degree 为 1 时，返回 `True`。
- 其他情况返回 `False`；SP 尚未初始化时不会查询 ring degree。

Qwen-Image 在构造每层 attention 时根据返回值直接创建 `AscendLongContextAttention` 或
`USPAttention`，各层持有独立实例。辅助函数复用已有的 MindIE 可用性检查；具体 attention
API 的可用性仍由 attention backend 的 `check_availability()` 校验。

### 4.3 AscendLongContextAttention

**文件**: `diffsynth_engine/layers/attention/ascend_long_context.py`

从原 `layer.py` 提取为独立模块，专门针对昇腾 NPU 优化:

- **Ulysses Sequence Parallelism**: 将长序列切分到多卡并行处理
- **AllToAll 通信计算重叠** (overlap mode): 隐藏通信延迟
- **切分优化** (cut mode): 减少显存占用

### 4.4 平台注册与初始化

`registry.py` 提供 `register_platform(device_type, "module:Class")` 和
`get_platform_class(device_type)`，复用 `LazyImport`
和插件加载机制。重复注册会警告并跳过；获取 backend 只加载类，不检测设备或设置当前平台。
CPU、CUDA、ROCm、MPS 类仍放在 `platforms/__init__.py` 中默认加载，Ascend 类按需导入。

`PipelineConfig` 在配置阶段解析设备、检查注册类型、设备可用性和格式，并继续检查 attention。
这个过程不设置进程级 `_platform`。`auto` 按 CUDA/ROCm、NPU、MPS、CPU 顺序检测并缓存结果，
不受已选平台影响；显式设备编号保持不变。
设备规范化结果以字符串回写配置；显式加速设备编号须落在可见设备数量范围内。

初始化 pipeline 前需调用 `set_platform(config.device)` 初始化平台，否则调用 `get_platform()` 等依赖已选平台的接口时会报错。

选择 MindIE attention 时，可用性检查失败会报错；MindIE compile 可用性检查失败时回退默认编译器。

## 5. 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | `str` | `"auto"` | 设备类型: `"npu"`, `"cuda"`, `"auto"` |
| `attn_type` | `str` | `"sdpa"` | Attention 后端: `"sdpa"`, `"flash"`, `"mindie"` |
| `op_fusion` | `bool` | `True` | 是否启用算子融合 |
| `compile_ffn` | `bool` | `False` | 是否编译 FFN (实验性，当前 NPU 无收益) |
| `parallelism` | `int` | `1` | 并行卡数 |
| `sp_ulysses_degree` | `int` | `1` | Ulysses SP 并行度 |

## 6. 性能数据 (单卡, Atlas 950 ProR)

| 场景 | Steps | 耗时 (s) | 每步 (ms) |
|------|-------|----------|-----------|
| text-to-image 1024×1024 | 28 | 15.85 | 565.9 |
| image-edit | 50 | 78.29 | 1565.7 |
| image-edit-plus | 50 | 70.38 | 1407.5 |
| layered-generation | 50×3 | 33.56 | 671.2 |

## 7. 已知限制

1. **torch.compile 对 FFN 无优化效果**
   - 在当前 CANN 9.1.0 上，torch.compile 对 FFN 模块无加速效果且存在精度退化
   - 建议保持 `compile_ffn=False`

2. **AscendLongContextAttention 与 dynamo 不兼容**
   - 内部使用 stream/event 进行通信计算重叠
   - 已通过 `@torch.compiler.disable` 装饰器规避

3. **多卡模式需要 HCCL 初始化**
   - 框架内部自动处理，无需手动配置
   - 需确保所有 NPU 设备可见

4. **多卡性能数据待补充**
   - 当前仅验证单卡场景
   - 多卡 Ulysses SP 性能数据待后续补充

## 8. 测试

### NPU 单卡测试

```bash
python -m pytest tests/test_pipelines/test_qwen_image_npu.py -v
```

### NPU 多卡测试 (4 卡)

```bash
torchrun --nproc_per_node=4 -m pytest tests/test_pipelines/test_qwen_image_npu_parallel.py -v
```

### GPU 回归测试

```bash
python -m pytest tests/test_pipelines/test_qwen_image.py -v
```

## 9. 故障排查

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `MindIE SDK not found` | mindiesd 未安装或不在搜索路径 | 确认 mindiesd 已安装且 `PYTHONPATH` 正确 |
| `HCCL init failed` | HCCL 通信环境异常 | 检查 NCCL/HCCL 环境，确认所有 NPU 可见 |
| 精度异常 (SSIM < 0.95) | 融合算子未启用或版本不匹配 | 检查 `USE_MINDIESD_FUSE` 环境变量是否为 `true` |
