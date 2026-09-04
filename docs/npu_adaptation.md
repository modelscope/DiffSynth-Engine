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

### 4.2 Attention 工厂

**文件**: `diffsynth_engine/layers/attention/factory.py`

`create_parallel_attention()` 根据平台和并行配置自动选择 Attention 实现:

```
┌─────────────────────────────────────────────┐
│         create_parallel_attention()          │
├─────────────────────────────────────────────┤
│  NPU + SP  → AscendLongContextAttention     │
│             (Ulysses SP + AllToAll overlap)  │
│  GPU/其他   → USPAttention                   │
└─────────────────────────────────────────────┘
```

### 4.3 AscendLongContextAttention

**文件**: `diffsynth_engine/layers/attention/ascend_long_context.py`

从原 `layer.py` 提取为独立模块，专门针对昇腾 NPU 优化:

- **Ulysses Sequence Parallelism**: 将长序列切分到多卡并行处理
- **AllToAll 通信计算重叠** (overlap mode): 隐藏通信延迟
- **切分优化** (cut mode): 减少显存占用

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
