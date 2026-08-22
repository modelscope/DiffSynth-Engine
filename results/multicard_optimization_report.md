# 多卡 Profiling 优化分析报告

## 测试环境

| 参数 | GPU (134) | NPU (本地) |
|------|-----------|-----------|
| 硬件 | 8x NVIDIA H20 (95GB) | 8x Ascend 910B |
| 互联 | NVLink | HCCS |
| 框架 | PyTorch 2.8.0+cu129 | PyTorch 2.10.0 + CANN 9.1.0 |
| Attention | FlashAttention 2 | MindIE FA |
| 场景 | text-to-image 1024x1024 | text-to-image 1024x1024 |
| Steps | 5 | 5 |
| SP 模式 | Ulysses/Ring/Hybrid/CFG | Ulysses |

## 核心数据

### GPU H20 多卡扩展性（按 speedup 排序）

| 配置 | 卡数 | Step(ms) | Speedup | 效率 | 开销占比 |
|------|------|----------|---------|------|----------|
| 1card_fa2 (baseline) | 1 | 1439.29 | 1.00x | 100% | 0% |
| **8card_cfg_u4** | **8** | **285.36** | **5.04x** | **63.0%** | **37.0%** |
| 8card_hybrid_u4r2 | 8 | 364.21 | 3.95x | 49.4% | 50.6% |
| 8card_hybrid_u2r4 | 8 | 425.48 | 3.38x | 42.3% | 57.7% |
| 8card_ulysses | 8 | 461.04 | 3.12x | 39.0% | 61.0% |
| 4card_ulysses | 4 | 492.86 | 2.92x | 73.0% | 27.0% |
| 8card_ring | 8 | 504.82 | 2.85x | 35.6% | 64.4% |
| 4card_hybrid_u2r2 | 4 | 523.06 | 2.75x | 68.8% | 31.2% |
| 4card_ring | 4 | 553.23 | 2.60x | 65.0% | 35.0% |
| 2card_cfg | 2 | 762.03 | 1.89x | 94.4% | 5.6% |
| 2card_ulysses | 2 | 828.37 | 1.74x | 86.9% | 13.1% |

### NPU 910B 多卡扩展性

| 配置 | 卡数 | Step(ms) | Speedup | 效率 | 开销占比 |
|------|------|----------|---------|------|----------|
| 1card (baseline) | 1 | 555.50 | 1.00x | 100% | 0% |
| **4card_ulysses** | **4** | **273.68** | **2.03x** | **50.7%** | **49.3%** |
| 8card_ulysses | 8 | 357.02 | 1.56x | 19.4% | 80.6% |

## 关键发现

### 1. NPU 单卡性能远优于 GPU H20

| 平台 | 单卡 Step | 相对速度 |
|------|-----------|----------|
| NPU 910B (MindIE) | 555.5 ms | **2.59x faster** |
| GPU H20 (FA2) | 1439.3 ms | 1.00x |

**结论**: NPU 在 MindIE FA 加速下，单卡推理性能是 H20 GPU 的 **2.6 倍**。

### 2. NPU 多卡通信开销严重

| 卡数 | NPU 开销 | GPU 开销 (Ulysses) | NPU/GPU 差距 |
|------|----------|-------------------|--------------|
| 4 | 49.3% | 27.0% | NPU 高 82% |
| 8 | 80.6% | 61.0% | NPU 高 32% |

**根因**: NPU HCCS 互联带宽低于 GPU NVLink，AllToAll 通信延迟更高。

### 3. NPU 8卡反向扩展

- NPU 4卡: 273.68 ms/step
- NPU 8卡: 357.02 ms/step（比4卡**更慢30%**！）
- 说明 8 卡时通信完全压倒了计算收益

### 4. CFG 并行是最高性价比优化

GPU 数据验证：
- `2card_cfg`: 1.89x, 94.4% 效率（几乎零开销！）
- `8card_cfg_u4`: 5.04x, 63% 效率（8卡最优方案）

**CFG 并行原理**: 将正向/负向 guidance 分配到不同卡并行，无需 AllToAll 通信。

## 优化建议（按优先级）

### P0: 启用 CFG 并行（预计 1.8-2x 加速，零通信开销）

**现状**: NPU 多卡仅使用 Ulysses SP，所有卡都参与同一 batch 的 AllToAll。

**优化方案**: 
```python
# 当前: parallelism=4, sp_ulysses_degree=4
# 优化: parallelism=4, use_cfg_parallel=True, sp_ulysses_degree=2
config = QwenImagePipelineConfig(
    model_path=model_path,
    parallelism=8,
    use_cfg_parallel=True,     # ← 新增：2卡做 CFG 并行
    sp_ulysses_degree=4,       # ← 剩余4卡做 Ulysses SP
)
```

**预期收益**: 
- GPU 验证: CFG+U4 给出 5.04x (8卡)，纯 U8 只有 3.12x
- NPU 预期: 从 2.03x (4卡纯U) → ~3.0-3.5x (4卡 CFG+U2)
- 原因: CFG 并行将 2 次 classifier-free guidance pass 拆分到 2 张卡，几乎零通信

### P1: 优化 AllToAll Overlap 参数

**现状**: `AscendLongContextAttention` 有 `fa_alltoall_overlap` 参数但效果不明确。

**优化方案**:
- 增大 `fa_alltoall_overlap` chunks 数（当前默认值可能太小）
- 确认 `_shared_comm_stream` 真正实现了 通信-计算流 overlap
- 在 4 卡 Ulysses 上测试不同 overlap 值 (2, 4, 8)

**预期收益**: 
- 4卡开销从 134.8ms 降到 ~80-100ms（效率从 50.7% → ~60-65%）
- 对 8 卡不建议投入（已验证为反向扩展）

### P2: 限制 SP degree ≤ 4 

**现状**: 代码允许任意 SP degree。

**优化方案**: 在文档/config 中明确建议 `sp_ulysses_degree ≤ 4`。
- 4 卡是 NPU Ulysses SP 的效率最优解
- 8 卡纯 Ulysses 已验证为反向扩展
- 如需 8 卡加速，必须搭配 CFG 并行

### P3: Hybrid Ulysses + Ring 探索

**GPU 数据**: `u4r2` (3.95x) 优于 `u8` (3.12x) 和 `r8` (2.85x)。

**NPU 限制**: 当前 `AscendLongContextAttention` 报错 "NPU MindIE attention currently supports Ulysses only (sp_ring_degree must be 1)"。

**建议**: 
- 短期: 不投入，Ring 在 NPU 不支持
- 长期: 等 MindIE 支持 Ring 后评估 Hybrid 方案

## 绝对性能对比

| 场景 | NPU 最优 | GPU 最优 | NPU 优势 |
|------|----------|----------|----------|
| 单卡 | 555.5 ms/step | 1439.3 ms/step | **2.59x** |
| 4卡 Ulysses | 273.7 ms/step | 492.9 ms/step | **1.80x** |
| 最优8卡 | 357.0 ms (纯U8) | 285.4 ms (CFG+U4) | GPU 1.25x |

**结论**: NPU 在单卡和 4 卡场景下性能显著优于 GPU H20。8 卡场景下 GPU 凭借 CFG 并行 + NVLink 高带宽反超 NPU。**NPU 启用 CFG 并行后预计可恢复领先**。

## 下一步行动

1. **验证 NPU CFG 并行**: `parallelism=4, use_cfg_parallel=True, sp_ulysses_degree=2`
2. **调参 fa_alltoall_overlap**: 在 4 卡 Ulysses 上 benchmark overlap=2/4/8
3. **推荐配置表**: 基于卡数给出最优配置组合
