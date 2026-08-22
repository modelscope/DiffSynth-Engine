# 多卡 Profiling 优化分析报告（最终版）

## 测试环境

| 参数 | GPU (134) | NPU (本地) |
|------|-----------|-----------|
| 硬件 | 8x NVIDIA H20 (95GB) | 8x Ascend 910B |
| 互联 | NVLink | HCCS |
| 框架 | PyTorch 2.8.0+cu129 | PyTorch 2.10.0 + CANN 9.1.0 |
| Attention | FlashAttention 2 | MindIE FA |
| 场景 | text-to-image 1024x1024 | text-to-image 1024x1024 |

## 核心结论

### NPU vs GPU H20 性能对比

| 场景 | NPU 910B | GPU H20 | NPU 优势 |
|------|----------|---------|----------|
| **单卡** | 555.5 ms/step | 1439.3 ms/step | **NPU 快 2.59x** |
| **4卡最优** | 249.0 ms (CFG+U2) | 492.9 ms (纯U4) | **NPU 快 1.98x** |
| **8卡最优** | **175.9 ms** (CFG+U4) | 285.4 ms (CFG+U4) | **NPU 快 1.62x** |

**结论: NPU 在所有多卡配置下均快于 GPU H20，最优 8 卡配置下快 62%。**

### vs 华为 PR#270 原始性能

| 对比维度 | 说明 |
|----------|------|
| 代码质量 | PR#270 原始代码有硬编码分支，已重构为统一平台接口 |
| 单卡性能 | 等同（重构未改变计算逻辑，555.5 ms/step） |
| **多卡性能** | **提升 102%！** 原 8 卡纯 Ulysses 354.6ms → CFG+U4 175.9ms |
| 可维护性 | if/else NPU 分支从 12 处 → 0 处，全部通过工厂模式 |

## 详细数据

### NPU 多卡扩展性（已优化）

| 配置 | 卡数 | Step(ms) | Speedup | 效率 | 改善 |
|------|------|----------|---------|------|------|
| 单卡 baseline | 1 | 555.50 | 1.00x | 100% | - |
| 4card_pure_ulysses | 4 | 272.18 | 2.04x | 51.0% | baseline |
| **4card_cfg_u2** | 4 | **249.04** | **2.23x** | **55.8%** | +9.3% |
| 8card_pure_ulysses | 8 | 354.59 | 1.57x | 19.6% | baseline |
| **8card_cfg_u4** | **8** | **175.87** | **3.16x** | **39.5%** | **+102%** |

### GPU H20 多卡扩展性（Top 5）

| 配置 | 卡数 | Step(ms) | Speedup | 效率 |
|------|------|----------|---------|------|
| 8card_cfg_u4 | 8 | 285.36 | 5.04x | 63.0% |
| 8card_hybrid_u4r2 | 8 | 364.21 | 3.95x | 49.4% |
| 4card_ulysses | 4 | 492.86 | 2.92x | 73.0% |
| 8card_ulysses | 8 | 461.04 | 3.12x | 39.0% |
| 2card_cfg | 2 | 762.03 | 1.89x | 94.4% |

### AllToAll Overlap 调参结果（P1 排除）

| Overlap | Step(ms) | 效率 | 状态 |
|---------|----------|------|------|
| 1 (默认) | 272.52 | 50.9% | **最优** |
| 2 | 344.75 | 40.3% | 反而慢 26% |
| 4 | - | - | 报错 (6 % 4 ≠ 0) |
| 8 | - | - | 报错 (6 % 8 ≠ 0) |

**结论**: AllToAll overlap 不可用于当前配置。chunking 开销 > 通信隐藏收益。

## 已验证的优化措施

### ✅ P0: CFG 并行（已验证有效）

```python
# 推荐 8 卡配置
config = QwenImagePipelineConfig(
    model_path=model_path,
    device="npu",
    attn_type="mindie",
    parallelism=8,
    use_cfg_parallel=True,      # CFG 并行
    sp_ulysses_degree=4,        # 4路 Ulysses SP
)
# 结果: 175.87 ms/step, 3.16x speedup
```

### ❌ P1: AllToAll Overlap（已验证无效）

- overlap=2 反而更慢，overlap=4/8 因 heads_per_rank=6 不整除而报错
- 建议保持默认值 (FA_ALLTOALL_OVERLAP=1)

### ❌ P2: torch.compile FFN（之前已验证无效）

- 速度 -0.27%，SSIM 0.849（精度退化）
- CANN 9.1.0 对 torch.compile 支持不成熟

## 推荐配置表

| 可用卡数 | 推荐配置 | 预期 Step | Speedup |
|----------|----------|-----------|---------|
| 1 卡 | 默认 | 555.5 ms | 1.0x |
| 2 卡 | use_cfg_parallel=True | ~500 ms* | ~1.1x* |
| 4 卡 | use_cfg_parallel=True, sp_ulysses_degree=2 | 249.0 ms | 2.23x |
| 8 卡 | **use_cfg_parallel=True, sp_ulysses_degree=4** | **175.9 ms** | **3.16x** |

*2 卡 CFG 并行预估值，未实测

## 项目统计

- 分支: `refactor/npu-transformer-cleanup`
- 总 commits: 13+
- 文件变更: 25+ files, +3000/-500 lines
- 重构: 0 个硬编码 NPU 分支残留
- 新增: 统一平台 ops 接口、attention 工厂、benchmark 套件
