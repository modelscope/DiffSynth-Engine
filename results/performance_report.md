# DiffSynth-Engine NPU 性能分析报告

> 测试环境: 华为昇腾 NPU (8卡) | PyTorch 2.10.0 | CANN 9.1.0 | MindIE FlashAttention | BFloat16  
> 测试日期: 2026-08

---

## 1. 执行摘要

DiffSynth-Engine 已完成华为昇腾 NPU 全场景适配，覆盖 text-to-image、image-edit、image-edit-plus、layered-generation 四大推理场景，全部通过正确性验证并稳定运行。

**当前性能水平：**
- 核心场景 (text-to-image 1024×1024) 端到端耗时 **15.85s / 28 steps**，单步耗时 **556ms**
- Denoising 阶段占管线 **98.3%**，其中 Attention 和 FFN 各占约 47% 和 46%，是绝对性能瓶颈
- 经评估，当前 CANN 栈下 `torch.compile` 对 FFN 无加速收益且引入精度退化，**不采用**
- 高收益优化方向（CFG Distillation 44%、Step Reduction 49%）均需模型训练介入，记录为后续方向

---

## 2. 场景性能矩阵

| 场景 | Steps | NPU 耗时(s) | 每步耗时(ms) | 峰值显存(MB) |
|------|-------|-------------|-------------|-------------|
| text-to-image-1024x1024 | 28 | 15.845 | 555.5 | 62,278.5 |
| image-edit | 50 | 78.286 | 1,565.7 | 62,301.3 |
| image-edit-plus | 50 | 70.377 | 1,407.5 | 62,289.3 |
| layered-generation | 50 × 3 layers | 33.561 | 671.2 | 63,487.4 |

**说明：**
- 所有场景均经过 2 次 warmup + 3 次计时取平均值
- image-edit 场景因输入分辨率较大（含参考图拼接），单步耗时高于 text-to-image
- layered-generation 为 3 层独立生成，每步耗时约为单层 text-to-image 的 1.2x

---

## 3. 组件耗时分解（text-to-image 场景）

基于 hook profiling 实测数据，管线总耗时 **15,831ms**（估算）/ **15,570ms**（实测中位数）:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Text Encode        35ms   (0.2%)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Denoising (28步)   15,555ms   (98.3%)                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Attention (MindIE FA)    263ms/step   47.4%                  │  │
│  │  FFN (GeLU + Linear)     255ms/step   46.0%                  │  │
│  │  Modulation (SiLU+Linear) 34ms/step    6.0%                  │  │
│  │  Other (Norm/RoPE/残差)    3ms/step    0.6%                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  VAE Decode         241ms   (1.5%)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

**关键观察：**
- 每步执行 60 blocks × 2 CFG passes = 120 次 Attention + 120 次 FFN 调用
- Attention 单次调用耗时 2.195ms，FFN (图像分支) 单次调用 1.939ms
- Text encode 调用 2 次（prompt + negative），每次 17.6ms
- VAE decode 仅占 1.5%，优化 ceiling 极低

---

## 4. 优化探索结果

| 优化项 | 理论 Ceiling | 实测结果 | 决策 |
|--------|-------------|---------|------|
| torch.compile FFN | 9.0% (1,430ms) | **-0.27%** (无收益) + SSIM=0.849 (精度退化) | **不采用** |
| VAE SDPA → MindIE FA | 0.5% (72ms) | 未实施 (ceiling < 5% 阈值) | **跳过** |
| 通信重叠优化 | N/A (单卡) | 未实施 (单卡无跨设备通信) | **跳过** |
| CFG Distillation 2→1 pass | **44.2%** (7,000ms) | 需模型蒸馏重训 (非代码优化) | **记录为后续方向** |
| Step Reduction 28→14 | **49.1%** (7,778ms) | 需一致性蒸馏训练 (非代码优化) | **记录为后续方向** |

### torch.compile 详细分析

| 指标 | Baseline | Compiled | Delta |
|------|----------|----------|-------|
| 5-step 耗时 | 3.310s | 3.319s | +0.27% |
| 单步耗时 | 661.9ms | 663.7ms | +1.77ms |
| 峰值显存 | 62,259.5 MB | 62,262.9 MB | +3.4 MB |
| 输出 SSIM | — | 0.849 | **< 0.95 阈值** |

**结论：** MindIE compile backend 在当前 CANN 9.1.0 栈上未能有效优化 FFN 内核，eager 模式已接近硬件效率上限。同时 compile 引入数值偏差导致图像质量不可接受。

---

## 5. GPU 基线对比

| 指标 | NPU (昇腾) | GPU (H20) |
|------|-----------|-----------|
| text-to-image 1024×1024 | 15.845s | — |
| 峰值显存 | 62,278 MB | — |

> ⚠️ **注意：** 133 GPU 机器 (H20) 在采集期间不可达，GPU 基线数据暂缺。

**后续补充方式：**
1. 待 GPU 机器恢复后，运行 `benchmarks/bench_gpu_baseline.py` 采集同口径数据
2. 对比维度：端到端延迟、单步延迟、峰值显存、吞吐量
3. 补充数据后更新本节表格

---

## 6. 代码质量改进

本次 NPU 适配过程中完成了以下架构改进：

| 改进项 | 变更内容 | 收益 |
|--------|---------|------|
| 提取 AscendLongContextAttention | 独立为 `layers/attention/ascend_long_context.py` | 解耦 NPU 特定逻辑，便于单独维护 |
| 创建统一 platform ops 接口 | `platforms/ops.py` 提供 3 个统一函数 | GPU/NPU 代码路径统一 |
| 创建 attention 工厂函数 | `layers/attention/factory.py` 按设备自动路由 | 消除 transformer 中的硬编码分支 |
| Transformer 去条件分支 | 删除 96 行 NPU `if-else` 分支 → 18 行统一接口调用 | 代码可维护性显著提升 |

**净效果：** 推理逻辑与设备选择完全解耦，新增设备适配只需实现 ops 接口 + attention backend，无需修改模型代码。

---

## 7. 后续优化建议

按预期收益排序：

### 优先级 1：CFG Distillation（理论加速 44%）
- **原理：** 训练无需 negative prompt 的 guidance-free 模型，将每步 2-pass CFG 降为 1-pass
- **预期收益：** 单步从 556ms 降至 ~308ms，端到端从 15.8s 降至 ~8.9s
- **前置条件：** 需要训练蒸馏版模型权重
- **工作量：** 模型训练 + 效果验证

### 优先级 2：Step Reduction 28→14（理论加速 49%）
- **原理：** 一致性蒸馏 (Consistency Distillation) 或 LCM 使模型在更少步数达到同等质量
- **预期收益：** 端到端从 15.8s 降至 ~8.1s
- **前置条件：** 需要专项蒸馏训练
- **工作量：** 蒸馏训练 + 质量评估 + 调度器适配

### 优先级 3：多卡 AllToAll 通信优化
- **原理：** 多卡并行时计算与通信重叠 (overlap)
- **前置条件：** 需要多卡 profiling 数据，确认通信占比
- **当前状态：** 单卡场景无跨设备通信，暂无法评估

### 优先级 4：等待 CANN 版本升级
- **原理：** 后续 CANN/MindIE 版本可能改善 `torch.compile` backend 效果
- **行动项：** 每个大版本发布后重新运行 `benchmarks/bench_compile_ffn.py` 验证

---

## 附录：测试配置

```json
{
  "device": "npu (华为昇腾)",
  "npu_count": 8,
  "attention": "MindIE FlashAttention",
  "dtype": "torch.bfloat16",
  "torch_version": "2.10.0",
  "torch_npu_version": "2.10.0.post4",
  "seed": 42,
  "warmup": 2,
  "timed_runs": 3
}
```
