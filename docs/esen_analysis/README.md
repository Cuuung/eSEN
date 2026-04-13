# ESEN 模型分析文档

## 📚 文档导航

本系列文档详细介绍了 fairchem 框架中 ESEN 模型的完整调用流程和 Hydra 架构。建议按以下顺序阅读：

| 序号 | 文档 | 内容概要 |
|------|------|----------|
| 1 | [CLI 入口点详解](./CLI_ENTRY_POINT.md) | 程序入口、配置加载、任务调度 |
| 2 | [ESEN 模型调用流程](./ESEN_CALL_FLOW.md) | 模型架构、数据流、训练流程 |
| 3 | [Hydra 模型架构](./HYDRA_MODEL.md) | Backbone/Head 替换、自定义组件 |
| 4 | [ESEN Head 层详解](./ESEN_HEAD_LAYERS.md) | Energy/Force/Stress 预测原理 |
| 5 | [Backbone 等变性实现](./ESEN_BACKBONE_EQUIVARIANCE.md) | 旋转矩阵、Wigner D、SO(2) 卷积、等变性保证 |
| 6 | [Nodewise FFN 层详解](./ESEN_FFN_LAYERS.md) | Spectral/Grid FFN、门控激活、网格投影 |
| 7 | [数据处理与 Embedding 层](./DATA_EMBEDDING.md) | 动态建边、PBC 处理、PyG Data/Batch、LMDB 数据加载 |
| 8 | [分布式训练详解](./DISTRIBUTED_TRAINING.md) | DDP 原理、torchrun 接口、SLURM 调度 |
| 9 | **[完整训练流程（结合代码）](./ESEN_TRAINING_PIPELINE.md)** | 命令行启动→train组装→Hydra model组装→数据输入，全链路代码级分析 |
| 10 | **[SO(2) 层详解](./SO2_LAYER.md)** | CoefficientMapping、SO2_m_Conv、SO2_Convolution、径向权重、等变性链路 |

---

## 📖 阅读指南

### 快速入门路径

```
命令行调用 (CLI_ENTRY_POINT.md)
        │
        ▼
    配置加载 → Runner → Trainer 创建
        │
        ▼
模型初始化 (ESEN_CALL_FLOW.md)
        │
        ▼
    Backbone → Heads → 训练循环
        │
        ▼
组件替换 (HYDRA_MODEL.md)
        │
        ▼
    修改配置文件 → 切换模型架构
```

### 各文档核心内容

#### 1️⃣ CLI 入口点详解 ([CLI_ENTRY_POINT.md](./CLI_ENTRY_POINT.md))

**核心问题：程序是如何启动的？**

- `main()` 函数解析命令行参数
- `build_config()` 加载 YAML 配置
- `Runner()` 创建训练器上下文
- `task.setup()` 和 `task.run()` 执行具体任务

**关键调用链：**
```python
main() → build_config() → Runner() → new_trainer_context() → task.run()
```

---

#### 2️⃣ ESEN 模型调用流程 ([ESEN_CALL_FLOW.md](./ESEN_CALL_FLOW.md))

**核心问题：模型是如何工作的？**

- 配置文件定义模型参数
- `eSEN_Backbone` 提取原子特征
- 输出头预测能量、力、应力
- `MLIPTrainer` 管理训练循环

**数据流：**
```
输入数据 → 图构建 → Backbone 特征提取 → Heads 预测 → 输出
```

---

#### 3️⃣ Hydra 模型架构 ([HYDRA_MODEL.md](./HYDRA_MODEL.md))

**核心问题：如何灵活替换模型组件？**

- Hydra 架构：Backbone + Heads 模块化设计
- 通过配置文件快速替换组件
- 自定义 Backbone 和 Head 的方法
- 迁移学习和微调配置

**核心优势：**
- ✅ 零代码修改，只需改配置文件
- ✅ 灵活组合不同 backbone 和 head

---

## 🔗 文档关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI_ENTRY_POINT.md                            │
│                    (程序入口与调度)                                │
│                                                                  │
│   main() → build_config() → Runner → new_trainer_context()      │
│                                              │                   │
└──────────────────────────────────────────────│──────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ESEN_CALL_FLOW.md                             │
│                    (模型架构与训练)                                │
│                                                                  │
│   Trainer.initialize() → Model(backbone + heads) → train()      │
│                              │                  │                │
└──────────────────────────────│──────────────────│────────────────┘
                               │                  │
              ┌────────────────┘                  └────────────────┐
              │                                                    │
              ▼                                                    ▼
┌──────────────────────────────┐      ┌──────────────────────────────┐
│      Backbone 实现            │      │      Heads 实现              │
│  (esen_backbone, ...)        │      │  (energy_head, force_head)   │
└──────────────────────────────┘      └──────────────────────────────┘
              │                                    │
              └────────────────┬───────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HYDRA_MODEL.md                                │
│                    (组件替换与扩展)                                │
│                                                                  │
│   修改配置 → 替换 backbone/head → 自定义组件 → 迁移学习            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 运行训练

```bash
python -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode train \
    --num-gpus 4
```

### 替换 Backbone

```yaml
# 修改配置文件
model:
  name: hydra
  backbone:
    model: esen_backbone  # 改为 equiformer_v2, gemnet_oc 等
```

### 替换 Head

```yaml
model:
  heads:
    forces:
      module: esen_linear_force_head  # 改为其他 force head
```

---

## 📁 相关源码文件

| 功能 | 文件路径 |
|------|----------|
| CLI 入口 | `src/fairchem/core/_cli.py` |
| 配置加载 | `src/fairchem/core/common/utils.py` |
| Hydra 模型 | `src/fairchem/core/models/base.py` |
| eSEN Backbone | `src/fairchem/core/models/esen/esen.py` |
| eSEN Block | `src/fairchem/core/models/esen/esen_block.py` |
| MLIP Trainer | `src/fairchem/core/models/esen/trainers/trainer.py` |
| Task 类 | `src/fairchem/core/tasks/task.py` |
| Registry | `src/fairchem/core/common/registry.py` |