# ESEN 模型调用流程分析

> **📖 文档导航**
> - **上一篇**: [CLI 入口点详解](./CLI_ENTRY_POINT.md)
> - **下一篇**: [Hydra 模型架构](./HYDRA_MODEL.md)
>
> 本文档介绍模型架构、数据流和训练流程。

---

## 1. 整体架构概览

```
配置文件 (YAML) → CLI入口 → Trainer → Model (Backbone + Heads) → 输出
```

## 2. 详细调用流程

### 第一步：配置加载 (`configs/eSEN_31M_omt24_direct_force.yml`)

配置文件定义了：
- `trainer: mlip_trainer` - 使用 MLIPTrainer
- `model.backbone.model: esen_backbone` - 使用 eSEN_Backbone 模型
- `model.heads` - 定义 energy/forces/stress 输出头
- `dataset` - 训练/验证数据集路径
- `optim` - 优化器、学习率调度等参数

### 第二步：CLI 入口 (`src/fairchem/core/_cli.py`)

```python
main() 
  → build_config(args, override_args)  # 加载 YAML 配置
  → Runner()(config)  # 启动训练
    → new_trainer_context(config)  # 创建 trainer 上下文
      → task.setup(trainer)  # 设置任务
      → task.run()  # 运行训练
```

### 第三步：Trainer 初始化 (`src/fairchem/core/models/esen/trainers/trainer.py`)

`MLIPTrainer` 继承自 `OCPTrainer`：
- 加载数据集 (`train_loader`, `val_loader`)
- 初始化模型 (`self.model`)
- 设置优化器和调度器
- 定义损失函数和评估指标

### 第四步：模型构建 (`src/fairchem/core/models/esen/esen.py`)

模型由两部分组成：

#### 1. Backbone: `eSEN_Backbone` (注册为 `"esen_backbone"`)

```python
__init__():
  - sphere_embedding: 原子类型嵌入
  - distance_expansion: 距离编码 (Gaussian/Bessel)
  - edge_degree_embedding: 边度嵌入
  - blocks: eSEN_Block 层 (num_layers层)
  - norm: 归一化层

forward(data_dict):
  1. 生成图结构 (边、距离)
  2. 计算旋转矩阵和 Wigner D 矩阵
  3. 初始化节点嵌入
  4. 通过多层 eSEN_Block 更新特征
  5. 返回 node_embedding
```

#### 2. Heads: 输出头

- `MLP_Energy_Head` (注册为 `"esen_mlp_energy_head"`) - 能量预测
- `Linear_Force_Head` (注册为 `"esen_linear_force_head"`) - 力预测 (direct_forces=True)
- `MLP_EFS_Head` (注册为 `"esen_mlp_efs_head"`) - 能量/力/应力联合预测

### 第五步：训练循环 (`MLIPTrainer.train()`)

```python
for epoch in range(max_epochs):
    for batch in train_loader:
        # 1. Forward
        out = self._forward(batch)
        
        # 2. Compute loss
        loss = self._compute_loss(out, batch)
        
        # 3. Backward
        loss.backward()
        optimizer.step()
        
        # 4. 验证
        if step % eval_every == 0:
            val_metrics = self.validate()
```

### 第六步：Forward 流程

```python
# 在 OCPTrainer._forward() 中
def _forward(self, batch):
    # 1. Backbone forward
    emb = self.model.backbone(batch)  # 返回 node_embedding
    
    # 2. 各 Head forward
    outputs = {}
    for head_name, head in self.model.output_heads.items():
        outputs.update(head(batch, emb))
    
    return outputs  # {"energy": ..., "forces": ..., "stress": ...}
```

## 3. 数据流图

```
输入数据 (batch)
    │
    ├── atomic_numbers ──→ sphere_embedding ──┐
    │                                          │
    ├── pos ──→ radius_graph_pbc ──→ edge_index, edge_distance
    │                              │
    │                              └─→ distance_expansion ──→ x_edge
    │                                                        │
    └────────────────────────────────────────────────────────┘
                              │
                              ▼
                    edge_degree_embedding
                              │
                              ▼
                    ┌─────────────────┐
                    │  eSEN_Block × N │ (消息传递、特征更新)
                    └─────────────────┘
                              │
                              ▼
                    node_embedding (球谐特征)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Energy_Head     Force_Head      Stress_Head
              │               │               │
              ▼               ▼               ▼
          energy          forces          stress
```

## 4. 关键文件总结

| 文件 | 作用 |
|------|------|
| `configs/eSEN_31M_omt24_direct_force.yml` | 模型配置 |
| `src/fairchem/core/_cli.py` | CLI 入口 |
| `src/fairchem/core/common/utils.py` | 配置加载、上下文管理 |
| `src/fairchem/core/models/esen/esen.py` | Backbone 和 Head 定义 |
| `src/fairchem/core/models/esen/esen_block.py` | eSEN 消息传递块 |
| `src/fairchem/core/models/esen/trainers/trainer.py` | MLIPTrainer |
| `src/fairchem/core/trainers/ocp_trainer.py` | 基础 Trainer |

## 5. 运行命令示例

```bash
python -m fairchem.core._cli --config-yml configs/eSEN_31M_omt24_direct_force.yml --mode train
```

## 6. 核心类继承关系

```
BaseTrainer (抽象基类)
    │
    └── OCPTrainer
            │
            ├── MLIPTrainer (注册为 "mlip_trainer")
            │       │
            │       └── MLIP_DeNS_Trainer (注册为 "mlip_dens_trainer")
            │
            └── EquiformerV2ForcesTrainer
```

## 7. 模型注册机制

fairchem 使用注册模式 (Registry Pattern) 管理模型、训练器等组件：

```python
# 注册模型
@registry.register_model("esen_backbone")
class eSEN_Backbone(nn.Module, GraphModelMixin):
    ...

@registry.register_model("esen_mlp_energy_head")
class MLP_Energy_Head(nn.Module, HeadInterface):
    ...

@registry.register_model("esen_linear_force_head")
class Linear_Force_Head(nn.Module, HeadInterface):
    ...

# 注册训练器
@registry.register_trainer("mlip_trainer")
class MLIPTrainer(OCPTrainer):
    ...
```

## 8. 配置文件详解

```yaml
# 训练器类型
trainer: mlip_trainer

# 任务配置
task:
  strict_load: False

# 数据集配置
dataset:
  train:
    format: aselmdb  # 数据格式
    src: /path/to/training/data
    transforms:
      normalizer:
        energy: {mean: ..., stdev: ...}
        forces: {mean: ..., stdev: ...}
  val:
    format: aselmdb
    src: /path/to/validation/data

# 模型配置
model:
  name: hydra  # 使用 Hydra 模型架构
  backbone:
    model: esen_backbone
    num_layers: 10
    cutoff: 6.0
    sphere_channels: 128
    lmax: 3
    mmax: 2
    direct_forces: True
    ...
  heads:
    energy:
      module: esen_mlp_energy_head
    forces:
      module: esen_linear_force_head

# 优化器配置
optim:
  batch_size: 32
  lr_initial: 0.0008
  optimizer: AdamW
  scheduler: LambdaLR
  max_epochs: 60