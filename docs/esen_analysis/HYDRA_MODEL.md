# Hydra 模型架构：快速替换 Backbone 和 Head

> **📖 文档导航**
> - **上一篇**: [ESEN 模型调用流程](./ESEN_CALL_FLOW.md)
> - **下一篇**: [ESEN Head 层详解](./ESEN_HEAD_LAYERS.md)
> - **返回**: [文档总览](./README.md)
>
> 本文档介绍 Hydra 架构、组件替换和自定义开发方法。

---

## 1. 什么是 Hydra 模型？

Hydra 是 fairchem 中的**模块化模型架构**，允许独立配置和替换：
- **Backbone（骨干网络）**：负责特征提取
- **Heads（输出头）**：负责预测具体属性（能量、力、应力等）

```
┌─────────────────────────────────────────┐
│              HydraModel                  │
│  ┌─────────────────────────────────┐    │
│  │         Backbone                │    │
│  │   (eSEN, Equiformer, GemNet...) │    │
│  └─────────────────────────────────┘    │
│                   │                      │
│                   ▼                      │
│           node_embedding                 │
│                   │                      │
│     ┌─────────────┼─────────────┐       │
│     ▼             ▼             ▼       │
│ ┌────────┐  ┌────────┐  ┌────────┐     │
│ │Head 1  │  │Head 2  │  │Head 3  │     │
│ │(energy)│  │(forces)│  │(stress)│     │
│ └────────┘  └────────┘  └────────┘     │
└─────────────────────────────────────────┘
```

---

## 2. 核心代码解析

**文件位置**: `src/fairchem/core/models/base.py`

### 2.1 HydraModel 初始化

```python
@registry.register_model("hydra")
class HydraModel(nn.Module, GraphModelMixin):
    def __init__(
        self,
        backbone: dict | None = None,      # Backbone 配置
        heads: dict | None = None,          # Heads 配置
        finetune_config: dict | None = None, # 微调配置
        otf_graph: bool = True,
        pass_through_head_outputs: bool = False,
        freeze_backbone: bool = False,      # 是否冻结 backbone
    ):
        # 1. 加载 Backbone
        if backbone is not None:
            backbone_model_name = backbone.pop("model")  # 获取模型名称
            self.backbone = registry.get_model_class(backbone_model_name)(**backbone)
        
        # 2. 加载 Heads
        if heads is not None:
            self.output_heads = {}
            for head_name, head_config in heads.items():
                module_name = head_config.pop("module")  # 获取 head 模块名称
                self.output_heads[head_name] = registry.get_model_class(module_name)(
                    self.backbone,  # 传入 backbone 引用
                    **head_config,
                )
            self.output_heads = torch.nn.ModuleDict(self.output_heads)

    def forward(self, data: Batch):
        # 1. Backbone forward
        emb = self.backbone(data)
        
        # 2. 所有 Head forward
        out = {}
        for k in self.output_heads:
            out[k] = self.output_heads[k](data, emb)
        
        return out
```

### 2.2 接口定义

```python
class BackboneInterface(metaclass=ABCMeta):
    """Backbone 必须实现的接口"""
    @abstractmethod
    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        """返回 embedding 字典"""
        return


class HeadInterface(metaclass=ABCMeta):
    """Head 必须实现的接口"""
    @abstractmethod
    def forward(self, data: Batch, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """接收 data 和 embedding，返回预测结果"""
        return
```

---

## 3. 快速替换 Backbone

### 3.1 配置文件方式（推荐）

只需修改 YAML 配置文件中的 `model.backbone.model` 字段：

```yaml
# configs/my_model.yml
model:
  name: hydra
  
  backbone:
    model: esen_backbone    # ← 修改这里替换 backbone
    num_layers: 10
    cutoff: 6.0
    sphere_channels: 128
    lmax: 3
    mmax: 2
    
  heads:
    energy:
      module: esen_mlp_energy_head
    forces:
      module: esen_linear_force_head
```

### 3.2 可用的 Backbone 列表

| Backbone 名称 | 注册名 | 特点 |
|--------------|--------|------|
| eSEN | `esen_backbone` | 最新模型，高性能 |
| EquiformerV2 | `equiformer_v2` | 等变 Transformer |
| GemNet-OC | `gemnet_oc` | 图神经网络 |
| PaiNN | `painn` | 物理感知神经网络 |
| SCN | `scn` | 球谐卷积网络 |

### 3.3 替换示例

**从 eSEN 切换到 EquiformerV2：**

```yaml
# 使用 eSEN
model:
  name: hydra
  backbone:
    model: esen_backbone
    num_layers: 10
    # ...

# 切换到 EquiformerV2
model:
  name: hydra
  backbone:
    model: equiformer_v2
    num_layers: 8
    # ...
```

---

## 4. 快速替换 Head

### 4.1 配置文件方式

修改 `model.heads` 部分的 `module` 字段：

```yaml
model:
  name: hydra
  backbone:
    model: esen_backbone
    # ...
    
  heads:
    energy:
      module: esen_mlp_energy_head    # ← 修改这里替换 head
      # head 特定参数...
    forces:
      module: esen_linear_force_head  # ← 或使用其他 force head
      # head 特定参数...
```

### 4.2 可用的 Head 列表

| Head 类型 | 模块名称 | 用途 |
|----------|---------|------|
| Energy Head | `esen_mlp_energy_head` | 能量预测 |
| Force Head (Direct) | `esen_linear_force_head` | 直接力预测 |
| Force Head (Gradient) | `esen_gradient_force_head` | 梯度力预测 |
| EFS Head | `esen_mlp_efs_head` | 能量/力/应力联合预测 |

### 4.3 添加新的 Head

```yaml
model:
  name: hydra
  backbone:
    model: esen_backbone
    # ...
    
  heads:
    energy:
      module: esen_mlp_energy_head
    forces:
      module: esen_linear_force_head
    stress:                          # ← 添加新的 head
      module: esen_mlp_stress_head
```

---

## 5. 完整配置示例

### 5.1 eSEN 模型配置

```yaml
# configs/eSEN_31M_omt24_direct_force.yml
model:
  name: hydra
  backbone:
    model: esen_backbone
    otf_graph: true
    regress_forces: true
    direct_forces: true
    num_layers: 10
    sphere_channels: 128
    attn_hidden_channels: 64
    num_heads: 8
    attn_dropout: 0.0
    cutoff: 6.0
    use_pbc: true
    max_neighbors: 100
    lmax: 3
    mmax: 2
    distance_expansion: gaussian
    distance_expansion_params:
      num_gaussians: 128
      max_distance: 6.0
      scale: 0.5
    
  heads:
    energy:
      module: esen_mlp_energy_head
      otf_graph: true
      regress_forces: true
      direct_forces: true
      energy_head: mlp
      hidden_channels: 128
      num_layers: 3
      act: silu
      
    forces:
      module: esen_linear_force_head
      otf_graph: true
      regress_forces: true
      direct_forces: true
```

### 5.2 EquiformerV2 模型配置

```yaml
model:
  name: hydra
  backbone:
    model: equiformer_v2
    num_layers: 8
    sphere_channels: 128
    # ... equiformer 特定参数
    
  heads:
    energy:
      module: equiformer_v2_energy_head
    forces:
      module: equiformer_v2_force_head
```

---

## 6. 自定义 Backbone 和 Head

### 6.1 创建自定义 Backbone

```python
# src/fairchem/core/models/my_model/my_backbone.py
from fairchem.core.models.base import BackboneInterface, GraphModelMixin
from fairchem.core.common.registry import registry

@registry.register_model("my_backbone")  # ← 注册名称
class MyBackbone(nn.Module, BackboneInterface, GraphModelMixin):
    def __init__(
        self,
        num_layers: int = 6,
        hidden_channels: int = 256,
        cutoff: float = 6.0,
        **kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        # ... 初始化网络层
    
    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        # 1. 生成图
        graph = self.generate_graph(data)
        
        # 2. 特征提取
        node_embedding = self._extract_features(data, graph)
        
        # 3. 返回 embedding 字典
        return {
            "node_embedding": node_embedding,
            "graph": graph,
        }
```

### 6.2 创建自定义 Head

```python
# src/fairchem/core/models/my_model/my_head.py
from fairchem.core.models.base import HeadInterface
from fairchem.core.common.registry import registry

@registry.register_model("my_energy_head")  # ← 注册名称
class MyEnergyHead(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone,  # 自动传入 backbone 引用
        hidden_channels: int = 128,
        **kwargs
    ):
        super().__init__()
        self.backbone = backbone
        self.energy_mlp = nn.Sequential(
            nn.Linear(backbone.hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
    
    def forward(self, data: Batch, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node_emb = emb["node_embedding"]
        
        # 全局池化
        graph_emb = scatter(node_emb, data.batch, dim=0, reduce="add")
        
        # 能量预测
        energy = self.energy_mlp(graph_emb)
        
        return {"energy": energy.squeeze(-1)}
```

### 6.3 使用自定义组件

```yaml
# configs/my_custom_model.yml
model:
  name: hydra
  backbone:
    model: my_backbone       # ← 使用自定义 backbone
    num_layers: 8
    hidden_channels: 256
    cutoff: 6.0
    
  heads:
    energy:
      module: my_energy_head  # ← 使用自定义 head
      hidden_channels: 128
```

---

## 7. 微调和迁移学习

### 7.1 从检查点微调

```yaml
model:
  name: hydra
  finetune_config:
    starting_checkpoint: checkpoints/pretrained_model.pt
    override:
      # 可选：覆盖 backbone 的某些属性
      max_neighbors: 150
  
  # 可选：指定新的 heads（不指定则使用原模型的 heads）
  heads:
    energy:
      module: esen_mlp_energy_head
    forces:
      module: esen_linear_force_head
```

### 7.2 冻结 Backbone

```yaml
model:
  name: hydra
  freeze_backbone: true  # ← 冻结 backbone 参数
  finetune_config:
    starting_checkpoint: checkpoints/pretrained_model.pt
  
  # 只训练 heads
  heads:
    energy:
      module: esen_mlp_energy_head
```

---

## 8. 命令行快速替换

### 8.1 通过命令行参数覆盖

```bash
# 替换 backbone
python -m fairchem.core._cli \
    --config-yml configs/base.yml \
    --mode train \
    model.backbone.model=esen_backbone \
    model.backbone.num_layers=12

# 替换 head
python -m fairchem.core._cli \
    --config-yml configs/base.yml \
    --mode train \
    model.heads.forces.module=esen_linear_force_head
```

---

## 9. 总结

| 操作 | 修改位置 | 示例 |
|------|---------|------|
| 替换 Backbone | `model.backbone.model` | `esen_backbone` → `equiformer_v2` |
| 替换 Head | `model.heads.<name>.module` | `esen_mlp_energy_head` → `my_energy_head` |
| 添加新 Head | `model.heads.<new_name>` | 添加 `stress` head |
| 冻结 Backbone | `model.freeze_backbone: true` | 迁移学习时使用 |
| 从检查点加载 | `model.finetune_config` | 微调预训练模型 |

**核心优势：**
- ✅ 零代码修改，只需改配置文件
- ✅ 灵活组合不同 backbone 和 head
- ✅ 支持自定义组件扩展
- ✅ 便于迁移学习和微调