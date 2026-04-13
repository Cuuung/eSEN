# ESEN Head 层详解：能量、力、应力预测

> **📖 文档导航**
> - **上一篇**: [Hydra 模型架构](./HYDRA_MODEL.md)
> - **返回**: [文档总览](./README.md)
>
> 本文档详细解析 eSEN 模型的三种输出头实现原理。

---

## 1. Head 层概述

Head 层接收 Backbone 提取的 `node_embedding`，将其转换为物理量预测：

```
┌─────────────────────────────────────────────────────────────┐
│                    Backbone 输出                            │
│  node_embedding: [N, (lmax+1)², sphere_channels]          │
│     │                                                         │
│     ├── l=0 (标量): 1 个系数 × sphere_channels              │
│     ├── l=1 (矢量): 3 个系数 × sphere_channels              │
│     ├── l=2 (二阶张量): 5 个系数 × sphere_channels          │
│     └── ...                                                 │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │ Energy Head │      │ Force Head  │      │ Stress Head │
   │ (MLP)       │      │ (SO3_Linear)│      │ (Gradient)  │
   └─────────────┘      └─────────────┘      └─────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
      能量 (scalar)         力 (vector)        应力 (tensor)
```

---

## 2. Head 接口定义

所有 Head 必须实现 `HeadInterface` 接口：

```python
# src/fairchem/core/models/base.py

class HeadInterface(metaclass=ABCMeta):
    @property
    def use_amp(self):
        return False

    @abstractmethod
    def forward(
        self, data: Batch, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Head forward.

        Arguments
        ---------
        data: DataBatch
            原子系统输入
        emb: dict[str->torch.Tensor]
            Backbone 生成的 embedding

        Returns
        -------
        outputs: dict[str->torch.Tensor]
            返回预测的物理量
        """
        return
```

---

## 3. Energy Head（能量预测）

### 3.1 代码实现

```python
# src/fairchem/core/models/esen/esen.py

@registry.register_model("esen_mlp_energy_head")
class MLP_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        
        # 3层 MLP：sphere_channels → hidden → hidden → 1
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        # 1. 提取 l=0 的标量特征（仅取第一个球谐系数）
        # shape: [N, 1, sphere_channels] → [N, sphere_channels]
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        # 2. 初始化能量数组（按系统分批）
        # shape: [num_graphs]
        energy = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        # 3. 聚合每个系统的原子能量
        # index_add_: 按 batch 索引累加
        energy.index_add_(0, data_dict["batch"], node_energy.view(-1))
        
        # 4. 可选：按原子数归一化
        if self.reduce == "sum":
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
```

### 3.2 能量计算流程图

```
node_embedding: [N, 9, 128]  (lmax=2, sphere_channels=128)
       │
       │ narrow(dim=1, start=0, length=1)
       ▼
[atom_energy]: [N, 1, 128]  ← 仅取 l=0 的标量特征
       │
       │ energy_block (MLP)
       ▼
[atom_energy]: [N, 1, 1]    ← 每个原子预测一个能量值
       │
       │ view(-1)
       ▼
[atom_energy]: [N]          ← 展平为原子能量数组
       │
       │ index_add_(batch)
       ▼
[system_energy]: [num_graphs]  ← 按系统聚合
       │
       │ 可选: / natoms
       ▼
最终输出: {"energy": tensor}
```

### 3.3 为什么只取 l=0？

- **l=0** 是标量（球谐函数 Y₀⁰），在旋转下不变
- 能量是标量物理量，应具有旋转不变性
- Backbone 已将原子类型信息编码在 l=0 系数中

---

## 4. Force Head（直接力预测）

### 4.1 SO3_Linear 层

在理解 Force Head 之前，需要先了解 `SO3_Linear`：

```python
# src/fairchem/core/models/esen/nn/so3_layers.py

class SO3_Linear(torch.nn.Module):
    """等变的 SO(3) 线性变换"""
    
    def __init__(self, in_features: int, out_features: int, lmax: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        # 权重形状: [(lmax+1), out_features, in_features]
        # 每层 l 有独立的权重
        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        # expand_index 用于索引：l → 系数位置
        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index)

    def forward(self, input_embedding):
        # input: [batch, (lmax+1)², channels]
        
        # 扩展权重索引
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(lmax+1)², out_features, in_features]

        # 爱因斯坦求和：逐层线性变换
        # input_embedding[b, m, i] * weight[m, o, i] → out[b, m, o]
        out = torch.einsum("bmi, moi -> bmo", input_embedding, weight)
        
        # 偏置只加到 l=0（标量层）
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias
        return out
```

### 4.2 Force Head 代码实现

```python
# src/fairchem/core/models/esen/esen.py

@registry.register_model("esen_linear_force_head")
class Linear_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        # SO3_Linear: 输入 sphere_channels → 输出 1
        # lmax=1: 仅处理 l=0 和 l=1 的球谐系数
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        # 1. 提取 l ≤ 1 的特征
        # shape: [N, 4, sphere_channels] (l=0 有 1 个系数，l=1 有 3 个系数)
        # narrow(dim=1, start=0, length=4) 取前 4 个系数
        forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))
        
        # 2. 提取 l=1 的矢量部分（去掉 l=0 的标量）
        # narrow(dim=1, start=1, length=3) 跳过第 0 个系数
        forces = forces.narrow(1, 1, 3)  # shape: [N, 3, 1]
        
        # 3. 展平为原子力
        forces = forces.view(-1, 3).contiguous()  # shape: [N, 3]
        
        return {"forces": forces}
```

### 4.3 力预测原理

```
力是矢量 (3D)，需要 l=1 的球谐系数来表示：
┌────────────────────────────────────────┐
│ node_embedding[:, :4, :]               │
│ ┌──────┬───────────────────────────────┤
│ │ l=0  │ l=1 (m=-1, 0, +1)             │
│ │ 标量 │ 矢量系数                        │
│ │ 1 个 │ 3 个系数                        │
│ └──────┴───────────────────────────────┘
└────────────────────────────────────────┘
           │
           │ SO3_Linear (lmax=1)
           ▼
┌────────────────────────────────────────┐
│ forces[:, :3]                          │
│ ┌──────────────────────────────────────┤
│ │ 力矢量 (x, y, z)                     │
│ │ 3D 矢量                              │
│ └──────────────────────────────────────┘
└────────────────────────────────────────┘
```

### 4.4 为什么力不需要 l>1？

- 力是**矢量**（一阶张量）
- l=1 的球谐系数足以表示所有 3D 矢量方向
- 高阶系数 (l≥2) 会引入不必要的自由度

### 4.5 直接力预测 vs 梯度力预测

| 方式 | 实现 | 优缺点 |
|------|------|--------|
| **直接预测** | `Linear_Force_Head` | 快，但需额外训练 |
| **梯度计算** | `MLP_EFS_Head` | 利用能量守恒，但需计算梯度 |

---

## 5. EFS Head（能量/力/应力联合预测）

### 5.1 代码实现

```python
# src/fairchem/core/models/esen/esen.py

@registry.register_model("esen_mlp_efs_head")
class MLP_EFS_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        
        # 能量预测 MLP
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )
        
        # 禁用直接力预测，改用梯度计算
        backbone.direct_forces = False

    @conditional_grad(torch.enable_grad())  # 确保梯度计算开启
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = {}

        # ========== 能量预测 ==========
        # 提取 l=0 标量特征
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        # 按系统聚合能量
        energy = torch.zeros(
            len(data["natoms"]), device=data["pos"].device, dtype=node_energy.dtype
        )
        energy.index_add_(0, data["batch"], node_energy.view(-1))
        outputs["energy"] = energy

        # ========== 力和应力计算（梯度法）==========
        if self.regress_stress:
            # 能量对位置和晶胞的梯度
            grads = torch.autograd.grad(
                [energy.sum()],
                [data["pos"], emb["displacement"]],
                create_graph=self.training,  # 训练时保留计算图
            )
            
            # 力 = -∇E (能量对位置的负梯度)
            forces = torch.neg(grads[0])
            
            # 应力计算
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)  # Virial convention
            
            outputs["forces"] = forces
            outputs["stress"] = stress.view(-1, 9)  # 9 个分量
            
            # 恢复原始晶胞
            data["cell"] = emb["orig_cell"]
            
        elif self.regress_forces:
            # 仅预测力：能量对位置的梯度
            forces = -1 * torch.autograd.grad(
                energy.sum(), data["pos"], create_graph=self.training
            )[0]
            outputs["forces"] = forces
            
        return outputs
```

### 5.2 梯度法预测力/应力原理

```
物理定律：力 = 负梯度
┌─────────────────────────────────────────┐
│ F_i = -∂E/∂r_i                         │
│                                         │
│ 原子受力 = 能量对原子位置的负梯度          │
└─────────────────────────────────────────┘

物理定律：应力 = 体积归一化的 Virial
┌─────────────────────────────────────────┐
│ σ = -1/V * ∂E/∂ε                       │
│                                         │
│ ε: 应变张量                             │
│ V: 晶胞体积                             │
└─────────────────────────────────────────┘
```

### 5.3 应力预测的特殊处理

```python
# 在 Backbone 中处理位移扰动
if self.regress_stress and not self.direct_forces:
    # 创建可学习的位移扰动
    displacement = torch.zeros((3, 3), dtype=data_dict["pos"].dtype, device=data_dict["pos"].device)
    displacement.requires_grad = True
    
    # 对称化处理（确保物理一致性）
    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    
    # 应用位移扰动到原子位置
    data_dict["pos"] = data_dict["pos"] + torch.bmm(
        data_dict["pos"].unsqueeze(-2), symmetric_displacement
    ).squeeze(-2)
```

---

## 6. Head 层配置

### 6.1 配置文件示例

```yaml
model:
  name: hydra
  backbone:
    model: esen_backbone
    sphere_channels: 128
    lmax: 3
    mmax: 2
    
  heads:
    # 能量预测头
    energy:
      module: esen_mlp_energy_head
      reduce: sum  # or "mean"
    
    # 直接力预测头
    forces:
      module: esen_linear_force_head
```

### 6.2 不同 Head 组合

```yaml
# 仅预测能量
heads:
  energy:
    module: esen_mlp_energy_head

# 能量 + 直接力
heads:
  energy:
    module: esen_mlp_energy_head
  forces:
    module: esen_linear_force_head

# 能量 + 梯度力 + 应力
heads:
  efs:
    module: esen_mlp_efs_head
```

---

## 7. Head 层数学总结

### 7.1 旋转等变性分析

| Head | 输出类型 | 等变性 | 原因 |
|------|---------|--------|------|
| `MLP_Energy_Head` | 标量 | **旋转不变** | 仅使用 l=0 特征 |
| `Linear_Force_Head` | 矢量 | **旋转等变** | 使用 l=1 特征 + SO3_Linear |
| `MLP_EFS_Head` | 标量/矢量/张量 | 混合 | 能量不变，力/应力等变 |

### 7.2 关键公式

**能量（不变）：**
$$E_{system} = \sum_{i \in system} MLP(f_i^{l=0})$$

**力（等变）：**
$$\vec{F}_i = SO3\_Linear(f_i^{l \leq 1})[:, l=1]$$

**梯度力：**
$$\vec{F}_i = -\frac{\partial E}{\partial \vec{r}_i}$$

---

## 8. 源码文件索引

| 文件 | 内容 |
|------|------|
| `src/fairchem/core/models/esen/esen.py` | Head 类定义 |
| `src/fairchem/core/models/esen/nn/so3_layers.py` | SO3_Linear 层 |
| `src/fairchem/core/models/base.py` | HeadInterface 定义 |
