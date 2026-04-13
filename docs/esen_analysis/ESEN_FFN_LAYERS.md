# ESEN Nodewise Feed-Forward 层详解

> **📖 文档导航**
> - **上一篇**: [Backbone 等变性实现](./ESEN_BACKBONE_EQUIVARIANCE.md)
> - **下一篇**: [Hydra 模型架构](./HYDRA_MODEL.md)
> - **返回**: [文档总览](./README.md)
>
> 本文档详细解析 eSEN 模型中 Nodewise Feed-Forward (FFN) 的两种实现方式。

---

## 1. Feed-Forward 在 eSEN 中的角色

在 eSEN 的 `eSEN_Block` 中，每个块包含两个子层：

```
┌─────────────────────────────────────────────────────────────┐
│                    eSEN_Block                               │
│                                                             │
│  输入 x                                                      │
│     │                                                       │
│     ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. Edgewise (Message Passing)                       │    │
│  │    - 聚合邻居节点信息                                 │    │
│  │    - 基于 SO2 卷积实现                                │    │
│  └─────────────────────────────────────────────────────┘    │
│     │                                                       │
│     ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 2. Atomwise / Nodewise FFN                           │    │
│  │    - 对每个原子独立进行非线性变换                      │    │
│  │    - 不进行跨原子信息交互                              │    │
│  └─────────────────────────────────────────────────────┘    │
│     │                                                       │
│     ▼                                                       │
│  输出 x                                                     │
└─────────────────────────────────────────────────────────────┘
```

**核心思想**：Nodewise Feed-Forward 只对单个原子的特征进行非线性变换，不需要像注意力机制那样计算原子间的交互。

---

## 2. 两种 FFN 实现方式

eSEN 提供了两种 Nodewise FFN 实现：

| 类型 | 类名 | 实现方式 | 等变性 |
|------|------|----------|--------|
| **Spectral** | `SpectralAtomwise` | SO(3) 线性变换 + 门控激活 | ✅ 等变 |
| **Grid** | `GridAtomwise` | 网格投影 + 点态 MLP | ✅ 等变 |

---

## 3. SpectralAtomwise（频域 FFN）

### 3.1 核心思想

Spectral 方法在球谐函数的频域（spectral domain）进行操作：

1. **门控机制**：使用 l=0 的标量特征生成门控信号
2. **SO(3) 线性变换**：对所有球谐系数进行线性投影
3. **门控激活**：用门控信号控制高阶特征的激活

### 3.2 代码实现

```python
# src/fairchem/core/models/esen/esen_block.py

class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        # 门控 MLP：从 l=0 标量特征生成门控信号
        # 输出维度: lmax * hidden_channels
        # 每个 l (1, 2, ..., lmax) 对应 hidden_channels 个门控值
        self.scalar_mlp = nn.Sequential(
            nn.Linear(
                self.sphere_channels,
                self.lmax * self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(),
        )

        # SO(3) 线性变换 1: sphere_channels → hidden_channels
        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels, self.hidden_channels, lmax=self.lmax
        )
        
        # 门控激活层
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        
        # SO(3) 线性变换 2: hidden_channels → sphere_channels
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        # 1. 从 l=0 提取门控标量
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))  # shape: [N, 1, sphere_channels]
        
        # 2. SO(3) 线性变换
        x = self.so3_linear_1(x)
        
        # 3. 门控激活
        x = self.act(gating_scalars, x)
        
        # 4. 残差连接前的变换
        return self.so3_linear_2(x)
```

### 3.3 数据流图解

```
输入 x: [N, (lmax+1)², sphere_channels]
   │
   │ narrow(dim=1, start=0, length=1)  ← 取 l=0
   ▼
gate_x: [N, 1, sphere_channels]
   │
   │ scalar_mlp (Linear + SiLU)
   ▼
gating_scalars: [N, 1, lmax * hidden_channels]
   │ 分割为 l=1,2,...,lmax 个门控向量
   ▼
┌───────────────────────────────────────────────┐
│ gate_l=1: [N, 1, hidden_channels] 控制 l=1   │
│ gate_l=2: [N, 1, hidden_channels] 控制 l=2   │
│ ...                                           │
│ gate_l=lmax: [N, 1, hidden_channels] 控制 lmax │
└───────────────────────────────────────────────┘

输入 x: [N, (lmax+1)², sphere_channels]
   │
   │ so3_linear_1
   ▼
x: [N, (lmax+1)², hidden_channels]
   │
   │ act(gating_scalars, x) - 门控激活
   ▼
x: [N, (lmax+1)², hidden_channels]  ← 各阶特征被门控调制
   │
   │ so3_linear_2
   ▼
输出: [N, (lmax+1)², sphere_channels]
```

### 3.4 门控机制详解

GateActivation 的工作原理：

```
┌─────────────────────────────────────────────────────────────┐
│                    GateActivation                            │
│                                                             │
│  输入:                                                        │
│    - gating_scalars: [N, 1, lmax * hidden_channels]          │
│    - x: [N, (lmax+1)², hidden_channels]                      │
│                                                             │
│  分解 gating_scalars:                                        │
│    gate_l=1: 取第 [0:hidden_channels]                        │
│    gate_l=2: 取第 [hidden_channels:2*hidden_channels]         │
│    ...                                                       │
│                                                             │
│  对每个 l (l=1 到 lmax):                                     │
│    取 x[:, l²:(l+1)², :] 的系数                              │
│    与 gate_l 进行逐元素乘法                                   │
│                                                             │
│  结果: l=0 保持不变, l>0 的系数被门控调制                      │
│                                                             │
│  输出: [N, (lmax+1)², hidden_channels]                        │
└─────────────────────────────────────────────────────────────┘
```

**关键特性**：
- **l=0 保持不变**：标量特征不经过门控，保持旋转不变性
- **l>0 被门控调制**：高阶矢量特征通过门控实现非线性变换
- **每阶独立门控**：不同角动量 l 的特征有独立的门控信号

---

## 4. GridAtomwise（网格 FFN）

### 4.1 核心思想

Grid 方法先将球谐系数投影到 3D 网格空间：

1. **球面 → 网格投影**：使用 SO3_grid 将球谐系数映射到离散网格点
2. **点态 MLP**：在网格上进行标准的点对点非线性变换
3. **网格 → 球面反投影**：将变换后的网格值投影回球谐系数

### 4.2 代码实现

```python
# src/fairchem/core/models/esen/esen_block.py

class GridAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        # 标准点态 MLP：sphere_channels → hidden → hidden → sphere_channels
        self.grid_mlp = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.sphere_channels, bias=False),
        )

    def forward(self, x):
        # 1. 球面 → 网格投影
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        
        # 2. 点态 MLP（在网格上进行）
        x_grid = self.grid_mlp(x_grid)
        
        # 3. 网格 → 球面反投影
        return self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)
```

### 4.3 数据流图解

```
输入 x (球谐系数): [N, (lmax+1)², sphere_channels]
   │
   │ to_grid(lmax, lmax)
   ▼
x_grid (3D 网格): [N, num_grid_points, sphere_channels]
   │
   │ grid_mlp (3层 Linear + SiLU)
   ▼
x_grid: [N, num_grid_points, sphere_channels]
   │
   │ from_grid(lmax, lmax)
   ▼
输出 x (球谐系数): [N, (lmax+1)², sphere_channels]
```

**关键特性**：
- 在 3D 空间中进行变换，利用了 GPU 对点态操作的优化
- 使用 bias=False 保持等变性（线性变换不破坏等变性）
- 网格投影/反投影过程本身是等变的

---

## 5. 两种方法的对比

### 5.1 特性对比

| 特性 | SpectralAtomwise | GridAtomwise |
|------|------------------|--------------|
| **计算方式** | 频域门控 | 空间网格变换 |
| **非线性** | 门控激活 (SiLU) | 点态 MLP (SiLU) |
| **参数量** | 较少 | 较多 |
| **计算效率** | 较高 | 依赖网格大小 |
| **表达能力** | 受限（门控形式） | 更强（完整 MLP） |
| **等变性保证** | ✅ 理论保证 | ✅ 投影/反投影等变 |

### 5.2 等变性分析

**SpectralAtomwise**：
- l=0 (标量) 保持不变 → 旋转不变性 ✓
- l>0 特征通过 SO3_Linear 变换 → 旋转等变性 ✓
- 门控机制只做逐元素乘法，不改变空间变换性质 ✓

**GridAtomwise**：
- 球面 → 网格投影是等变操作 ✓
- 点态 MLP 使用无偏置线性层 + SiLU 激活 ✓
- 网格 → 球面反投影是等变操作 ✓

### 5.3 配置选择

```yaml
# 在 eSEN backbone 配置中选择 FFN 类型
model:
  backbone:
    model: esen_backbone
    mlp_type: spectral  # 或 "grid"
    # ...
```

**建议**：
- `spectral`：计算效率优先时使用
- `grid`：表达能力优先时使用

---

## 6. 在 eSEN_Block 中的使用

### 6.1 Block 结构

```python
# src/fairchem/core/models/esen/esen_block.py

class eSEN_Block(torch.nn.Module):
    def __init__(self, ..., mlp_type: str, ...) -> None:
        # ...
        
        # Edge-wise (Message Passing)
        self.edge_wise = Edgewise(...)
        
        # Atom-wise FFN（可选择 spectral 或 grid）
        if mlp_type == "spectral":
            self.atom_wise = SpectralAtomwise(...)
        elif mlp_type == "grid":
            self.atom_wise = GridAtomwise(...)
    
    def forward(self, x, ...):
        # 1. 残差连接前的 Norm
        x_res = x
        x = self.norm_1(x)
        
        # 2. Edge-wise Message Passing
        x = self.edge_wise(x, ...)
        x = x + x_res  # 残差连接
        
        # 3. 再次 Norm
        x_res = x
        x = self.norm_2(x)
        
        # 4. Atom-wise FFN
        x = self.atom_wise(x)
        return x + x_res  # 残差连接
```

### 6.2 完整数据流

```
输入 h^(l)
   │
   │ LayerNorm
   ▼
   │
   │ Edgewise (SO2 Conv + 邻居聚合)
   ▼
   │
   │ + h^(l)  (残差)
   ▼
   │
   │ LayerNorm
   ▼
   │
   │ Atomwise (Spectral 或 Grid FFN)
   ▼
   │
   │ + h^(l)  (残差)
   ▼
输出 h^(l+1)
```

---

## 7. 与标准 Transformer FFN 的对比

### 7.1 标准 Transformer FFN

```python
# 标准 FFN (如 BERT, GPT)
ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model),
)
```

**特点**：
- 点态操作（每个 token 独立）
- 两层线性变换 + 非线性激活
- 无空间结构假设

### 7.2 eSEN Nodewise FFN

**SpectralAtomwise**：
- SO(3) 等变的频域变换
- 门控机制实现非线性
- 保持旋转等变性

**GridAtomwise**：
- 网格空间中的点态 MLP
- 完整的非线性表达能力
- 保持旋转等变性

### 7.3 核心区别

| 方面 | 标准 FFN | eSEN Spectral | eSEN Grid |
|------|----------|---------------|-----------|
| **等变性** | 无 | SO(3) 等变 | SO(3) 等变 |
| **非线性** | ReLU/GeLU | 门控激活 | 点态 MLP |
| **空间结构** | 忽略 | 球谐分解 | 3D 网格 |
| **适用场景** | NLP | 3D 分子/材料 | 3D 分子/材料 |

---

## 8. 源码文件索引

| 文件 | 内容 |
|------|------|
| `src/fairchem/core/models/esen/esen_block.py` | SpectralAtomwise, GridAtomwise, eSEN_Block 定义 |
| `src/fairchem/core/models/esen/nn/so3_layers.py` | SO3_Linear 层 |
| `src/fairchem/core/models/esen/nn/activation.py` | GateActivation |
| `src/fairchem/core/models/esen/nn/so3_grid.py` | SO3Grid, to_grid, from_grid |
