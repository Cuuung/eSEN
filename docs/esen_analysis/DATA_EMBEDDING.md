# eSEN 数据处理与 Embedding 层详解

> **📖 文档导航**
> - **上一篇**: [Hydra 模型架构](./HYDRA_MODEL.md)
> - **所属系列**: [文档导航](./README.md)
>
> 本文档详细介绍 eSEN 模型在进入 Backbone 前的数据处理流程，包括动态建边（OTF graph）、PBC 处理、距离 embedding、原子 embedding 和等变的初始节点特征构建。

---

## 0. Data 与 Batch 数据结构详解

### 0.0 PyG (PyTorch Geometric) 简介

**PyG (PyTorch Geometric)** 是一个基于 PyTorch 的几何深度学习扩展库，专门用于处理图结构数据。

**官网**: https://pytorch-geometric.readthedocs.io/

**核心组件**:

| 组件 | 说明 |
|------|------|
| `Data` | 单个图的数据结构（节点、边、特征） |
| `Batch` | 多个图的批量结构，自动管理图间索引 |
| `Dataset` | 数据集抽象，支持从文件加载图数据 |
| `DataLoader` | 批量加载器 |
| `MessagePassing` | 消息传递基类（简化 GNN 实现） |

**在 fairchem 中的使用**:

```python
# fairchem 使用 PyG 的核心类
from torch_geometric.data import Data, Batch

# Data: 表示单个原子系统（分子或晶体）
data = Data(
    atomic_numbers=torch.tensor([6, 1, 1, 1]),  # C, H, H, H
    pos=torch.randn(4, 3),                       # 4个原子的3D坐标
    edge_index=torch.tensor([[0, 1], [1, 0]]),   # 边连接
)

# Batch: 表示多个原子系统合并（用于批量处理）
batch = Batch.from_data_list([data1, data2])
# 自动生成 batch 字段标记每个原子属于哪个系统
```

**为什么使用 PyG**:
- 原生支持图神经网络的各种操作（消息传递、池化等）
- 与 PyTorch 完全兼容
- 提供高效的图数据加载和批处理
- 内置丰富的图转换操作

---

### 0.1 单条数据（Data）的结构

单条原子系统数据继承自 `torch_geometric.data.Data`，包含以下核心字段：

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `atomic_numbers` | LongTensor | `[N]` | 每个原子的原子序数（如 H=1, C=6, O=8） |
| `pos` | FloatTensor | `[N, 3]` | 每个原子的 3D 坐标（笛卡尔坐标系） |
| `cell` | FloatTensor | `[3, 3]` | 晶胞基矢：3 个向量，每个 3 维（a₁, a₂, a₃） |
| `batch` | LongTensor | `[N]` | 每个原子属于哪个结构（用于 batch 区分） |
| `natoms` | LongTensor | `[B]` | 每个结构的原子数（B = batch 中的结构数） |

**示例：一个包含 2 个结构的 batch**
```
结构 0: 5 个原子 (C, H, H, H, H)  → batch = [0,0,0,0,0]
结构 1: 3 个原子 (O, H, H)         → batch = [1,1,1]

atomic_numbers: [6, 1, 1, 1, 1, 8, 1, 1]
pos:           [8个原子的3D坐标]
batch:         [0, 0, 0, 0, 0, 1, 1, 1]
natoms:        [5, 3]
```

### 0.2 Batch 的概念与作用

**为什么需要 Batch？**

分子动力学模拟中，系统通常包含多个独立的原子结构（分子或晶体）。批量处理（batch）可以：
- 提高 GPU 利用率（并行计算多个结构）
- 共享计算（建图、距离计算等）

**Batch 的数据结构：**

```
┌─────────────────────────────────────────────────────────┐
│                     Batch (多个 Data 合并)                │
│                                                         │
│  Data 0          Data 1          Data 2                  │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐           │
│  │ N₀ 个原子│     │ N₁ 个原子│     │ N₂ 个原子│           │
│  │ batch=0 │     │ batch=1 │     │ batch=2 │           │
│  └────┬────┘     └────┬────┘     └────┬────┘           │
│       │               │               │                 │
│       └───────────────┼───────────────┘                 │
│                       ▼                                   │
│              concat 后的数组                              │
│                                                         │
│  atomic_numbers: [N₀+N₁+N₂]  ← 原子序数拼接             │
│  pos:           [N₀+N₁+N₂, 3] ← 坐标拼接                 │
│  batch:         [0×N₀, 1×N₁, 2×N₂] ← 标记所属结构       │
│  natoms:        [N₀, N₁, N₂]  ← 每个结构的原子数        │
└─────────────────────────────────────────────────────────┘
```

### 0.3 边的表示（Edge Index）

边的定义使用 PyG 的 `edge_index` 格式：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `edge_index[0]` (src) | LongTensor | `[E]` | 源节点索引 |
| `edge_index[1]` (tgt) | LongTensor | `[E]` | 目标节点索引 |

```
edge_index 示例（4个原子，E=5条边）：
┌──────────────────────────────────────┐
│ edge_index = [[0, 1, 2, 2, 3],      │  ← src: 边的起点
│                [1, 0, 0, 3, 2]]      │  ← tgt: 边的终点
│ 说明：                                │
│   边0: 0 → 1                         │
│   边1: 1 → 0  (双向边)               │
│   边2: 2 → 0                         │
│   边3: 2 → 3                         │
│   边4: 3 → 2                          │
└──────────────────────────────────────┘
```

### 0.4 周期性边界条件（PBC）的边表示

对于晶体系统，边需要记录跨越晶胞的偏移：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `edge_index` | LongTensor | `[2, E]` | 边的端点索引 |
| `cell_offsets` | LongTensor | `[E, 3]` | 每个边跨越的晶胞偏移 (Δn₁, Δn₂, Δn₃) |

```
PBC 边示例：
原子 i 在原胞 (0,0,0)，原子 j 在相邻 image cell (+1,0,-1)

edge_index = [[j], [i]]  ← j→i 的边
cell_offsets = [[1, 0, -1]]  ← j 在 (+1,0,-1) 个晶胞偏移处

真实位置计算：
pos_j_real = pos_j + cell_offsets @ cell  # = pos_j + 1*a₁ + 0*a₂ + (-1)*a₃
```

### 0.5 从 LMDB 读取到 Batch 的完整调用链

数据从 LMDB 到模型输入的完整流程如下：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          1. Dataset 层                                   │
└─────────────────────────────────────────────────────────────────────────┘

LmdbDataset (src/fairchem/core/datasets/lmdb_dataset.py)
      │
      │ __getitem__(idx)
      │
      ├─ 从 LMDB 读取 pickled bytes
      │     │
      │     │ pickle.loads()
      │     ▼
      │   原始 Dict (键值如: atomic_numbers, pos, cell, y, force, ...)
      │
      ├─ pyg2_data_transform()  ← 转换为 PyG Data 格式
      │     │
      │     ▼
      │   Data 对象 (torch_geometric.data.Data)
      │
      └─ transforms()  ← 应用数据增强/标准化
            │
            ▼
          Data 对象 (增强后)

┌─────────────────────────────────────────────────────────────────────────┐
│                          2. DataLoader 层                                │
└─────────────────────────────────────────────────────────────────────────┘

DataLoader (PyTorch)
      │
      │ 迭代时调用 collate_fn
      │
      ▼
data_list_collater() (src/fairchem/core/datasets/lmdb_dataset.py:174-194)
      │
      │ batch_list = [Data₁, Data₂, ..., Dataₙ]  (n = batch_size)
      │
      ├─ Batch.from_data_list(batch_list)
      │     │
      │     ├─ concat 所有 Data 的 tensor 字段
      │     ├─ 自动生成 batch 字段：标记每个原子属于哪个 Data
      │     └─ 自动生成 ptr 字段：每个 Data 的边界索引
      │
      ├─ (可选) 读取预计算的 edge_index + neighbors
      │     └─ batch.neighbors = [...]
      │
      └─ 返回: Batch 对象

┌─────────────────────────────────────────────────────────────────────────┐
│                          3. Trainer 层                                   │
└─────────────────────────────────────────────────────────────────────────┘

BaseTrainer.load_datasets() (src/fairchem/core/trainers/base_trainer.py:341-344)
      │
      ├─ collater = partial(data_list_collater, otf_graph=True/False)
      │
      ├─ create_dataset() → LmdbDataset
      │
      └─ get_dataloader() → DataLoader(dataset, collate_fn=collater)

┌─────────────────────────────────────────────────────────────────────────┐
│                          4. 训练循环                                      │
└─────────────────────────────────────────────────────────────────────────┘

for batch in train_loader:
      │
      │ batch = Batch 对象
      │
      ├─ batch.pos         [N, 3]        ← 原子坐标
      ├─ batch.cell        [B, 3, 3]     ← 晶胞向量
      ├─ batch.atomic_numbers [N]         ← 原子序数
      ├─ batch.batch       [N]           ← 所属结构标记
      ├─ batch.natoms      [B]           ← 每结构原子数
      ├─ batch.y           [B]           ← 能量标签
      ├─ batch.force       [N, 3]        ← 力标签
      └─ batch.edge_index  [2, E]       ← 边索引（可选）
            │
            ▼
      model(batch)  →  模型前向传播
```

### 0.5.1 LmdbDataset 的关键代码

```python
# src/fairchem/core/datasets/lmdb_dataset.py

class LmdbDataset(BaseDataset):
    def __getitem__(self, idx):
        # 1. 从 LMDB 读取序列化数据
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        
        # 2. pickle 反序列化
        data_dict = pickle.loads(datapoint_pickled)
        
        # 3. 转换为 PyG Data 对象
        data_object = pyg2_data_transform(data_dict)
        
        # 4. 应用数据增强/变换
        data_object = self.transforms(data_object)
        
        return data_object  # 类型: torch_geometric.data.Data
```

### 0.5.2 data_list_collater 的关键代码

```python
# src/fairchem/core/datasets/lmdb_dataset.py

def data_list_collater(
    data_list: list[BaseData],  # 来自 Dataset.__getitem__() 的 n 个 Data
    otf_graph: bool = False,
    to_dict: bool = False
) -> BaseData | dict[str, torch.Tensor]:
    
    # 核心: PyG 的 Batch.from_data_list() 自动完成:
    #   - concat 所有 tensor 字段 (沿 dim=0)
    #   - 生成 batch 字段 (标记每个原子属于哪个原始 Data)
    #   - 生成 ptr 字段 (边界索引)
    batch = Batch.from_data_list(data_list)
    
    # 如果不是 OTF 建图，读取预计算的邻居数
    if not otf_graph:
        try:
            n_neighbors = []
            for data in data_list:
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            logging.warning("LMDB does not contain edge index...")
    
    return batch
```

### 0.5.3 BaseTrainer 中的配置

```python
# src/fairchem/core/trainers/base_trainer.py

def load_datasets(self) -> None:
    # 1. 创建 collater (使用 partial 绑定 otf_graph 参数)
    self.collater = partial(
        data_list_collater, 
        otf_graph=self.config["model"].get("otf_graph", False)
    )
    
    # 2. 创建 dataset
    self.train_dataset = create_dataset(config, "train")
    
    # 3. 创建 sampler (处理 batch 内各结构原子数不同的填充问题)
    self.train_sampler = self.get_sampler(self.train_dataset, batch_size, shuffle=True)
    
    # 4. 创建 DataLoader
    self.train_loader = DataLoader(
        self.train_dataset,
        collate_fn=self.collater,    # ← 关键: 每次迭代调用 data_list_collater
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=self.train_sampler,
    )
```

### 0.5.4 Batch.from_data_list() 的自动处理

PyG 的 `Batch.from_data_list()` 会自动：

| 操作 | 结果 |
|------|------|
| concat tensor 字段 | `pos: [N₀,3], [N₁,3] → [N₀+N₁, 3]` |
| 生成 batch 索引 | `[0×N₀, 1×N₁, 2×N₂, ...]` |
| 生成 ptr 边界 | `[0, N₀, N₀+N₁, N₀+N₁+N₂, ...]` |
| 复制标量字段 | `y, natoms` 变成 `[B]` |
| 忽略缺失字段 | 自动处理不同 Data 的字段差异 |

### 0.5.5 训练循环中的使用

```python
# 典型训练循环
for batch in train_loader:  # DataLoader 返回 Batch
    # batch 包含合并后的所有张量
    # batch.batch 标记每个原子属于哪个结构
    
    # 模型前向传播
    outputs = model(batch)
    
    # 损失计算 (使用 batch.batch 进行 scatter 聚合)
    loss = criterion(outputs["energy"], batch.y)
```

### 0.6 Batch 中的关键计算

**按结构聚合（scatter 操作）：**

```python
# 将原子能量聚合到系统能量
system_energy = scatter(atom_energy, batch, dim=0, reduce="sum")
# 结果：[N] → [B]，每个结构一个能量值

# 按结构统计邻居数
atom_index_offset = natoms.cumsum(dim=0).roll(1)
atom_index_offset[0] = 0  # 用于修正 batch 中边的全局索引
```

### 0.7 完整 Data 字段清单

根据源码分析，单条 Data 可能包含以下字段：

| 字段 | 类型 | 形状 | 必需 | 说明 |
|------|------|------|------|------|
| `atomic_numbers` | LongTensor | `[N]` | ✅ | 原子序数 |
| `pos` | FloatTensor | `[N, 3]` | ✅ | 原子坐标 |
| `cell` | FloatTensor | `[3, 3]` | ⚠️ | 晶胞基矢（PBC 系统必需） |
| `natoms` | LongTensor | `[B]` | ⚠️ | 每结构原子数 |
| `y` | FloatTensor | `[1]` 或 `[B]` | ⚠️ | 能量标签 |
| `force` | FloatTensor | `[N, 3]` | ⚠️ | 力标签 |
| `stress` | FloatTensor | `[9]` | ⚠️ | 应力标签 |
| `edge_index` | LongTensor | `[2, E]` | ❌ | 边索引（可选，OTF 时不需要） |
| `cell_offsets` | LongTensor | `[E, 3]` | ❌ | 晶胞偏移 |
| `edge_distance` | FloatTensor | `[E]` | ❌ | 边距离 |
| `batch` | LongTensor | `[N]` | ✅ | 所属结构标记 |
| `fid` | LongTensor | `[1]` | ❌ | 帧 ID |
| `sid` | LongTensor | `[1]` | ❌ | 结构 ID |

---

---

## 1. 整体数据流概览

```
输入 batch
  │
  ├─ [otf_graph=True]  动态建图 generate_graph()
  │       │
  │       ├─ use_pbc=True, use_pbc_single=False → radius_graph_pbc(整批)
  │       ├─ use_pbc=True, use_pbc_single=True  → 逐结构调用后拼接
  │       └─ use_pbc=False                      → radius_graph(无PBC)
  │
  ├─ [otf_graph=False] 从 data 中读取预计算的 edge_index + cell_offsets
  │
  ▼
graph_dict: {edge_index, edge_distance, edge_distance_vec, ...}
  │
  ├─ get_rotmat_and_wigner()  → wigner, wigner_inv  (等变旋转矩阵)
  │
  ├─ sphere_embedding(atomic_numbers)              → x_message[:, 0, :]   (初始l=0节点特征)
  │
  ├─ distance_expansion(edge_distance)             → edge_distance_embedding (标量距离编码)
  ├─ source_embedding(atomic_numbers[src])         → source_embedding (源原子类型编码)
  ├─ target_embedding(atomic_numbers[tgt])         → target_embedding (目标原子类型编码)
  │     └─ cat → x_edge  (边标量特征, 形状 [E, num_distance_basis + 2*edge_channels])
  │
  └─ edge_degree_embedding(x_message, x_edge, ...)  → x_message (等变初始化后的节点特征)
           │
           └─ eSEN_Block × num_layers → node_embedding
```

---

## 2. 动态建边：`generate_graph()` 详解

### 2.1 为什么需要动态建边？

晶体材料的原子排列具有**周期性边界条件（PBC）**——每个原子不仅与同一 supercell 内的原子相互作用，还与相邻 image cell 中的周期性镜像原子相互作用。静态存储所有边代价极高，因此模型在每次前向传播时动态计算。

配置文件中：
```yaml
model:
  backbone:
    otf_graph: true   # on-the-fly 动态建图
    cutoff: 6.0       # 截断半径 (Å)
    max_neighbors: 300
    use_pbc: true
    use_pbc_single: true
```

### 2.2 无 PBC 的情况（`use_pbc=False`）

调用 `torch_geometric.nn.radius_graph`：

```python
# base.py:144-160
edge_index = radius_graph(
    data.pos,          # 所有原子坐标 [N, 3]
    r=cutoff,          # 截断半径
    batch=data.batch,  # 每个原子属于哪个结构
    max_num_neighbors=max_neighbors,
)
j, i = edge_index
distance_vec = data.pos[j] - data.pos[i]   # 边向量 [E, 3]
edge_dist = distance_vec.norm(dim=-1)       # 边长度 [E]
```

不涉及周期性，逻辑简单，适用于分子系统。

### 2.3 有 PBC 的情况（`use_pbc=True`）

调用自定义的 `radius_graph_pbc()`，核心步骤：

#### 步骤 1：确定需要扩展的 image cell 数量

```python
# utils.py:688-716
# 计算各方向所需重复次数（确保 cutoff 球能覆盖所有 image cell）
cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2])  # a2 x a3 法向量
cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1)  # 晶胞体积

# a1 方向: 两个 (a2,a3) 面之间的最短距离 = V / |a2 x a3|
inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol)
rep_a1 = torch.ceil(radius * inv_min_dist_a1)   # 需要 ±rep_a1 个 image
# 同理计算 rep_a2, rep_a3
```

**直觉**：截断球半径 `cutoff` 需要延伸到足够多的 image cell，使球不会遗漏任何近邻。沿 a1 方向需要的重复数是 `cutoff / d(a2a3 面)` 向上取整。

#### 步骤 2：构建候选对

```python
# 对 batch 中所有原子对 (i,j) × 所有 image cell 计算距离
unit_cell = torch.cartesian_prod(arange(-rep_a1, rep_a1+1),
                                  arange(-rep_a2, rep_a2+1),
                                  arange(-rep_a3, rep_a3+1))
# num_cells = (2*rep_a1+1) * (2*rep_a2+1) * (2*rep_a3+1)
pbc_offsets = data_cell @ unit_cell   # [batch, 3, num_cells]

# j 原子位置加上 PBC 偏移
pos2 = pos2 + pbc_offsets_per_atom   # 考虑所有 image
atom_distance_sqr = ||pos1 - pos2||²
```

#### 步骤 3：筛选有效边

```python
mask_within_radius = atom_distance_sqr <= radius²    # 在截断内
mask_not_same = atom_distance_sqr > 0.0001           # 排除自环
mask = mask_within_radius & mask_not_same

# 进一步限制每个原子的最大邻居数
mask_num_neighbors, _ = get_max_neighbors_mask(...)
```

返回：`edge_index [2, E]`, `cell_offsets [E, 3]`（表示哪个 image cell），`neighbors [N]`

### 2.4 `use_pbc_single=True` 的作用

**问题背景**：当 batch 中同时存在不同尺寸的晶胞时，整批 `radius_graph_pbc` 会取所有结构中最大的重复数，造成内存爆炸。

```python
# base.py:711-716
# 问题：不同结构的 rep_a1/a2/a3 差异很大
# 用最大值 padding → 候选对数量爆炸
max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]
```

**代码注释原文**（`base.py:118-127`）：
```python
## TODO this is the original call, but blows up with memory
## using two different samples
## sid='mp-675045-mp-675045-0-7' (MPTRAJ)
## sid='75396' (OC22)
edge_index, cell_offsets, neighbors = radius_graph_pbc(...)
```

**解决方案**（`use_pbc_single=True`）：**逐个结构**单独调用 `radius_graph_pbc`，每个结构只扩展自己所需的 image cell 数量，再把结果拼接：

```python
# base.py:87-116
if use_pbc_single:
    (edge_index_per_system, cell_offsets_per_system, neighbors_per_system) = list(
        zip(*[
            radius_graph_pbc(data[idx], cutoff, max_neighbors, ...)
            for idx in range(len(data))
        ])
    )
    # 修正原子索引偏移（每个结构的原子在 batch 中的全局索引）
    atom_index_offset = data.natoms.cumsum(dim=0).roll(1)
    atom_index_offset[0] = 0
    edge_index = torch.hstack([
        edge_index_per_system[idx] + atom_index_offset[idx]
        for idx in range(len(data))
    ])
    cell_offsets = torch.vstack(cell_offsets_per_system)
    neighbors = torch.hstack(neighbors_per_system)
```

**内存对比**：
| 模式 | 候选对数量 | 适用场景 |
|------|-----------|---------|
| `use_pbc_single=False` | `N² × max_cells` (全局最大) | 结构尺寸相近 |
| `use_pbc_single=True`  | `Σ Ni² × cells_i` (各自最优) | 混合数据集（如 MPTraj+OC22） |

**典型触发场景**：MPTraj 中的大超胞（如 675045 结构）与 OC22 中的小晶胞同批时。

---

## 3. 距离向量计算

建图完成后，需要计算带 PBC 偏移的真实距离向量：

```python
# base.py:129-142 / esen.py:266-288
# 方式1 (OTF): get_pbc_distances() 在 generate_graph 内部计算
out = get_pbc_distances(
    data.pos, edge_index, data.cell, cell_offsets, neighbors,
    return_offsets=True, return_distance_vec=True
)
distance_vec = out["distance_vec"]   # [E, 3]  j→i 方向的实空间向量

# 方式2 (非OTF, 从预存储数据读取):
shifts = einsum("ij,ijk->ik", cell_offsets.float(), cell_per_edge)  # [E, 3]
edge_distance_vec = pos[edge_index[0]] - pos[edge_index[1]] + shifts
```

**注意方向约定**：`edge_distance_vec` = `pos[src] - pos[tgt]`，即从 target 指向 source（与 `edge_index[0]=src, edge_index[1]=tgt` 对应）。

---

## 4. 原子嵌入层（Atom Embedding）

`eSEN_Backbone.__init__()` 中定义了三种原子嵌入：

```python
# esen.py:107-138
# 1. 节点特征初始化：纯 l=0 标量 embedding
self.sphere_embedding = nn.Embedding(max_num_elements, sphere_channels)
# 形状: [100, 128]（100种元素，128个通道）

# 2. 边特征中的源原子 embedding（标量，用于 x_edge 构建）
self.source_embedding = nn.Embedding(max_num_elements, edge_channels)
nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)

# 3. 边特征中的目标原子 embedding（标量，用于 x_edge 构建）
self.target_embedding = nn.Embedding(max_num_elements, edge_channels)
nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
```

### 4.1 三种 Embedding 的角色

| 名称 | 用途 | 初始化 |
|------|------|--------|
| `sphere_embedding` | 初始化节点的 l=0 成分，是节点在消息传递前的初始特征 | 默认（正态） |
| `source_embedding` | 构建边标量 `x_edge` 的一部分，提供源原子的化学信息 | 接近零的均匀分布 |
| `target_embedding` | 构建边标量 `x_edge` 的一部分，提供目标原子的化学信息 | 接近零的均匀分布 |

`source/target_embedding` 初始化接近零是为了让 `x_edge` 在训练初期主要由距离信息主导，逐步学习原子类型的影响。

### 4.2 节点特征初始化

```python
# esen.py:298-305
x_message = torch.zeros(
    N,                    # 原子数
    (lmax+1)²,            # 球谐系数数量，如 lmax=3 → 16
    sphere_channels,      # 通道数，如 128
    device=..., dtype=...,
)
# 只有 l=0 成分被填充，l>0 全为 0（等变要求：初始特征不能有方向偏好）
x_message[:, 0, :] = self.sphere_embedding(atomic_numbers)
#          ↑ 第0个系数 = l=0, m=0（标量），对旋转不变
```

**等变性保证**：l=0 的球谐系数（Y_0^0 = 常数）在任何旋转下不变，因此用它初始化节点特征不引入任何方向偏好。

---

## 5. 距离嵌入（Distance Embedding）

### 5.1 Gaussian Smearing（默认）

```python
# radial.py:45-61
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=cutoff, num_gaussians=512, basis_width_scalar=2.0):
        offset = torch.linspace(start, stop, num_gaussians)  # 512 个均匀分布的中心
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])) ** 2

    def forward(self, dist):  # dist: [E]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)   # [E, 512]
        return torch.exp(self.coeff * dist ** 2)              # [E, 512]
```

每个边的距离被投影到 512 个高斯基函数上。高斯宽度 `basis_width_scalar=2.0` 控制重叠程度（值越大，相邻基函数重叠越多，编码越平滑）。

**示意图**（cutoff=6Å，512 个 Gaussian）：
```
                     ↓ d=2.5Å
|___|___|_|__|_|__|__|__|...
0  0.5  1   2   3   4   6  Å
每个基函数在对应距离处响应最强，共 512 维输出
```

### 5.2 EnvelopedBesselBasis（`use_envelope=True`）

```python
# radial.py:99-113
class EnvelopedBesselBasis(torch.nn.Module):
    def forward(self, d):
        d_scaled = d / cutoff                        # 归一化到 [0,1]
        env = PolynomialEnvelope(d_scaled)           # 在 cutoff 处平滑衰减到 0
        return env[:, None] * SphericalBesselBasis(d_scaled)  # 截断的 Bessel 基
```

多项式 Envelope（p=5次）保证边特征在 cutoff 处平滑衰减：
```
f(d/r_cut) = 1 + a*(d/r)^5 + b*(d/r)^6 + c*(d/r)^7
             → 0 when d → r_cut  (且导数也连续到0)
```

### 5.3 边标量特征 `x_edge` 的构建

```python
# esen.py:308-317
edge_distance_embedding = self.distance_expansion(edge_distance)  # [E, 512]
source_embedding = self.source_embedding(atomic_numbers[edge_index[0]])  # [E, 128]
target_embedding = self.target_embedding(atomic_numbers[edge_index[1]])  # [E, 128]

x_edge = torch.cat((edge_distance_embedding, source_embedding, target_embedding), dim=1)
#  形状: [E, 512 + 128 + 128] = [E, 768]
#  含义: 每条边的标量（旋转不变）特征
```

`x_edge` 包含：
- **距离信息**（512维）：这条边有多长
- **源原子类型**（128维）：谁在发送消息
- **目标原子类型**（128维）：谁在接收消息

这是**旋转不变**的标量特征，全程不直接参与等变运算，而是作为 SO(2) 卷积的 `rad_func` 输入，动态生成卷积权重。

---

## 6. EdgeDegreeEmbedding：等变初始化节点特征

这是建图和正式消息传递之间最关键的步骤——将边的标量信息转换为节点的等变特征（类似 graph neural network 的"第 0 层"）。

### 6.1 设计动机

初始的 `x_message` 只有 l=0 成分（旋转不变），高阶球谐成分全为零。为了让第一层 `eSEN_Block` 能有效工作，需要一个**初始化步骤**把邻居信息聚合进来，且这个聚合必须是等变的。

### 6.2 实现流程

```python
# embedding.py:79-122
def forward(self, x, x_edge, edge_distance, edge_index, wigner_inv, node_offset=0):
    # 步骤 1: 用 RadialMLP 将边标量特征映射到球谐系数空间
    x_edge_m_0 = self.rad_func(x_edge)   # [E, m0_coeff * sphere_channels]
    # m0_coeff = mappingReduced.m_size[0] = lmax+1 (只有m=0成分)
    x_edge_m_0 = x_edge_m_0.reshape(-1, m0_coeff, sphere_channels)  # [E, lmax+1, C]

    # 步骤 2: 高阶 m 的位置用 0 填充（只有 m=0 有信号）
    x_edge_m_pad = zeros([E, m_all - m0_coeff, C])
    x_edge_embedding = cat((x_edge_m_0, x_edge_m_pad), dim=1)  # [E, m_total, C]

    # 步骤 3: m-major 排序 → l-major 排序（等价于应用 mappingReduced.to_m 的逆）
    x_edge_embedding = einsum("nac,ab->nbc", x_edge_embedding, mappingReduced.to_m)
    # 重排索引，从 m-major 转为 l-major，形状仍 [E, (L+1)², C]

    # 步骤 4: 将边特征"反旋转"回全局坐标系
    x_edge_embedding = bmm(wigner_inv[:, :, out_mask], x_edge_embedding)
    # wigner_inv: [E, (L+1)², (L+1)²]  (等变反旋转)
    # out_mask: 只取 mmax 范围内的系数

    # 步骤 5 (可选): 乘以 envelope 使边特征在 cutoff 处平滑衰减
    if use_envelope:
        env = PolynomialEnvelope(edge_distance / cutoff)
        x_edge_embedding = x_edge_embedding * env.view(-1, 1, 1)

    # 步骤 6: 按目标原子聚合（加法），等价于"求和消息传递"
    x.index_add_(0, edge_index[1] - node_offset, x_edge_embedding / rescale_factor)
    return x
```

### 6.3 为什么要"反旋转"？

`EdgeDegreeEmbedding` 的核心思想是：

1. 边的标量特征 `x_edge` 没有方向性，通过 MLP 生成的 `x_edge_m_0` 只有 l=0 成分（旋转不变）
2. 这些 m=0 的系数在"边局部坐标系"中是合法的球谐系数（因为 m=0 对应"竖直方向"的分量）
3. 要把它们加到全局坐标的节点特征上，必须先用 `wigner_inv` 把它们"反旋转"到全局坐标系

```
边局部坐标系            全局坐标系
   (m=0 特征)  →[D^l(R)^{-1}]→  节点特征
```

等变性保证：若全局坐标旋转 R，则 `wigner_inv` 也跟着变，使得聚合后的节点特征也正确变换。

### 6.4 `out_mask`：只用 mmax 范围的系数

```python
# esen.py:155-158
out_mask = SO3_grid["lmax_lmax"].mapping.coefficient_idx(lmax, mmax)
# 例如 lmax=3, mmax=2: out_mask 选取所有 |m| <= 2 的系数
# 排除 |m|=3 的系数（这些在 SO(2) 卷积中不使用）
```

原因：eSEN 的 SO(2) 卷积只处理 `|m| <= mmax` 的成分（而不是全部 `|m| <= l`），因此 wigner_inv 也只需要用这部分列。

### 6.5 `rescale_factor=5.0`：聚合归一化

边的数量随截断半径的三次方增长。对于一个原子，其邻居数可能高达 200+，直接求和会使节点特征的量级随邻居数变化。除以 `rescale_factor=5.0` 提供粗略的归一化。

更精细的归一化在每个 `eSEN_Block` 的 `LayerNorm` 中完成。

---

## 7. 完整数据流维度标注

以 `lmax=3, mmax=2, sphere_channels=128, edge_channels=128, num_distance_basis=512, cutoff=6Å` 为例：

```
输入 batch:
  atomic_numbers: [N]          (原子序数)
  pos:            [N, 3]       (原子坐标, Å)
  cell:           [B, 3, 3]    (B个结构的晶胞向量)
  batch:          [N]          (每个原子属于哪个结构)
  natoms:         [B]          (每个结构的原子数)

▼ generate_graph() (OTF, use_pbc=True, use_pbc_single=True)
  edge_index:         [2, E]   (E=总边数, 含 PBC image cell 贡献的边)
  edge_distance:      [E]      (每条边的实际长度, Å)
  edge_distance_vec:  [E, 3]   (每条边的实空间向量, 从 tgt 指向 src)
  cell_offsets:       [E, 3]   (每条边的 image cell 整数偏移)

▼ get_rotmat_and_wigner()
  wigner:     [E, 16, 16]    ((lmax+1)²=16 维 Wigner D 矩阵, 每条边一个)
  wigner_inv: [E, 16, 16]    (wigner 的转置, 即逆矩阵)

▼ 初始化节点特征
  x_message:  [N, 16, 128]  全零, 只有 x_message[:, 0, :] 被 sphere_embedding 填充

▼ 构建边标量特征
  edge_distance_embedding: [E, 512]  (Gaussian smearing)
  source_embedding:        [E, 128]
  target_embedding:        [E, 128]
  x_edge:                  [E, 768]  (三者 concat)

▼ EdgeDegreeEmbedding
  x_edge_m_0:    [E, 4, 128]   (RadialMLP 输出, m0_coeff=lmax+1=4)
  x_edge_embed:  [E, 16, 128]  (pad 到全尺寸后 to_m 排序)
  → wigner_inv 反旋转
  → index_add 到 x_message
  x_message:     [N, 16, 128]  (等变初始化完毕)

▼ eSEN_Block × num_layers
  x_message:     [N, 16, 128]  (每层后维度不变)

▼ Final LayerNorm
  node_embedding: [N, 16, 128]

▼ Heads
  MLP_Energy_Head: x[:, 0, :] → Linear → [N] → scatter_add → [B]  (energy)
  Linear_Force_Head: x[:, 0:4, :] → SO3_Linear(C,1,lmax=1) → x[:, 1:4, :] → [N, 3]  (forces)
```

---

## 8. 关键设计决策总结

| 设计选择 | 原因 |
|---------|------|
| `sphere_embedding` 只初始化 l=0 | l>0 成分有方向性，随机初始化会破坏等变性 |
| `source/target_embedding` 接近零初始化 | 避免干扰距离信息主导的早期训练 |
| `x_edge` 为标量（不含方向信息） | 需要对旋转不变，只有标量才满足要求 |
| `use_pbc_single=True` | 避免混合数据集中因晶胞尺寸差异导致的内存爆炸 |
| `EdgeDegreeEmbedding` 用 `wigner_inv` 反旋转 | 将边局部坐标的 m=0 特征正确映射到全局坐标，保持等变性 |
| `enforce_max_neighbors_strictly=False` | 防止在退化边情况下任意选邻居导致能量不连续 |
| Gaussian Smearing 带宽 `basis_width_scalar=2.0` | 增大相邻高斯的重叠，使距离编码更平滑，对微小扰动更稳定 |

---

## 9. 相关源码文件索引

| 文件 | 关键内容 |
|------|---------|
| `src/fairchem/core/models/base.py:50-172` | `GraphModelMixin.generate_graph()` 建边入口，`use_pbc_single` 分支 |
| `src/fairchem/core/common/utils.py:618-780` | `radius_graph_pbc()` 完整实现（PBC 重复、候选对、邻居数限制） |
| `src/fairchem/core/models/esen/esen.py:107-159` | 三种 embedding 初始化 |
| `src/fairchem/core/models/esen/esen.py:296-324` | 前向传播：x_message 初始化、x_edge 构建、EdgeDegreeEmbedding 调用 |
| `src/fairchem/core/models/esen/nn/embedding.py` | `EdgeDegreeEmbedding` 完整实现 |
| `src/fairchem/core/models/esen/nn/radial.py` | `GaussianSmearing`, `EnvelopedBesselBasis`, `RadialMLP` |
