# eSEN Backbone 工程实现：旋转、SO(2) 卷积与等变性

> **📖 文档导航**
> - **上一篇**: [ESEN 模型调用流程](./ESEN_CALL_FLOW.md)
> - **下一篇**: [Nodewise FFN 层详解](./ESEN_FFN_LAYERS.md)
>
> 本文档深入讲解 eSEN backbone 的核心数学机制在工程上的具体实现。

---

## 0. 为什么需要等变性？

材料模拟中，分子旋转后其能量不变（不变性），而力向量需要跟着旋转（等变性）。模型如果不满足这些性质，就需要用大量数据增广才能学好这些对称性，效率极低。

eSEN 的核心思路：

- 用**球谐函数**（spherical harmonics）表达节点特征，使特征天然携带旋转信息；
- 用**Wigner D 矩阵**描述旋转对特征的作用，保证每一步操作都满足等变性；
- 用**SO(2) 卷积**在"对齐到边"的局部坐标系下做消息传递，从 SO(3) 等变降低到 SO(2)，大幅减少计算量。

---

## 1. 特征的数学结构

### 1.1 球谐函数展开

每个原子 $i$ 有特征 $\mathbf{x}_i \in \mathbb{R}^{(L+1)^2 \times C}$：

- 第一维 $(L+1)^2$ 是所有球谐次数 $l=0,1,\ldots,L$ 的系数总数（每个 $l$ 有 $2l+1$ 个 $m$ 分量）；
- 第二维 $C$ 是特征通道数（`sphere_channels`）。

```python
# esen.py: 初始化节点特征
x_message = torch.zeros(
    num_atoms,
    self.sph_feature_size,   # = (L+1)^2
    self.sphere_channels,    # = C
)
x_message[:, 0, :] = self.sphere_embedding(atomic_numbers)  # l=0 分量填入原子嵌入
```

所有 $l>0$ 的分量初始为零，后续通过 `EdgeDegreeEmbedding` 注入方向信息。

### 1.2 等变性的定义

设 $R \in SO(3)$ 是一个旋转，$D^{(l)}(R)$ 是 $l$ 阶 Wigner D 矩阵（$(2l+1)\times(2l+1)$ 的正交矩阵）。

特征等变 = 对每个 $l$ 分量，旋转作用等价于乘以 $D^{(l)}(R)$：

$$\mathbf{x}_i^{(l)} \xrightarrow{R} D^{(l)}(R)\, \mathbf{x}_i^{(l)}$$

整个特征张量在旋转下的变换是分块的：

$$\mathbf{x}_i \xrightarrow{R} \text{BlockDiag}(D^{(0)}(R), D^{(1)}(R), \ldots, D^{(L)}(R))\, \mathbf{x}_i$$

---

## 2. 旋转矩阵的计算

### 2.1 边对齐旋转矩阵

**核心思路**：对每条边 $(i \to j)$，构建一个旋转 $R_{ij}$，把全局坐标系的 $y$ 轴对准到 $\vec{r}_{ij}$（边方向）。

代码入口：`rotation.py:init_edge_rot_mat()`

```python
# rotation.py: init_edge_rot_mat()
# 1. 归一化边方向向量 → 作为新坐标系的 x 轴
norm_x = edge_vec_0 / edge_vec_0_distance.view(-1, 1)

# 2. 生成随机向量，通过 Gram-Schmidt 正交化得到 z 轴和 y 轴
norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
norm_z = norm_z / norm_z.norm(dim=1, keepdim=True)
norm_y = torch.cross(norm_x, norm_z, dim=1)
norm_y = norm_y / norm_y.norm(dim=1, keepdim=True)

# 3. 拼成旋转矩阵 R_inv = [norm_z | norm_x | norm_y]  (列向量)
edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)  # (E, 3, 3)
edge_rot_mat = edge_rot_mat_inv.transpose(1, 2)  # R = R_inv^T (正交矩阵的逆是转置)
```

**为什么用随机辅助向量**：边方向只确定一个轴，还差一个 DOF（绕该轴的旋转）。这个自由度对于 SO(2) 等变性是**允许的**——正是利用了这一点把 SO(3) 降到 SO(2)。代码中用随机向量+正交化生成稳定的正交基，同时检查不与主轴对齐（`assert max(vec_dot) < 0.99`）。

### 2.2 Wigner D 矩阵的计算

`rotation.py:rotation_to_wigner()` 将 $3\times3$ 旋转矩阵 $R$ 转换为 Wigner D 矩阵（每个 $l$ 阶一个块）。

**分解步骤（ZYZ Euler 角参数化）**：

$$R = R_z(\alpha) \cdot R_y(\beta) \cdot R_z(\gamma)$$

对应的 Wigner D 矩阵：

$$D^{(l)}(R) = D^{(l)}(\alpha, \beta, \gamma) = e^{-im\alpha} \cdot d^{(l)}(\beta) \cdot e^{-im'\gamma}$$

```python
# rotation.py: rotation_to_wigner()
x = edge_rot_mat @ torch.tensor([0., 1., 0.])   # R 作用在 y 轴
alpha, beta = o3.xyz_to_angles(x)               # 从旋转后的 y 轴提取 alpha, beta
R_partial = angles_to_matrix(alpha, beta, 0).T @ edge_rot_mat
gamma = torch.atan2(R_partial[..., 0, 2], R_partial[..., 0, 0])  # 提取 gamma

# 对每个 l 计算 D^(l)(alpha, beta, gamma)
for lmax in range(start_lmax, end_lmax + 1):
    block = wigner_D(lmax, alpha, beta, gamma, Jd)  # (E, 2l+1, 2l+1)
    wigner[:, start:end, start:end] = block
```

`wigner_D` 的核心是 `_z_rot_mat(angle, l)`，它构建绕 z 轴旋转 l 阶的矩阵：

```python
# rotation.py: _z_rot_mat()
# 对角元素：cos(m * angle)，其中 m = l, l-1, ..., -l
# 反对角元素：sin(m * angle)
for i, freq in enumerate(frequencies):   # frequencies = [l, l-1, ..., -l]
    M[..., inds[i], inds[i]]         = torch.cos(freq * angle)
    M[..., inds[i], reversed_inds[i]] = torch.sin(freq * angle)
```

最终 `wigner` 张量形状为 `(E, (L+1)^2, (L+1)^2)`，每条边一个分块对角矩阵。

`wigner_inv = wigner.transpose(1, 2)` — Wigner D 矩阵是正交的，转置即逆。

---

## 3. SO(2) 卷积：工程核心

### 3.1 问题：直接做 SO(3) 等变卷积很贵

完整的 SO(3) Clebsch-Gordan 张量积复杂度为 $O(L^6 C^2)$。eSEN 用"把特征旋转到边局部坐标系"的技巧，把 SO(3) 降到 SO(2)，复杂度降至 $O(L^2 C^2)$。

### 3.2 旋转到边局部坐标系

```python
# esen_block.py: Edgewise.forward()
# x_message: (E, (L+1)^2, 2C) — source 和 target 特征拼接
x_message = torch.bmm(wigner[:, self.out_mask, :], x_message)
```

**数学含义**：

$$\tilde{\mathbf{x}}_{ij} = D(R_{ij}) \cdot \mathbf{x}$$

旋转后，特征的 $m=0$ 分量对应边方向（新的 $y$ 轴），$m \neq 0$ 分量是垂直于边的横向模式。

**为什么旋转后可以省计算**：在边局部坐标系里，绕边方向的旋转（SO(2)）不改变 $m$ 的绝对值，只改变相位。因此不同 $m$ 的分量之间**不需要混合**，可以分 $m$ 独立处理。

### 3.3 系数重排：从 l-major 到 m-major

SO(2) 卷积按 $m$ 分组处理，而球谐系数默认是按 $l$ 排列的。用 `CoefficientMapping.to_m` 重排：

```python
# so2_layers.py: SO2_Convolution.forward()
x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)
# 变换前：x[edge, l_coeff_idx, channel]  (l-major)
# 变换后：x[edge, m_coeff_idx, channel]  (m-major)
```

`to_m` 是一个置换矩阵（仅含 0 和 1），把每个 $m$ 值对应的所有 $(l, m)$ 系数排列在一起。

重排后，对每个 $m$，系数的排列为：所有满足 $l \geq |m|$ 的 $l$ 值对应的 $(l, m)$ 系数。

### 3.4 m=0 的处理（纯实数）

$m=0$ 的球谐函数是实数，没有虚部：

```python
# so2_layers.py: SO2_Convolution.forward()
x_0 = x.narrow(1, 0, self.mappingReduced.m_size[0])  # 取 m=0 的系数
x_0 = x_0.reshape(num_edges, -1)                      # (E, (L+1)*C)
if self.rad_func is not None:
    x_0 = x_0 * x_edge_0                              # 乘以径向函数权重
x_0 = self.fc_m0(x_0)                                 # 线性变换
```

### 3.5 m>0 的 SO(2) 卷积（核心）

对每个 $m > 0$，有实部和虚部两组系数（形状 `(E, 2, num_l_coeffs * C)`）。

**SO(2) 卷积的数学**：绕 $z$ 轴旋转角 $\phi$ 对 $(m, -m)$ 分量的作用是：

$$\begin{pmatrix} f_m' \\ f_{-m}' \end{pmatrix} = \begin{pmatrix} \cos(m\phi) & -\sin(m\phi) \\ \sin(m\phi) & \cos(m\phi) \end{pmatrix} \begin{pmatrix} f_m \\ f_{-m} \end{pmatrix}$$

"SO(2) 等变的线性层"就是与一个可学习的 $2\times2$ 旋转等变矩阵作用——其唯一自由度是旋转角度，等价于对 $(f_m + if_{-m})$ 乘以复数 $e^{i\theta}$。

工程上拆成实部虚部：

```python
# so2_layers.py: SO2_m_Conv.forward()
x_m = self.fc(x_m)                             # 线性变换：(E, 2, K) → (E, 2, 2*K')
x_r = x_m[..., :out_channels_half]             # 前半部分
x_i = x_m[..., out_channels_half:]             # 后半部分
# SO(2) 等变的复数乘法：
# 输出实部 = x_r * Re(w) - x_i * Im(w)
# 输出虚部 = x_r * Im(w) + x_i * Re(w)
x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)   # 实部：Re(f) * Re(w) - Im(f) * Im(w)
x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)   # 虚部：Re(f) * Im(w) + Im(f) * Re(w)
return torch.cat((x_m_r, x_m_i), dim=1)
```

**为什么这保持等变性**：`fc` 层的权重矩阵是 $4K \times K'$ 的普通线性层，但输入是 $\begin{pmatrix} \text{Re}(f_m) \\ \text{Im}(f_{-m}) \end{pmatrix}$，输出按复数乘法公式组合，天然满足 SO(2) 等变性。

径向函数（`rad_func`）输出标量权重，乘到每个 $m$ 组的特征上，不破坏等变性（标量乘法保等变）：

```python
x_edge_m = x_edge.narrow(1, offset_rad, self.so2_m_conv[m-1].fc.in_features)
x_m = x_m * x_edge_m.reshape(num_edges, 1, -1)   # (E, 2, K) * (E, 1, K) — 广播
```

### 3.6 旋转回全局坐标系

SO(2) 卷积后，特征仍在边局部坐标系，需旋转回全局：

```python
# esen_block.py: Edgewise.forward()
x_message = torch.bmm(wigner_inv[:, :, self.out_mask], x_message)
# wigner_inv = wigner^T (正交矩阵逆 = 转置)
```

最后用 `index_add_` 聚合（sum aggregation）邻居消息：

```python
new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)
```

---

## 4. 等变性的系统保证

### 4.1 等变操作列表

| 操作 | 是否等变 | 原因 |
|------|---------|------|
| 球谐函数初始化 | ✅ | $l=0$ 分量是标量，不受旋转影响 |
| Wigner 矩阵旋转 | ✅ | 精确的群表示 |
| SO(2) 卷积 | ✅ | 按 $m$ 分组，复数相乘保等变 |
| 径向函数（MLP） | ✅ | 作用于标量距离，输出标量权重 |
| `index_add_` 聚合 | ✅ | 线性操作，等变性可传递 |
| 残差连接 | ✅ | 相同空间的等变张量相加 |
| LayerNorm（球谐版本） | ✅ | 按 $l$ 分组归一化，不混合不同 $l$ |

### 4.2 非线性激活的等变处理

等变网络中，逐元素非线性（如 ReLU）**会破坏等变性**，因为它混合了不同 $m$ 的信息。eSEN 用两种方法解决：

#### Gate Activation（`GateActivation`）

用 $l=0$ 的标量（不受旋转影响）做门控，对高阶分量做缩放：

$$\mathbf{x}^{(l)} \leftarrow \sigma(g^{(l)}) \cdot \mathbf{x}^{(l)}, \quad g^{(l)} \text{ 是标量门控}$$

```python
# activation.py: GateActivation
# x_0_gating: (E, L*C)  — 来自 SO(2) conv 的额外 m=0 输出
# x: (E, (L+1)^2, C)   — 高阶特征
```

标量门控由 `extra_m0_output_channels` 提供，只对 $l>0$ 的分量做缩放，$l=0$ 标量分量正常经过激活函数。

#### S2 Activation（`SeparableS2Activation`）

投影到球面网格（实空间），在网格点做逐点非线性，再投影回球谐空间：

```python
# grid → point-wise nonlinearity → irreps
x_grid = SO3_grid.to_grid(x, lmax, mmax)   # (N, lat, long, C)
x_grid = activation(x_grid)                 # 逐点
x = SO3_grid.from_grid(x_grid, lmax, mmax) # 回到球谐
```

球面网格采样在 SO(3) 作用下是等变的（旋转只是网格点置换），因此此操作保持等变性。

### 4.3 LayerNorm 的等变版本

标准 LayerNorm 对所有维度归一化，会破坏不同 $l$ 之间的相对大小。eSEN 用 `EquivariantRMSNormArraySphericalHarmonicsV2`：

- 对每个 $l$ 分量**分别计算 RMS 范数**，然后归一化；
- 不混合不同 $l$ 的信息；
- 可学习的缩放因子是标量（$l=0$ 的量），不影响等变性。

### 4.4 力预测头的等变性

**直接力预测（direct force head）**：

```python
# esen.py: Linear_Force_Head
self.linear = SO3_Linear(sphere_channels, 1, lmax=1)   # 只用 l=0,1 分量
forces = self.linear(node_embedding.narrow(1, 0, 4))    # 取 (l=0,1) 共 1+3=4 个分量
forces = forces.narrow(1, 1, 3)                         # 取 l=1 的 3 个分量 (m=-1,0,1)
forces = forces.view(-1, 3)                             # 对应 (x, y, z) 力分量
```

$l=1$ 的球谐函数恰好是 $(y, z, x)$ 的线性组合，旋转下变换方式与向量相同，因此力预测天然等变。

**梯度力预测（auto-grad force head）**：能量是标量（不变量），对位置坐标求梯度得到力，等变性由微积分链式法则自动保证。

---

## 5. 完整数据流图（含维度）

```
输入: atomic_numbers (N,), pos (N, 3), edge_index (2, E)
         │
         ▼
初始化节点特征:
  x: (N, (L+1)², C)       全零初始化
  x[:, 0, :] ← sphere_embedding(atomic_numbers)    (N, C)
         │
         ▼
边特征编码:
  edge_dist:      (E,)
  x_edge:         (E, D_basis + 2*D_embed)
         │
         ├─ distance_expansion: (E,) → (E, D_basis)    [Gaussian/Bessel]
         ├─ source_embedding:   (E,) → (E, D_embed)    [原子类型]
         └─ target_embedding:   (E,) → (E, D_embed)    [原子类型]
         │
         ▼
旋转矩阵 & Wigner D:
  edge_rot_mat:  (E, 3, 3)        [init_edge_rot_mat]
  wigner:        (E, (L+1)², (L+1)²)   [rotation_to_wigner]
  wigner_inv:    (E, (L+1)², (L+1)²)   = wigner.T
         │
         ▼
EdgeDegreeEmbedding: 初始化等变方向信息
  x: (N, (L+1)², C)
         │
         ▼
eSEN_Block × num_layers:
  │
  ├─ LayerNorm (等变)
  │
  ├─ Edgewise (消息传递):
  │   ├─ 取 source/target: x_source, x_target → cat → (E, (L+1)², 2C)
  │   ├─ 旋转到边坐标系: wigner @ x → (E, N_coeffs, 2C)
  │   ├─ SO2_Convolution 1: (E, N_coeffs, 2C) → (E, N_coeffs, H) + gating scalars
  │   ├─ GateActivation / S2Activation
  │   ├─ SO2_Convolution 2: (E, N_coeffs, H) → (E, N_coeffs, C)
  │   ├─ Envelope(距离加权)
  │   ├─ 旋转回全局: wigner_inv @ x → (E, (L+1)², C)
  │   └─ index_add_: (E, (L+1)², C) → (N, (L+1)², C)
  │
  ├─ 残差连接
  ├─ LayerNorm (等变)
  │
  ├─ Atomwise (节点 MLP):
  │   ├─ SpectralAtomwise: SO3_Linear + GateActivation + SO3_Linear
  │   └─ GridAtomwise: to_grid + MLP + from_grid
  │
  └─ 残差连接
         │
         ▼
Final LayerNorm
         │
  node_embedding: (N, (L+1)², C)
         │
         ├─ Energy Head: narrow(dim=1, 0, 1) → (N, 1, C) → squeeze → MLP → scalar
         └─ Force Head:  narrow(dim=1, 0, 4) → SO3_Linear(lmax=1) → l=1 分量 → (N, 3)
```

---

## 6. 关键超参数的影响

| 参数 | 默认值 | 含义 | 增大的影响 |
|------|--------|------|-----------|
| `lmax` | 2~3 | 球谐最高阶数 | 表达能力↑，计算量 $O(L^3)$ 级增长 |
| `mmax` | 2 | SO(2) 卷积最高 $m$ 阶 | $\leq$ lmax；减小可降低 SO(2) 卷积开销 |
| `sphere_channels` | 128 | 特征通道数 | 容量↑，显存线性增长 |
| `hidden_channels` | 128 | SO(2) 卷积中间层通道 | 中间表达能力↑ |
| `num_layers` | 10 | eSEN Block 层数 | 感受野↑，计算量线性增长 |
| `cutoff` | 6.0 Å | 近邻截断半径 | 邻居数↑，长程相互作用↑ |

**`mmax < lmax` 的节省**：SO(2) 卷积只对 $|m| \leq $ mmax 的分量做混合，跳过高 $m$ 值，节省约 $(L - M_{max}) / L$ 比例的计算。

---

## 7. 与其他等变模型的对比

| 模型 | 等变机制 | 复杂度 | eSEN 的区别 |
|------|---------|--------|------------|
| NequIP / MACE | CG 张量积 | $O(L^6 C^2)$ | eSEN 用 SO(2) 降到 $O(L^2 C^2)$ |
| SEGNN | CG 张量积 | 同上 | 同上 |
| EquiformerV2 | SO(2) 卷积（类似 ESCN） | $O(L^2 C^2)$ | eSEN 去掉了 attention，用 gate/s2 激活 |
| ESCN | SO(2) 卷积 | $O(L^2 C^2)$ | eSEN 加了 atomwise MLP 和更灵活的激活 |

---

## 8. 相关源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| 边旋转矩阵 | `common/rotation.py` | `init_edge_rot_mat()` |
| Wigner D 矩阵 | `common/rotation.py` | `rotation_to_wigner()`, `wigner_D()` |
| 系数映射 (l↔m) | `common/so3.py` | `CoefficientMapping` |
| 球面网格 | `common/so3.py` | `SO3_Grid` |
| SO(2) 卷积 | `nn/so2_layers.py` | `SO2_m_Conv`, `SO2_Convolution` |
| SO(3) 线性 | `nn/so3_layers.py` | `SO3_Linear` |
| 等变激活 | `nn/activation.py` | `GateActivation`, `SeparableS2Activation` |
| 等变 LayerNorm | `nn/layer_norm.py` | `EquivariantRMSNormArraySphericalHarmonicsV2` |
| 消息传递块 | `esen_block.py` | `Edgewise`, `eSEN_Block` |
| Backbone 主体 | `esen.py` | `eSEN_Backbone` |
