# SO(2) 层详解：eSEN 等变消息传递的核心

> **📖 文档导航**
> - 完整训练流程见 [ESEN_TRAINING_PIPELINE.md](./ESEN_TRAINING_PIPELINE.md)
> - 源码：`src/fairchem/core/models/esen/nn/so2_layers.py`
> - 依赖：`src/fairchem/core/models/esen/common/so3.py`（`CoefficientMapping`）

---

## 一、背景：为何用 SO(2) 而非 SO(3)

### 1.1 SO(3) 直接操作的瓶颈

完整 SO(3) 等变消息传递需要张量积（Clebsch-Gordan 系数），复杂度为 O(L⁶)，当 `lmax=3` 时已非常昂贵。

### 1.2 eSEN 的关键思路

**将节点特征旋转到以边方向为 z 轴的局部坐标系**。在这个坐标系中，绕 z 轴旋转（即 SO(2)）是该局部坐标系的对称操作。

SO(2) 的 m 阶不可约表示只是一个 2×2 旋转矩阵：

$$D^{(m)}(\phi) = \begin{pmatrix} \cos(m\phi) & -\sin(m\phi) \\ \sin(m\phi) & \cos(m\phi) \end{pmatrix}$$

因此，球谐系数按 m 阶分组后，每组只需做独立的 2×2 操作，复杂度从 O(L⁶) 降到 **O(L²)**。

### 1.3 全局等变性如何保证

```
全局坐标系特征
    │
    ├── wigner × x  →  旋转到边局部坐标系
    │
    ├── SO(2) 卷积（m 阶独立操作）
    │
    └── wigner_inv × x  →  旋转回全局坐标系
```

旋转到局部 → SO(2) 操作 → 旋转回来，这一往返操作整体上是 SO(3) 等变的。

---

## 二、CoefficientMapping：l ↔ m 重排

**文件**：`common/so3.py:16-163`

### 2.1 问题

球谐系数标准存储按 l（degree）排列：

```
l=0: Y_0^0
l=1: Y_1^{-1}, Y_1^0, Y_1^1
l=2: Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2
l=3: ...
```

SO(2) 卷积需要按 m（order）分组处理：

```
m=0: Y_0^0, Y_1^0, Y_2^0, Y_3^0  （所有 l 的 m=0 系数）
m=1: Y_1^{+1}, Y_1^{-1}, Y_2^{+1}, Y_2^{-1}, ...  （所有 l 的 m=±1 系数）
...
```

### 2.2 to_m 置换矩阵构造（`so3.py:50-66`）

```python
to_m = torch.zeros([num_coefficients, num_coefficients])
offset = 0
for m in range(mmax + 1):
    idx_r, idx_i = self.complex_idx(m, ...)
    # idx_r: 所有 l 中 +m 分量的原始位置
    # idx_i: 所有 l 中 -m 分量的原始位置（m=0 时为空）
    for idx_out, idx_in in enumerate(idx_r):
        to_m[idx_out + offset, idx_in] = 1.0
    offset += len(idx_r)
    # 虚部（-m）紧跟实部（+m）
    for idx_out, idx_in in enumerate(idx_i):
        to_m[idx_out + offset, idx_in] = 1.0
    offset += len(idx_i)
    self.m_size[m] = len(idx_r)  # 每个 m 阶有多少个 l 值
```

`to_m` 是一个置换矩阵（只含 0/1），通过 einsum 实现 O(1) 的重排：

```python
# SO2_Convolution.forward:148
x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)
# n=edge, a=旧系数维, b=新系数维（m序）, c=channel
```

**lmax=3, mmax=2 时的 m_size 分布：**

| m | 包含的 l | 实部数量（m_size[m]） |
|---|---------|-------------------|
| 0 | l=0,1,2,3 | 4 |
| 1 | l=1,2,3   | 3（+1 实部 3 虚部） |
| 2 | l=2,3     | 2（+2 实部 2 虚部） |

---

## 三、SO2_m_Conv：单个 m 阶的 SO(2) 卷积

**文件**：`so2_layers.py:20-68`

```python
class SO2_m_Conv(nn.Module):
    def __init__(self, m, sphere_channels, m_output_channels, lmax, mmax):
        # 该 m 阶包含的 l 数量：l=m, m+1, ..., lmax
        num_coefficients = lmax - m + 1
        num_channels = num_coefficients * sphere_channels

        # 输出 2 倍通道：分别作用在实部和虚部
        out_channels_half = m_output_channels * num_coefficients
        self.fc = Linear(num_channels, 2 * out_channels_half, bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))  # 初始化缩放补偿
```

### 3.1 forward：SO(2) 等变旋转

```python
def forward(self, x_m):
    # x_m: [E, 2, num_coefficients * sphere_channels]
    #         ↑ 第1维：实部（+m）和 虚部（-m）

    x_m = self.fc(x_m)              # 线性变换：每半部分独立
    # 分割为两半
    x_r = x_m[:, :, :half]          # 作用在实部的输出
    x_i = x_m[:, :, half:]          # 作用在虚部的输出

    # SO(2) 旋转：等价于复数乘法 (a+ib)(c+id) = (ac-bd) + i(bc+ad)
    x_m_r = x_r[:, 0:1] - x_i[:, 1:2]   # 输出实部：cos 项
    x_m_i = x_r[:, 1:2] + x_i[:, 0:1]   # 输出虚部：sin 项
    return torch.cat((x_m_r, x_m_i), dim=1)
```

**等变性数学原理**：

设输入 $(f_r, f_i)$ 表示 m 阶复球谐系数，SO(2) 旋转角 φ 作用为：

$$\begin{pmatrix} f_r' \\ f_i' \end{pmatrix} = \begin{pmatrix} \cos(m\phi) & -\sin(m\phi) \\ \sin(m\phi) & \cos(m\phi) \end{pmatrix} \begin{pmatrix} f_r \\ f_i \end{pmatrix}$$

线性层 `fc` 将实部和虚部混合后，上面的交叉项操作 `x_m_r = x_r[0] - x_i[1]` 正是实现了这个旋转矩阵的乘法，所以输出在 SO(2) 旋转下保持等变。

---

## 四、SO2_Convolution：处理所有 m 阶

**文件**：`so2_layers.py:71-208`

### 4.1 初始化

```python
class SO2_Convolution(nn.Module):
    def __init__(self, sphere_channels, m_output_channels, lmax, mmax,
                 mappingReduced, internal_weights=True,
                 edge_channels_list=None, extra_m0_output_channels=None):

        # m=0：只有实部（标量），用普通 Linear
        num_channels_m0 = (lmax + 1) * sphere_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)  # so2_layers.py:115

        # m=1 到 mmax：每个 m 阶一个 SO2_m_Conv
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, mmax + 1):
            self.so2_m_conv.append(SO2_m_Conv(m, sphere_channels, m_output_channels, lmax, mmax))

        # 径向函数：将边的不变量特征映射为 channel 权重
        if not internal_weights:
            self.rad_func = RadialMLP(edge_channels_list)  # MLP: edge_dim → total_channels
```

`extra_m0_output_channels`：m=0 输出额外通道用于门控激活（`Edgewise.so2_conv_1` 使用）。

### 4.2 forward 详细步骤

```python
def forward(self, x, x_edge):
    # x:      [E, m序系数总数, sphere_channels]
    # x_edge: [E, edge_channels]（距离+元素类型等不变量）

    # ── 步骤1：l序 → m序重排 ──────────────────────────────────────
    x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)
    # x: [E, m序系数总数, sphere_channels]

    # ── 步骤2：径向函数（可选）────────────────────────────────────
    if self.rad_func is not None:
        x_edge = self.rad_func(x_edge)  # [E, total_input_channels]
        # 为每个 m 阶的每个输入 channel 提供边依赖的权重
    offset_rad = 0

    # ── 步骤3：m=0 处理（只有实部，无虚部）──────────────────────────
    x_0 = x[:, :self.mappingReduced.m_size[0], :].reshape(E, -1)   # [E, (L+1)*C]
    if self.rad_func is not None:
        x_edge_0 = x_edge[:, :self.fc_m0.in_features]
        x_0 = x_0 * x_edge_0   # 逐元素乘：径向权重调制
    x_0 = self.fc_m0(x_0)      # 线性变换
    # 分离 extra_m0（门控用）和普通 m=0 输出
    if self.extra_m0_output_channels is not None:
        x_0_extra = x_0[:, :extra_m0_output_channels]   # → 门控激活
        x_0 = x_0[:, extra_m0_output_channels:]
    out.append(x_0.view(E, -1, m_output_channels))

    # ── 步骤4：m>0 处理（实部+虚部 pair）────────────────────────────
    offset = self.mappingReduced.m_size[0]
    for m in range(1, mmax + 1):
        # 取出 m 阶的实部和虚部：共 2 * m_size[m] 行
        x_m = x[:, offset : offset + 2 * self.mappingReduced.m_size[m], :]
        x_m = x_m.reshape(E, 2, -1)  # [E, 2, m_size[m]*sphere_channels]

        # 径向权重调制
        if self.rad_func is not None:
            x_edge_m = x_edge[:, offset_rad : offset_rad + fc.in_features]
            x_edge_m = x_edge_m.reshape(E, 1, fc.in_features)  # 广播到 2 个半部分
            x_m = x_m * x_edge_m

        # SO(2) 旋转
        x_m = self.so2_m_conv[m - 1](x_m)             # → [E, 2, out_channels_half]
        out.append(x_m.view(E, -1, m_output_channels))

        offset += 2 * self.mappingReduced.m_size[m]
        offset_rad += fc.in_features

    # ── 步骤5：拼接并 m序 → l序恢复 ────────────────────────────────
    out = torch.cat(out, dim=1)
    out = torch.einsum("nac,ab->nbc", out, self.mappingReduced.to_m)
    # 注意：to_m 是正交矩阵，to_m.T = to_m 的逆（恢复 l 序）

    return out  # [E, 系数总数, m_output_channels]
```

### 4.3 径向权重的作用

`x_edge` 包含每条边的不变量信息（距离、源/目标元素），经过 `RadialMLP` 后产生逐通道标量权重，**调制**对应 m 阶的球谐系数。

```
x_edge [E, edge_dim]
  → RadialMLP（多层 Linear + SiLU）
  → 权重 [E, total_input_channels]（每条边一套权重）
  → 逐元素乘到对应 m 阶的展平特征
```

等价于：相同等变操作，权重依赖于化学环境（距离/元素），使模型能区分近距离和远距离邻居，以及不同元素对间的相互作用。

---

## 五、SO2_Linear：无径向权重的版本

**文件**：`so2_layers.py:211-322`

结构与 `SO2_Convolution` 几乎相同，差异：
- 默认 `internal_weights=False`，但 `rad_func` 仍可选
- m=0 用 `Linear`，m>0 用独立 `Linear`（无 SO(2) 旋转交叉项）

等价于：**每个 m 阶独立线性变换，不做实部/虚部混合**，保持等变性但不引入 SO(2) 旋转的相位混合。

用途：`Linear_Force_Head` 中的 `SO3_Linear` 和部分不需要旋转的通道映射。

---

## 六、在 Edgewise 中的完整使用

**文件**：`esen_block.py:60-81`

```python
# Edgewise.__init__
self.so2_conv_1 = SO2_Convolution(
    2 * sphere_channels,    # 输入：源+目标拼接
    hidden_channels,        # 输出通道
    lmax, mmax,
    internal_weights=False, # 使用边的径向函数调制
    edge_channels_list=...,
    extra_m0_output_channels=hidden_channels,  # 额外 m=0 用于门控
)

self.so2_conv_2 = SO2_Convolution(
    hidden_channels,        # 第一次卷积的输出
    sphere_channels,        # 恢复原始通道数
    lmax, mmax,
    internal_weights=True,  # 不用径向函数（权重已在 conv_1 中注入）
)
```

**两次卷积 + 门控激活的数据流：**

```
x_message [E, mmax_size, 2C]        ← 源+目标拼接，旋转到局部坐标系
    │
    ▼ so2_conv_1（径向权重调制）
x_message [E, mmax_size, hidden_C]  + x_0_extra [E, extra_C]
    │
    ▼ SeparableS2Activation（x_0_extra 作门控信号）
x_message [E, mmax_size, hidden_C]  ← 非线性
    │
    ▼ so2_conv_2（内部权重）
x_message [E, mmax_size, C]         ← 恢复 sphere_channels
```

`SeparableS2Activation`（`act_type="s2"`）：先在球面格点上做逐点 SiLU，再投影回球谐系数，是保等变性的非线性激活。

---

## 七、Wigner-D 矩阵：局部坐标系旋转

SO(2) 卷积的前提是将特征旋转到边的局部坐标系。这由 Wigner-D 矩阵完成。

**`common/rotation.py:19-75`**：
```python
def init_edge_rot_mat(edge_distance_vec):
    # 以边方向为 x 轴，构造正交局部坐标系
    norm_x = edge_distance_vec / |edge_distance_vec|  # 边方向（归一化）
    # 随机选取不与 norm_x 平行的向量，做 Gram-Schmidt 正交化
    norm_z = cross(norm_x, random_vec) / |...|
    norm_y = cross(norm_x, norm_z) / |...|
    # 构造旋转矩阵 R = [norm_z, norm_x, norm_y]^T
    edge_rot_mat = [norm_z, norm_x, norm_y]^T  # [E, 3, 3]
```

**`common/rotation.py:121-175`**：
```python
def rotation_to_wigner(edge_rot_mat, start_lmax, end_lmax, Jd):
    # 将 3D 旋转矩阵分解为 ZYZ 欧拉角 (α, β, γ)
    x = edge_rot_mat @ [0, 1, 0]
    alpha, beta = o3.xyz_to_angles(x)
    gamma = atan2(R[0,2], R[0,0])

    # 对每个 l 阶，计算 Wigner-D 矩阵块
    for lmax in range(start_lmax, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma, Jd)
        wigner[:, start:end, start:end] = block
```

`Jd.pt` 是预计算的 Wigner-D 参考矩阵（`J_l`），用于加速 ZYZ 分解：

$$D^{(l)}(\alpha,\beta,\gamma) = e^{-i\alpha J_z^{(l)}} \cdot J_l \cdot e^{-i\beta J_z^{(l)}} \cdot J_l \cdot e^{-i\gamma J_z^{(l)}}$$

**`rot_clip` 参数**：auto_grad 模式下开启，当边方向接近 y 轴时裁剪梯度，避免旋转矩阵对 y 方向的奇异性导致梯度爆炸。

---

## 八、SO3_Grid：球谐 ↔ 格点投影

**文件**：`common/so3.py:166-265`

```python
class SO3_Grid(nn.Module):
    def __init__(self, lmax, mmax, ...):
        # 预计算格点采样矩阵（e3nn 的 ToS2Grid / FromS2Grid）
        to_grid_mat   = einsum("mbi,am->bai", to_grid.shb, to_grid.sha)   # [lat, lon, coeffs]
        from_grid_mat = einsum("am,mbi->bai", from_grid.sha, from_grid.shb)

        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    def to_grid(self, embedding, lmax, mmax):
        # 球谐系数 → 球面格点值
        # embedding: [N, coeffs, C]
        # 返回: [N, lat, lon, C]
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        return torch.einsum("bai,zic->zbac", to_grid_mat, embedding)

    def from_grid(self, grid, lmax, mmax):
        # 球面格点值 → 球谐系数
        # grid: [N, lat, lon, C]
        # 返回: [N, coeffs, C]
        from_grid_mat = self.from_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        return torch.einsum("bai,zbac->zic", from_grid_mat, grid)
```

格点分辨率：`lat = 2*(lmax+1)`，`lon = 2*mmax+1`（lmax=mmax 时 lon=2*lmax+1）。

`GridAtomwise` 利用这对变换在格点空间做任意非线性 MLP，再投影回球谐系数，自动保持等变性（因为旋转在格点空间等价于坐标变换，而 MLP 是逐点操作，不破坏等变性）。

---

## 九、总结：SO(2) 卷积的等变性链路

```
全局坐标系节点特征 x [N, (L+1)², C]
    │
    ├── x[edge_index[0]], x[edge_index[1]] → x_message [E, (L+1)², 2C]
    │
    ├── wigner（局部坐标系旋转）:  bmm(wigner[:, out_mask, :], x_message)
    │     x_message → [E, mmax_size, 2C]  （等变：W·x）
    │
    ├── SO2_Convolution（m 阶独立操作）
    │     m=0:  Linear（标量，旋转不变）
    │     m>0:  SO2_m_Conv（等变：保持 m 阶 SO(2) 旋转行为）
    │     + 径向权重（不变量，不破坏等变性）
    │
    ├── wigner_inv（逆旋转回全局坐标系）: bmm(wigner_inv[:, :, out_mask], x_message)
    │     x_message → [E, (L+1)², C]  （等变：W⁻¹·SO2(W·x) = SO3等变）
    │
    └── index_add_（scatter sum，置换等变）
          new_embedding [N, (L+1)², C]  （全局坐标系，保持 SO(3) 等变）
```

整个操作链满足：若输入节点特征按全局旋转 R 变换，输出节点特征也按同样的 R 变换 — 这正是 SO(3) 等变消息传递。
