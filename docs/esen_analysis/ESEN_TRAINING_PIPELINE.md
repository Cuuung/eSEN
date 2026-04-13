# eSEN 完整训练流程：从命令行到模型组装

> **📖 文档导航**
> - 本文覆盖：命令行启动 → train 组装 → Hydra backbone/head 组装 → 数据输入
> - SO(2) 层细节见 [SO2_LAYER.md](./SO2_LAYER.md)

---

## 一、Registry：全局组件注册中心

### 1.1 设计思路

Registry 是一个**全局字典**，将字符串 key 映射到类。YAML 配置里写的 `trainer: mlip_trainer`、`model: esen_backbone` 等字符串，都通过 registry 找到对应的 Python 类再实例化。这样增加新模型/trainer 只需注册，无需修改框架代码。

源码：`src/fairchem/core/common/registry.py`

```python
# registry.py:60-68  内部存储结构
class Registry:
    mapping: ClassVar[NestedDict] = {
        "task_name_mapping":    {},   # "train"        → TrainTask
        "model_name_mapping":   {},   # "esen_backbone" → eSEN_Backbone
        "trainer_name_mapping": {},   # "mlip_trainer"  → MLIPTrainer
        "dataset_name_mapping": {},
        "loss_name_mapping":    {},
        "logger_name_mapping":  {},
        "state": {},
    }
```

### 1.2 注册：装饰器写法

用 `@registry.register_xxx("key")` 装饰器在**模块导入时**自动完成注册：

```python
# tasks/task.py:62-63
@registry.register_task("train")
class TrainTask(BaseTask): ...

# models/esen/trainers/trainer.py:37-38
@registry.register_trainer("mlip_trainer")
class MLIPTrainer(OCPTrainer): ...

# models/esen/esen.py:39-40
@registry.register_model("esen_backbone")
class eSEN_Backbone(nn.Module, GraphModelMixin): ...

# models/esen/esen.py:463
@registry.register_model("esen_mlp_energy_head")
class MLP_Energy_Head(nn.Module, HeadInterface): ...
```

装饰器的实现非常简单（`registry.py:159-162`）：

```python
def wrap(func):
    cls.mapping["model_name_mapping"][name] = func  # 把类存入字典
    return func                                      # 类本身不变
return wrap
```

### 1.3 查找与实例化

```python
# registry.py:293-294
@classmethod
def get_model_class(cls, name: str):
    return cls.get_class(name, "model_name_mapping")

# registry.py:266-278  get_class 支持两种形式：
# 1. 短名称：从字典里直接取
existing_mapping = cls.mapping[mapping_name].get(name, None)
# 2. 全限定名（fallback）：动态 import
#    e.g. "fairchem.core.models.esen.esen.eSEN_Backbone"
return _get_absolute_mapping(name)
```

使用时：

```python
trainer_cls = registry.get_trainer_class("mlip_trainer")   # 取到 MLIPTrainer 类
trainer = trainer_cls(config)                               # 实例化

model_cls = registry.get_model_class("esen_backbone")       # 取到 eSEN_Backbone 类
model = model_cls(**backbone_config)                        # 实例化
```

### 1.4 YAML key → 类的完整映射

| YAML 字段 | 值 | registry 字典 | 对应类 |
|---|---|---|---|
| `trainer:` | `mlip_trainer` | `trainer_name_mapping` | `MLIPTrainer` |
| `task.mode:` | `train` | `task_name_mapping` | `TrainTask` |
| `model.name:` | `hydra` | `model_name_mapping` | `HydraModel` |
| `backbone.model:` | `esen_backbone` | `model_name_mapping` | `eSEN_Backbone` |
| `heads.energy.module:` | `esen_mlp_energy_head` | `model_name_mapping` | `MLP_Energy_Head` |
| `heads.forces.module:` | `esen_linear_force_head` | `model_name_mapping` | `Linear_Force_Head` |
| `heads.stress.module:` | `esen_rank2_head` | `model_name_mapping` | `Rank2SymmetricTensorHead` |
| `logger:` | `tensorboard` | `logger_name_mapping` | `TensorboardLogger` |
| `dataset.train.format:` | `lmdb` | `dataset_name_mapping` | `LmdbDataset` |

### 1.5 注册触发时机

Registry 装饰器在**模块被 import 时**执行。fairchem 在启动时通过以下方式触发所有注册：

```python
# _cli.py 或 __init__.py 中有类似：
import fairchem.core.models      # 触发 models/ 下所有 @register_model
import fairchem.core.trainers    # 触发 @register_trainer
import fairchem.core.tasks       # 触发 @register_task
```

如果某个类没有被 import，即使写了装饰器，也不会出现在 registry 中 —— 这是新增组件时常见的遗漏点。

---

## 二、命令行启动与调用链

### 1.1 入口文件：`main_torchrun.py`

```python
# main_torchrun.py:1-22
config = build_config(args, override_args)

# torchrun 注入环境变量 RANK / LOCAL_RANK / WORLD_SIZE
config["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
config["distributed_backend"] = "nccl"
config["submit"] = False

Runner()(config)
```

启动命令：
```bash
torchrun --nproc_per_node=4 main_torchrun.py \
    --config-yml configs/eSEN_31M_Mptrj_direct_force.yml \
    --mode train
```

`torchrun` 负责启动多进程并向每个进程注入 `RANK/LOCAL_RANK/WORLD_SIZE`，`main_torchrun.py` 读取后交给 `Runner`。

### 1.2 完整调用链

```
Runner()(config)
  ├── distutils.setup()           # 初始化 NCCL 分布式通信
  ├── trainer = registry.get_trainer_class("mlip_trainer")(config)
  │                                # 实例化 MLIPTrainer（继承 OCPTrainer）
  ├── task = registry.get_task_class("train")(config)
  │                                # 实例化 TrainTask
  ├── task.setup(trainer)          # 加载 checkpoint（用户指定 or 崩溃恢复）
  └── task.run()                   # → trainer.train()
```

`task.setup()` 的 checkpoint 逻辑（`tasks/task.py:22-57`）：
```python
def setup(self, trainer) -> None:
    # 优先：用户 --checkpoint 参数
    if self.config["checkpoint"] is not None:
        self.trainer.load_checkpoint(checkpoint_path=..., inference_only=...)
    # 其次：checkpoint 目录存在历史文件 → 崩溃恢复
    elif os.path.isfile(self.chkpt_path) or (os.path.isdir(...) and len(...) > 0):
        self.trainer.load_checkpoint(checkpoint_path=self.chkpt_path)
```

`TrainTask.run()` 非常简单（`task.py:76-83`）：
```python
def run(self) -> None:
    self.trainer.train(disable_eval_tqdm=self.config.get("hide_eval_progressbar", False))
```

---

## 二、Train 组装流程

`MLIPTrainer` 继承链：`MLIPTrainer → OCPTrainer → BaseTrainer`

`BaseTrainer.__init__` 按顺序调用以下方法完成完整组装：

```
BaseTrainer.__init__()
  ├── load_seed_from_config()     # torch / numpy / random 设置随机种子
  ├── load_logger()               # TensorBoard 或 WandB
  ├── load_model()                # 实例化 HydraModel(backbone + heads)
  ├── load_datasets()             # 创建 train/val DataLoader
  ├── load_loss()                 # 解析 loss_functions 配置（MAE/L2MAE）
  ├── load_optimizer()            # AdamW + weight_decay
  └── load_extras()               # MLIPTrainer 重写：LRScheduler + EMA
```

### 2.1 MLIPTrainer.load_extras：epoch → step 转换

`trainers/trainer.py:44-88`：

```python
def load_extras(self) -> None:
    n_iter_per_epoch = len(self.train_loader)
    scheduler_params = self.config["optim"]["scheduler_params"]
    # 把所有 *_epochs 参数乘以每 epoch 的 step 数
    for k in scheduler_params:
        if "epochs" in k:
            scheduler_params[k] = int(scheduler_params[k] * n_iter_per_epoch)

    self.scheduler = LRScheduler(self.optimizer, self.config["optim"])

    # EMA 权重平均
    if self.ema_decay:
        self.ema = ExponentialMovingAverage(self.model.parameters(), self.ema_decay)
```

配置文件对应参数（`eSEN_31M_Mptrj_direct_force.yml:120-127`）：
```yaml
scheduler_params:
  lambda_type: cosine
  warmup_epochs: 0.1      # → warmup_epochs * n_iter_per_epoch steps
  warmup_factor: 0.2
  lr_min_factor: 0.01
```

### 2.2 Train 主循环关键逻辑（`trainer.py:106-231`）

```python
for epoch_int in range(start_epoch, max_epochs):
    for i in range(len(self.train_loader)):
        batch = next(train_loader_iter)

        # ── 1. OOM-safe forward（AMP 混精度）──
        try:
            with torch.autocast("cuda", enabled=self.scaler is not None):
                out = self._forward(batch)
                loss = self._compute_loss(out, batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom[0] = 1
                torch.cuda.empty_cache()

        # ── 2. 多卡 OOM 同步 ──
        dist.all_reduce(oom, op=dist.ReduceOp.MAX)
        if oom.item() > 0:
            self.optimizer.zero_grad()
            continue  # 所有卡跳过该 batch，记录到 oom_batches.log

        # ── 3. 反向传播 + NaN grad 检测 ──
        self._backward(loss)
        if self._nan_grad_detected:
            nan_stop = True
            break

        # ── 4. 滚动保存 checkpoint（保留最近 K 个）──
        # trainer.py:89-104
        if self.step % checkpoint_every == 0:
            self._save_rolling_checkpoint(epoch_int)

        # ── 5. 周期验证 ──
        if self.step % eval_every == 0:
            val_metrics = self.validate(split="val")
            self.update_best(primary_metric, val_metrics)

        # ── 6. LR 调度 ──
        self.scheduler.step()   # cosine: 每 step；ReduceLROnPlateau: 每次 val 后
```

`_save_rolling_checkpoint`（`trainer.py:89-104`）：
```python
def _save_rolling_checkpoint(self, epoch_int):
    keep_k = self.config["optim"].get("keep_last_k_checkpoints", 10)
    ckpt_file = f"checkpoint_epoch{epoch_int+1}_step{self.step}.pt"
    self.save(checkpoint_file=ckpt_file, training_state=True)
    self._ckpt_history.append(os.path.join(ckpt_dir, ckpt_file))
    while len(self._ckpt_history) > keep_k:
        os.remove(self._ckpt_history.pop(0))  # 删除最旧的
```

---

## 三、Hydra 模型组装：Backbone + Heads

### 3.1 YAML 配置（`eSEN_31M_Mptrj_direct_force.yml:76-108`）

```yaml
model:
  name: hydra                    # → 实例化 HydraModel
  pass_through_head_outputs: True
  backbone:
    model: esen_backbone         # → 实例化 eSEN_Backbone
    num_layers: 10
    lmax: 3
    mmax: 2
    sphere_channels: 128
    hidden_channels: 128
    edge_channels: 192
    cutoff: 6.0
    max_neighbors: 50
    distance_function: gaussian
    num_distance_basis: 10
    direct_forces: True
    regress_stress: False
    norm_type: rms_norm_sh
    act_type: s2
    mlp_type: grid
  heads:
    energy:
      module: esen_mlp_energy_head
    forces:
      module: esen_linear_force_head
    stress:
      module: esen_rank2_head
      output_name: stress
      num_mlp_layers: 2
```

`HydraModel` 组装顺序：
1. 用 `backbone.model` key 从 registry 找类，实例化 `eSEN_Backbone(…所有backbone参数…)`
2. 遍历 `heads`，对每个 head 调用 `registry.get_model_class(module)(backbone)`，**backbone 实例作为构造参数传入**，head 从中读取 `sphere_channels` 等超参

### 3.2 eSEN_Backbone 初始化（`esen.py:39-189`）

```python
@registry.register_model("esen_backbone")
class eSEN_Backbone(nn.Module, GraphModelMixin):
    def __init__(self, ...):
        # 1. Wigner-D 矩阵预计算（存入 buffer，不参与梯度）
        Jd_list = torch.load("Jd.pt")
        for l in range(lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])   # esen.py:91-93

        # 2. 球谐系数索引映射（l 序 ↔ m 序）
        self.mappingReduced = CoefficientMapping(lmax, mmax)  # esen.py:95

        # 3. SO3 格点（GridAtomwise 用）
        self.SO3_grid["lmax_lmax"] = SO3_Grid(lmax, lmax)    # 节点全阶
        self.SO3_grid["lmax_mmax"] = SO3_Grid(lmax, mmax)    # 边截断到 mmax

        # 4. 原子 embedding：标量，只填入 l=0 分量
        self.sphere_embedding = nn.Embedding(max_num_elements, sphere_channels)  # esen.py:107

        # 5. 边的不变量 embedding（距离 + 源/目标元素）
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_distance_basis)
        self.source_embedding = nn.Embedding(max_num_elements, edge_channels)    # esen.py:135
        self.target_embedding = nn.Embedding(max_num_elements, edge_channels)

        # 6. 初始等变 embedding：将边信息注入节点球谐特征
        self.edge_degree_embedding = EdgeDegreeEmbedding(...)  # esen.py:146

        # 7. 堆叠 num_layers 个 eSEN_Block
        self.blocks = nn.ModuleList([eSEN_Block(...) for _ in range(num_layers)])  # esen.py:167

        # 8. 最终归一化
        self.norm = get_normalization_layer(norm_type, lmax, sphere_channels)
```

### 3.3 eSEN_Backbone.forward 数据流（`esen.py:226-364`）

```
输入 data_dict（batch）
│
├── [auto_grad 模式] 注入可微位移 displacement → pos/cell 变换  esen.py:235-258
│
├── generate_graph()                          esen.py:263-264
│     ├── edge_index       [2, E]
│     ├── edge_distance    [E]
│     └── edge_distance_vec [E, 3]
│
├── get_rotmat_and_wigner()                   esen.py:290-292
│     ├── init_edge_rot_mat()   → 每条边局部坐标系 R [E, 3, 3]
│     ├── rotation_to_wigner()  → wigner     [E, (L+1)², (L+1)²]
│     └──                       → wigner_inv = wigner.T
│
├── 初始化节点特征 x_message [N, (L+1)², C]   esen.py:298-305
│     x_message[:, 0, :] = sphere_embedding(Z)   ← 只有 l=0 非零
│
├── EdgeDegreeEmbedding：将边信息注入球谐特征   esen.py:308-324
│     x_edge = cat(distance_emb, source_emb, target_emb)  [E, 10+192+192]
│     x_message = edge_degree_embedding(x_message, x_edge, ...)
│
├── 循环 num_layers 个 eSEN_Block             esen.py:330-352
│     x_message = block(x_message, x_edge, edge_distance, edge_index, wigner, wigner_inv)
│
├── final norm                                esen.py:355
│
└── 返回 {"node_embedding": x_message, "edge_index": ..., ...}
```

### 3.4 eSEN_Block（`esen_block.py:210-302`）

Pre-Norm Transformer 结构：

```python
def forward(self, x, x_edge, edge_distance, edge_index, wigner, wigner_inv, ...):
    # --- 分支1：边消息传递（等价于 attention） ---
    x_res = x
    x = self.norm_1(x)                              # RMS Norm（等变版）
    x = self.edge_wise(x, x_edge, ...)              # SO2 卷积聚合邻居
    x = x + x_res                                   # 残差

    # --- 分支2：节点 MLP ---
    x_res = x
    x = self.norm_2(x)
    x = self.atom_wise(x)                           # GridAtomwise 或 SpectralAtomwise
    return x + x_res
```

**Edgewise（`esen_block.py:92-131`）详细步骤：**

```python
# 1. 拼接源+目标节点特征
x_source = x[edge_index[0]]                        # [E, (L+1)², C]
x_message = cat((x_source, x_target), dim=2)       # [E, (L+1)², 2C]

# 2. 旋转到边的局部坐标系（等变性关键）
x_message = bmm(wigner[:, out_mask, :], x_message) # [E, mmax_size, 2C]

# 3. 第一次 SO2 卷积（带径向权重）+ 门控激活
x_message, x_0_gate = self.so2_conv_1(x_message, x_edge)
x_message = self.act(x_0_gate, x_message)          # S2 或 Gate 激活

# 4. 第二次 SO2 卷积（内部权重）
x_message = self.so2_conv_2(x_message, x_edge)

# 5. 旋转回全局坐标系
x_message = bmm(wigner_inv[:, :, out_mask], x_message)

# 6. scatter sum：每个目标节点聚合所有入边消息
new_embedding.index_add_(0, edge_index[1], x_message)
```

**GridAtomwise（`esen_block.py:177-207`）：**

```python
def forward(self, x):
    # 球谐系数 → 球面格点（离散采样球面函数值）
    x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, lmax, lmax)  # [N, lat, lon, C]
    # 格点上做普通 MLP（逐点非线性，无需等变操作）
    x_grid = self.grid_mlp(x_grid)   # Linear-SiLU-Linear-SiLU-Linear
    # 格点 → 球谐系数（逆球面调和变换）
    return self.SO3_grid["lmax_lmax"].from_grid(x_grid, lmax, lmax)
```

这是 eSEN 相比 EquiformerV2 的核心设计差异：用**格点表示**做 MLP 代替谱空间张量积，复杂度从 O(L⁶) 降到 O(L²·grid²)。

### 3.5 三个 Head 实现

**MLP_Energy_Head（`esen.py:463-498`）：**
```python
def forward(self, data_dict, emb):
    # 只取 l=0 分量（标量通道）→ 3层 MLP → 原子能量
    node_energy = self.energy_block(
        emb["node_embedding"].narrow(1, 0, 1).squeeze()  # [N, 1, C] → [N, C]
    ).view(-1, 1, 1)
    # scatter sum：∑原子能量 = 结构能量
    energy.index_add_(0, data_dict["batch"], node_energy.view(-1))
    # energy_block: Linear(C,C) → SiLU → Linear(C,C) → SiLU → Linear(C,1)
```

**Linear_Force_Head（`esen.py:501-511`）：**
```python
def forward(self, data_dict, emb):
    # 取前4个球谐系数（l=0标量1个 + l=1矢量3个）
    forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))  # SO3_Linear
    # 只保留 l=1（3个分量 = xyz 力向量）
    forces = forces.narrow(1, 1, 3).view(-1, 3)
    # SO3_Linear 保证输出 l=1 分量在旋转下按矢量变换（等变自动满足）
```

**MLP_EFS_Head（auto_grad 模式，`esen.py:399-460`）：**
```python
@conditional_grad(torch.enable_grad())
def forward(self, data, emb):
    energy = ...  # 与 MLP_Energy_Head 相同

    # backbone 已在 forward 中注入 displacement（可微变量）
    grads = torch.autograd.grad(
        [energy.sum()],
        [data["pos"], emb["displacement"]],
        create_graph=self.training,
    )
    forces = -grads[0]                      # F = -∂E/∂r
    virial = grads[1].view(-1, 3, 3)
    volume = torch.det(data["cell"]).abs()
    stress = virial / volume.view(-1, 1, 1) # σ = virial / V
```

---

## 四、数据输入流

### 4.1 完整流程

```
LMDB 文件（MPtrj/lmdb/train/）
  ↓
LmdbDataset.__getitem__(idx)
  ├── 读取 Data(pos, atomic_numbers, cell, energy, forces, stress, ...)
  └── DataTransforms.normalizer
        energy_norm  = (energy - mean) / stdev    # mean=-188.81, stdev=190.31
        forces_norm  = forces / stdev             # stdev=0.8079
        stress_norm  = (stress - mean) / stdev
  ↓
data_list_collater()  →  batch（PyG Batch 对象）
  ├── batch.pos            [ΣN, 3]     所有结构原子坐标拼接
  ├── batch.atomic_numbers [ΣN]
  ├── batch.batch          [ΣN]        每个原子属于哪个结构（0到B-1）
  ├── batch.natoms         [B]         每个结构的原子数
  ├── batch.cell           [B, 3, 3]   晶胞矩阵
  └── batch.energy / forces / stress（归一化后的标签）
```

### 4.2 YAML 数据配置（`eSEN_31M_Mptrj_direct_force.yml:1-33`）

```yaml
dataset:
  train:
    format: lmdb
    src: /path/to/MPtrj/lmdb/train
    transforms:
      normalizer:
        energy:
          mean: -188.811416
          stdev: 190.308646
        forces:
          mean: 0.000000
          stdev: 0.807904
        stress:
          mean: 0.956625
          stdev: 38.581393
```

### 4.3 BalancedBatchSampler：按原子数负载均衡

```yaml
optim:
  batch_size: 32
  load_balancing: atoms   # 按原子数而非结构数平衡 batch
  num_workers: 4
```

按原子数而非结构数组 batch，避免包含大晶胞的 batch 导致 OOM。

### 4.4 OTF 建图（`esen.py:263-264`）

```python
if self.otf_graph:
    graph_dict = self.generate_graph(data_dict)
```

`otf_graph=True` 时在 forward 中实时计算邻居（基于 cutoff=6.0Å + PBC），不需要预存边信息。`use_pbc_single=True` 防止小晶格产生自环导致卡死。

### 4.5 损失函数配置（`eSEN_31M_Mptrj_direct_force.yml:51-61`）

```yaml
loss_functions:
  - energy:
      fn: mae
      coefficient: 20
  - forces:
      fn: l2mae        # sqrt(mse)，对力向量整体长度敏感
      coefficient: 20
  - stress:
      fn: mae
      coefficient: 5
```

总损失 = `20·MAE(E) + 20·L2MAE(F) + 5·MAE(σ)`

---

## 五、整体数据流总结

```
torchrun
  └── main_torchrun.py → Runner → MLIPTrainer.train()
                                        │
                        LMDB → batch(pos, Z, cell, E/F/S, normalizer)
                                        │
                        eSEN_Backbone.forward()
                          ├── generate_graph → edge_index, dist_vec  [E]
                          ├── Wigner-D matrices（按边方向旋转矩阵）  [E, D, D]
                          ├── sphere_embedding(Z) → x[N, (L+1)², C]
                          ├── EdgeDegreeEmbedding（初始等变注入）
                          └── 10× eSEN_Block
                                ├── norm → Edgewise（SO2卷积）→ 残差
                                └── norm → GridAtomwise（MLP）→ 残差
                                        │
                        HydraModel heads
                          ├── MLP_Energy_Head  → E [B]
                          ├── Linear_Force_Head → F [ΣN, 3]
                          └── Rank2_Head        → σ [B, 9]
                                        │
                        loss = 20·MAE(E) + 20·L2MAE(F) + 5·MAE(σ)
                          └── backward → AdamW → cosine LR → EMA
```
