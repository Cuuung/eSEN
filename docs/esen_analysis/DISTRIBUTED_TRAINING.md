# fairchem 分布式训练详解

> **📖 文档导航**
> - **上一篇**: [ESEN Head 层详解](./ESEN_HEAD_LAYERS.md)
> - **返回**: [文档总览](./README.md)
>
> 本文档详细介绍 fairchem 的分布式训练机制，包括 DDP 原理、torchrun 接口和 SLURM 调度。

---

## 1. 分布式训练概述

### 1.1 为什么需要分布式训练？

深度学习模型的训练需要大量计算资源。当单 GPU 无法满足需求时，需要使用多 GPU 甚至多节点集群进行分布式训练。

fairchem 支持两种分布式训练方式：
- **数据并行（Data Parallelism）**：每个 GPU 持有完整模型副本，处理不同数据
- **混合并行**：结合数据并行和模型并行

### 1.2 fairchem 的分布式架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        多节点集群                                       │
│                                                                     │
│  ┌───────────────────┐     ┌───────────────────┐                  │
│  │    Node 0         │     │    Node 1         │                  │
│  │  ┌─────────────┐  │     │  ┌─────────────┐  │                  │
│  │  │   GPU 0     │  │     │  │   GPU 0     │  │                  │
│  │  │ (Rank 0)    │  │     │  │ (Rank 2)    │  │                  │
│  │  └─────────────┘  │     │  └─────────────┘  │                  │
│  │  ┌─────────────┐  │     │  ┌─────────────┐  │                  │
│  │  │   GPU 1     │  │     │  │   GPU 1     │  │                  │
│  │  │ (Rank 1)    │  │     │  │ (Rank 3)    │  │                  │
│  │  └─────────────┘  │     │  └─────────────┘  │                  │
│  └───────────────────┘     └───────────────────┘                  │
│           │                         │                              │
│           └────────────┬────────────┘                              │
│                        │                                            │
│                        ▼                                            │
│              ┌─────────────────────┐                                │
│              │  NCCL / GLOO 通信    │                                │
│              │  (GPU/CPU 通信)      │                                │
│              └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘

每个 GPU 上的进程:
  - 持有完整模型副本
  - 处理 batch / num_gpus 的数据
  - 定期同步梯度
```

---

## 2. DDP (DistributedDataParallel) 原理

### 2.1 DDP vs DataParallel

| 特性 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|-------------------------------|
| 并行方式 | 单进程多线程 | 多进程 |
| 通信方式 | 主进程收集 | 点对点通信 |
| 通信量 | 每次前反向后同步 | 只同步梯度 |
| 效率 | 较低（有瓶颈） | 高 |
| 负载均衡 | 自动 | 需要设置 |

### 2.2 DDP 工作原理

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DDP 训练步骤                                  │
└─────────────────────────────────────────────────────────────────────┘

1. 初始化 (torch.distributed.init_process_group)
   │
   ▼
2. 模型包装 (DistributedDataParallel)
   │
   ├─ 每个 rank 复制模型权重
   ├─ 注册梯度钩子用于 AllReduce
   └─ 记录参数柄用于集体通信
   │
   ▼
3. 前向传播 (各 rank 独立)
   │
   ├─ 每个 rank 用自己的数据前向传播
   └─ 无通信开销
   │
   ▼
4. 反向传播 (各 rank 独立)
   │
   ├─ 计算本地梯度
   └─ 触发梯度同步
   │
   ▼
5. 梯度同步 (AllReduce)
   │
   ├─ 所有 rank 的梯度求平均
   ├─ grad[i] = (grad[0] + grad[1] + ... + grad[n-1]) / n
   └─ 使用 NCCL/GLOO 集合通信
   │
   ▼
6. 参数更新
   │
   └─ 每个 rank 独立更新（梯度已同步）
```

### 2.3 梯度同步详解

```python
# DDP 梯度同步的核心代码逻辑（伪代码）

class DistributedDataParallel(nn.Module):
    def __init__(self, model):
        self.model = model
        # 为每个参数注册梯度钩子
        for param in model.parameters():
            param.register_hook(self._allreduce_hook)
    
    def _allreduce_hook(self, grad):
        # 在梯度计算后自动调用
        return dist.all_reduce(grad, op=dist.ReduceOp.SUM) / world_size
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
```

**AllReduce 操作**：
```
Rank 0: grad = [1.0, 2.0, 3.0]
Rank 1: grad = [2.0, 4.0, 6.0]
         ↓ AllReduce (SUM)
Result:  [3.0, 6.0, 9.0]
         ↓ 除以 world_size (2)
Average: [1.5, 3.0, 4.5]
```

### 2.4 fairchem 中 DDP 的使用

```python
# src/fairchem/core/trainers/base_trainer.py

from torch.nn.parallel.distributed import DistributedDataParallel

class BaseTrainer:
    def __init__(self, ..., local_rank: int, ...):
        # 初始化分布式环境
        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[local_rank],  # 本地 GPU
                output_device=local_rank,
            )
        else:
            self.model = self.model  # 非分布式模式
    
    def _unwrap_model(self):
        """获取原始模型（去除 DDP 包装）"""
        module = self.model
        while isinstance(module, DistributedDataParallel):
            module = module.module
        return module
```

---

## 3. torchrun 接口调用

### 3.1 torchrun 简介

`torchrun` 是 PyTorch 2.0 引入的新一代分布式训练启动器，替代了旧的 `torch.distributed.launch`。

**核心功能**：
- 自动设置分布式环境变量（`WORLD_SIZE`, `RANK`, `LOCAL_RANK` 等）
- 支持弹性训练（Elastic）
- 简化多节点启动

### 3.2 基本用法

```bash
# 单节点多 GPU
torchrun --nproc_per_node=4 train.py

# 多节点（2节点，每节点4 GPU）
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=job123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.1:29500 \
    train.py

# 常用参数
torchrun \
    --standalone                    # 单节点自动设置 master
    --nnodes=1                     # 节点数
    --nproc_per_node=4             # 每节点 GPU 数
    --max_restarts=3               # 最大重启次数
    --rdzv_id=unique_job_id       # 作业唯一标识
    --rdzv_backend=c10d             # 通信后端
    --rdzv_endpoint=host:port     # rendezvous 服务器地址
    train.py
```

### 3.3 fairchem 中的 torchrun 调用

```bash
# fairchem 推荐的多 GPU 训练命令
torchrun --nproc_per_node=8 \
    --standalone \
    -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode train
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `--nproc_per_node` | 每个节点的 GPU 数量 |
| `--standalone` | 单节点模式，自动设置 master |
| `-m fairchem.core._cli` | 以模块方式运行 CLI |
| `--config-yml` | 配置文件路径 |
| `--mode train` | 训练模式 |

### 3.4 环境变量

torchrun 会自动设置以下环境变量：

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `WORLD_SIZE` | 总进程数 | 8 |
| `RANK` | 全局进程编号 | 0-7 |
| `LOCAL_RANK` | 本地 GPU 编号 | 0-7 |
| `MASTER_ADDR` | 主节点地址 | 192.168.1.1 |
| `MASTER_PORT` | 主节点端口 | 29500 |

```
Node 0 上的 4 个进程:
  LOCAL_RANK=0, RANK=0
  LOCAL_RANK=1, RANK=1
  LOCAL_RANK=2, RANK=2
  LOCAL_RANK=3, RANK=3

Node 1 上的 4 个进程:
  LOCAL_RANK=0, RANK=4
  LOCAL_RANK=1, RANK=5
  LOCAL_RANK=2, RANK=6
  LOCAL_RANK=3, RANK=7
```

---

## 4. SLURM 调度集成

### 4.1 SLURM 简介

SLURM (Simple Linux Utility for Resource Management) 是超级计算机常用的作业调度系统。

### 4.2 fairchem 的 SLURM 支持

fairchem 使用 `submitit` 库实现 SLURM 集成：

```python
# src/fairchem/core/_cli.py 中的关键代码

from submitit import AutoExecutor

def submit_slurm_job(config, args):
    # 创建执行器
    executor = AutoExecutor(folder=args.logdir / "%j")
    
    # 配置 SLURM 参数
    executor.update_parameters(
        name=args.identifier,                    # 作业名称
        mem_gb=args.slurm_mem,                  # 内存 (GB)
        timeout_min=args.slurm_timeout * 60,    # 超时 (分钟)
        slurm_partition=args.slurm_partition,   # 分区
        gpus_per_node=args.num_gpus,            # 每节点 GPU 数
        nodes=args.num_nodes,                   # 节点数
        slurm_qos=args.slurm_qos,               # QoS
        slurm_account=args.slurm_account,       # 账户
    )
    
    # 提交作业
    job = executor.submit(Runner(), config)
```

### 4.3 SLURM 脚本示例

```bash
#!/bin/bash
#SBATCH --job-name=fairchem_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# 加载环境
source ~/.bashrc
conda activate fairchem

# 获取节点列表
echo "Nodes: $SLURM_JOB_NODELIST"

# 启动分布式训练
torchrun --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1):29500 \
    -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode train
```

### 4.4 fairchem 的 submitit 封装

```python
# src/fairchem/core/_cli_hydra.py

from dataclasses import dataclass
from submitit import AutoExecutor

@dataclass
class SlurmConfig:
    """SLURM 配置"""
    timeout_hr: int = 72              # 超时（小时）
    partition: str = "compute"        # 分区
    nodes: int = 1                    # 节点数
    ranks_per_node: int = 1           # 每节点 GPU 数
    cpus_per_task: int = 10           # 每任务 CPU 数
    mem_gb: int = 80                  # 内存 (GB)
    qos: str = None                   # QoS
    account: str = None               # 账户
    array_parallelism: int = 1        # 数组作业并行度

def run_on_slurm(cfg):
    executor = AutoExecutor(folder=cfg.job.metadata.log_dir)
    executor.update_parameters(
        name=cfg.job.run_name,
        timeout_min=cfg.scheduler.slurm.timeout_hr * 60,
        slurm_partition=cfg.scheduler.slurm.partition,
        nodes=cfg.scheduler.num_nodes,
        gpus_per_node=cfg.scheduler.ranks_per_node,
        cpus_per_task=cfg.scheduler.slurm.cpus_per_task,
    )
    job = executor.submit(Runner(), cfg)
```

### 4.5 SLURM 环境变量

| 环境变量 | 说明 |
|---------|------|
| `SLURM_JOB_ID` | 作业 ID |
| `SLURM_JOB_NODELIST` | 节点列表 |
| `SLURM_NNODES` | 节点数 |
| `SLURM_NTASKS_PER_NODE` | 每节点任务数 |
| `SLURM_NODEID` | 当前节点 ID |
| `SLURM_LOCALID` | 节点内进程 ID |
| `SLURM_PROCID` | 全局进程 ID |
| `SLURM_ARRAY_TASK_ID` | 数组任务 ID |

---

## 5. fairchem 分布式训练完整流程

### 5.1 调用链

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CLI 入口 (main.py / _cli.py)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Runner (main_torchrun.py)                    │
│  - torchrun 启动时调用                                              │
│  - 创建 trainer context                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   new_trainer_context() (utils.py)                  │
│  - 初始化分布式环境                                                 │
│  - 创建 Trainer 实例                                                 │
│  - 返回 context manager                                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BaseTrainer.__init__()                           │
│  - 初始化模型                                                       │
│  - 加载数据集                                                       │
│  - 创建 DataLoader                                                  │
│  - (如果分布式) 包装 DDP                                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     BaseTask.setup() (task.py)                      │
│  - trainer.setup() 设置训练器                                        │
│  - 加载 checkpoint（如果有）                                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BaseTask.run()                                 │
│  - 调用 trainer.train()                                             │
│  - 开始训练循环                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 关键代码

```python
# src/fairchem/core/_cli.py

def main():
    # 解析命令行参数
    parser = Flags().build()
    args, config = parser.parse()
    
    # 创建 runner
    runner = Runner()
    
    # 提交 SLURM 或直接运行
    if args.submit:
        # SLURM 提交模式
        executor = AutoExecutor(...)
        jobs = executor.map_array(runner, configs)
    else:
        # 本地运行模式
        runner(config)

# torchrun 会自动调用 main()
# torchrun 会设置必要的环境变量并启动多个进程

# src/fairchem/core/_cli.py

class Runner:
    def __call__(self, config: dict):
        with new_trainer_context(config=config) as ctx:
            # ctx.trainer: BaseTrainer 实例
            # ctx.task: BaseTask 实例
            
            ctx.task.setup(ctx.trainer)
            ctx.task.run()

# src/fairchem/core/common/utils.py

@contextmanager
def new_trainer_context(*, config: dict):
    # 创建 trainer
    trainer = create_trainer(config)
    
    # 创建 task
    task = create_task(config)
    
    yield _TrainingContext(config=config, task=task, trainer=trainer)
```

### 5.3 分布式环境初始化

```python
# fairchem 不需要手动初始化 torch.distributed
# torchrun 会自动设置环境变量并调用用户的 main() 函数

# fairchem 使用 distutils 模块检查分布式环境

from fairchem.core.common import distutils

# 检查是否在分布式环境中
if distutils.initialized():
    world_size = distutils.get_world_size()  # 总进程数
    rank = distutils.get_rank()              # 当前进程编号
    local_rank = distutils.get_local_rank()  # 本地 GPU 编号
    
    # 设置设备
    device = f"cuda:{local_rank}"
    model = model.to(device)
```

### 5.4 数据加载与分发

```python
# DataLoader 的 Sampler 处理数据分发

class BaseTrainer:
    def load_datasets(self):
        # Sampler 根据 rank 分发数据
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,  # 关键：Sampler 控制数据分发
            num_workers=num_workers,
            collate_fn=self.collater,
        )
```

**数据分发示意**：
```
Batch size = 32, World size = 4

Epoch 0:
  Rank 0: 数据 0-7    (samples 0,4,8,12,16,20,24,28)
  Rank 1: 数据 8-15   (samples 1,5,9,13,17,21,25,29)
  Rank 2: 数据 16-23  (samples 2,6,10,14,18,22,26,30)
  Rank 3: 数据 24-31  (samples 3,7,11,15,19,23,27,31)

Epoch 1 (shuffle 后):
  Rank 0: 数据 0-7
  Rank 1: 数据 8-15
  ...
```

---

## 6. 常见问题与解决方案

### 6.1 梯度不同步

**症状**：训练 loss 震荡或发散

**原因**：
- 混合精度训练时，梯度缩放器不同步
- 某些参数未注册到 DDP

**解决方案**：
```python
# 确保所有参数都在模型中
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Training param: {name}")
```

### 6.2 显存不均衡

**症状**：部分 GPU OOM，其他 GPU 显存空闲

**原因**：
- 不同 batch 的结构大小不同
- Batch 内 padding 不均匀

**解决方案**：
```yaml
# 使用 packed batch 或按大小排序
optim:
  batch_size: 16
  # 数据按原子数排序（需要配置 sampler）
```

### 6.3 通信瓶颈

**症状**：GPU 利用率低，训练速度慢

**原因**：
- 大模型导致梯度通信量大
- NCCL 配置不当

**解决方案**：
```bash
# 使用 NCCL 通信后端
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
```

### 6.4 Checkpoint 加载

```python
# 确保只在一个 rank 保存 checkpoint
if distutils.is_master():
    trainer.save_checkpoint("checkpoint.pt")

# 所有 rank 加载
trainer.load_checkpoint("checkpoint.pt")
```

---

## 7. 性能优化建议

### 7.1 通信计算重叠

```python
# 使用 DDP 的向后钩子
# 在反向传播时同时进行梯度同步

model = DistributedDataParallel(model, gradient_as_bucket_view=True)
```

### 7.2 梯度累积

```bash
# 当 GPU 显存不足时，使用梯度累积
# effective_batch_size = batch_size * gradient_accumulation_steps
torchrun --nproc_per_node=4 train.py \
    --gradient_accumulation_steps=4
```

### 7.3 混合精度训练

```python
# 启用 AMP 加速
trainer.scaler = torch.cuda.amp.GradScaler()

# 前向传播
with torch.autocast("cuda", enabled=trainer.scaler is not None):
    outputs = model(batch)
```

### 7.4 异步数据加载

```python
# 增加 num_workers
DataLoader(dataset, num_workers=8, pin_memory=True)
```

---

## 8. 源码文件索引

| 文件 | 内容 |
|------|------|
| `src/fairchem/core/_cli.py` | CLI 入口，SLURM 提交 |
| `src/fairchem/core/_cli_hydra.py` | Hydra 风格的 SLURM 配置 |
| `src/fairchem/core/trainers/base_trainer.py` | Trainer 基类，DDP 包装 |
| `src/fairchem/core/common/utils.py` | `new_trainer_context()` |
| `src/fairchem/core/common/slurm.py` | SLURM 辅助函数 |
| `src/fairchem/core/common/distutils.py` | 分布式工具函数 |

---

## 9. 推荐训练配置

### 9.1 单节点 8 GPU

```bash
torchrun --nproc_per_node=8 --standalone \
    -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode train
```

### 9.2 多节点集群

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00

torchrun --nnodes=4 --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1):29500 \
    -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode train
```

### 9.3 使用 fairchem submitit

```python
# 使用 Python API 提交
from fairchem.core._cli import Runner

config = {...}  # 配置文件
executor = AutoExecutor(folder="./logs/%j")
executor.update_parameters(
    name="fairchem_train",
    nodes=4,
    gpus_per_node=8,
    timeout_min=72*60,
)
job = executor.submit(Runner(), config)
print(f"Job ID: {job.job_id}")
```
