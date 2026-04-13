# CLI 入口点详细分析

> **📖 文档导航**
> - **上一篇**: [文档总览](./README.md)
> - **下一篇**: [ESEN 模型调用流程](./ESEN_CALL_FLOW.md)
>
> 本文档介绍程序入口、配置加载和任务调度机制。

---

## 概述

本文档详细解析 fairchem 的 CLI 入口点代码调用流程：

```python
main() 
  → build_config(args, override_args)  # 加载 YAML 配置
  → Runner()(config)  # 启动训练
    → new_trainer_context(config)  # 创建 trainer 上下文
      → task.setup(trainer)  # 设置任务
      → task.run()  # 运行训练
```

---

## 1. main() 函数

**文件位置**: `src/fairchem/core/_cli.py`

```python
def main(
    args: argparse.Namespace | None = None, 
    override_args: list[str] | None = None
):
    """Run the main fairchem program."""
    setup_logging()

    if args is None:
        parser: argparse.ArgumentParser = flags.get_parser()
        args, override_args = parser.parse_known_args()

    # 确保至少有一个 GPU
    assert args.num_gpus > 0, "num_gpus must be at least 1"
    
    # 1. 构建配置
    config = build_config(args, override_args)

    # 2. 提交到集群 or 本地运行
    if args.submit:  # Run on cluster
        # ... Slurm 提交逻辑
        executor = AutoExecutor(folder=args.logdir / "%j", slurm_max_num_timeout=3)
        jobs = executor.map_array(Runner(), configs)
    else:  # Run locally
        if args.num_gpus > 1:
            # 多 GPU: 使用 elastic_launch
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=args.num_gpus,
                rdzv_backend="c10d",
                max_restarts=0,
            )
            elastic_launch(launch_config, runner_wrapper)(config)
        else:
            # 单 GPU: 直接运行
            distutils.setup_env_local()
            runner_wrapper(config)
```

### 关键步骤说明

| 步骤 | 函数 | 作用 |
|------|------|------|
| 1 | `setup_logging()` | 配置日志系统 |
| 2 | `flags.get_parser()` | 获取命令行参数解析器 |
| 3 | `build_config()` | 加载并构建配置字典 |
| 4 | `elastic_launch()` 或 `runner_wrapper()` | 启动训练 |

---

## 2. build_config() 函数

**文件位置**: `src/fairchem/core/common/utils.py`

```python
def build_config(args, args_override, include_paths=None):
    # 1. 加载 YAML 配置文件
    config, duplicates_warning, duplicates_error = load_config(
        args.config_yml, include_paths=include_paths
    )
    
    # 2. 处理配置警告和错误
    if len(duplicates_warning) > 0:
        logging.warning(f"Overwritten config parameters: {duplicates_warning}")
    if len(duplicates_error) > 0:
        raise ValueError(f"Conflicting parameters: {duplicates_error}")

    # 3. 添加运行时参数
    config["mode"] = args.mode              # train/predict/validate
    config["identifier"] = args.identifier  # 实验标识符
    config["timestamp_id"] = args.timestamp_id
    config["seed"] = args.seed
    config["is_debug"] = args.debug
    config["run_dir"] = args.run_dir
    config["print_every"] = args.print_every
    config["amp"] = args.amp                # 自动混合精度
    config["checkpoint"] = args.checkpoint  # 检查点路径
    config["cpu"] = args.cpu
    
    # 4. 分布式训练配置
    config["submit"] = args.submit
    config["summit"] = args.summit
    config["world_size"] = args.num_nodes * args.num_gpus
    config["distributed_backend"] = "gloo" if args.cpu else "nccl"
    config["gp_gpus"] = args.gp_gpus

    # 5. 处理命令行覆盖参数
    if args_override != []:
        overrides = create_dict_from_args(args_override)
        config, _ = merge_dicts(config, overrides)

    return config
```

### load_config() 详细流程

```python
def load_config(path, files_previously_included=None, include_paths=None):
    """递归加载 YAML 配置，支持 includes 机制"""
    
    # 1. 读取 YAML 文件
    with open(path) as fp:
        current_config = yaml.load(fp, Loader=UniqueKeyLoader)

    # 2. 处理 includes（继承其他配置文件）
    includes_listed_in_config = current_config.pop("includes", [])
    
    config_from_includes = {}
    for include in includes_listed_in_config:
        include_config, _, _ = load_config(include_filename, ...)
        config_from_includes = merge_dicts(config_from_includes, include_config)

    # 3. 合并配置（当前文件优先级高于 includes）
    config_from_includes, _ = merge_dicts(config_from_includes, current_config)
    
    return config_from_includes, duplicates_warning, duplicates_error
```

---

## 3. Runner 类

**文件位置**: `src/fairchem/core/_cli.py`

```python
class Runner(Checkpointable):
    def __init__(self) -> None:
        self.config = None

    def __call__(self, config: dict) -> None:
        """核心执行方法"""
        with new_trainer_context(config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer
            self.task.setup(self.trainer)  # 设置任务
            self.task.run()                 # 运行任务

    def checkpoint(self, *args, **kwargs):
        """Slurm 抢占时保存检查点"""
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        return DelayedSubmission(new_runner, self.config)
```

---

## 4. new_trainer_context() 上下文管理器

**文件位置**: `src/fairchem/core/common/utils.py`

```python
@contextmanager
def new_trainer_context(*, config: dict[str, Any]):
    from fairchem.core.common import distutils, gp_utils
    
    # 1. 设置导入（注册所有模型、训练器、数据集）
    setup_imports(config)
    
    # 2. 设置分布式环境
    if config["mode"] == "train":
        distutils.setup(config)
        if config.get("gp_gpus", None) is not None:
            gp_utils.setup_graph_parallel_groups(config)
    
    # 3. 从注册表获取训练器类
    trainer_name = config.get("trainer", "ocp")  # 默认 ocp
    trainer_cls = registry.get_trainer_class(trainer_name)
    
    # 4. 实例化训练器
    trainer = trainer_cls(
        task=config["task"],
        model=config["model"],
        outputs=config.get("outputs", {}),
        dataset=config["dataset"],
        optimizer=config["optim"],
        loss_functions=config.get("loss_functions", []),
        evaluation_metrics=config.get("evaluation_metrics", {}),
        identifier=config["identifier"],
        # ... 其他参数
    )
    
    # 5. 从注册表获取任务类
    task_name = config["mode"]  # train/predict/validate/run-relaxations
    task_cls = registry.get_task_class(task_name)
    task = task_cls(config)
    
    # 6. 创建上下文对象
    @dataclass
    class Context:
        config: dict
        trainer: trainer_cls
        task: task_cls
    
    ctx = Context(config=config, trainer=trainer, task=task)
    
    try:
        yield ctx
    finally:
        # 清理资源
        pass
```

### 关键步骤说明

| 步骤 | 函数/操作 | 作用 |
|------|----------|------|
| 1 | `setup_imports()` | 动态导入所有模块，触发 `@registry.register_*` 装饰器 |
| 2 | `distutils.setup()` | 初始化分布式训练环境 |
| 3 | `registry.get_trainer_class()` | 获取训练器类（如 MLIPTrainer） |
| 4 | `trainer_cls(...)` | 实例化训练器，构建模型、数据加载器等 |
| 5 | `registry.get_task_class()` | 获取任务类（如 TrainTask） |

---

## 5. Task 类详解

**文件位置**: `src/fairchem/core/tasks/task.py`

### 5.1 BaseTask 基类

```python
class BaseTask:
    def __init__(self, config) -> None:
        self.config = config

    def setup(self, trainer) -> None:
        """设置任务，加载检查点"""
        self.trainer = trainer

        # 确定检查点路径
        format = get_checkpoint_format(self.config)
        if format == "pt":
            self.chkpt_path = os.path.join(
                self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
            )
        else:
            self.chkpt_path = self.trainer.config["cmd"]["checkpoint_dir"]

        # 加载用户指定的检查点
        if self.config["checkpoint"] is not None:
            logging.info(f"Loading checkpoint: {self.config['checkpoint']}")
            self.trainer.load_checkpoint(
                checkpoint_path=self.config["checkpoint"],
                inference_only=self.config.get("task", {}).get("load_weights_only", False),
            )
        # 或加载之前的检查点（恢复训练）
        elif os.path.isfile(self.chkpt_path):
            logging.info(f"Resuming from checkpoint: {self.chkpt_path}")
            self.trainer.load_checkpoint(checkpoint_path=self.chkpt_path)

    def run(self):
        raise NotImplementedError
```

### 5.2 TrainTask 训练任务

```python
@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError) -> None:
        """处理分布式训练错误"""
        if "find_unused_parameters" in str(e):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(f"Parameter {name} has no gradient.")

    def run(self) -> None:
        """执行训练"""
        try:
            self.trainer.train(
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False)
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e
```

### 5.3 PredictTask 预测任务

```python
@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self) -> None:
        """执行预测"""
        assert self.trainer.test_loader is not None, "Test dataset required"
        assert self.config["checkpoint"], "Checkpoint required for prediction"
        
        self.trainer.predict(
            self.trainer.test_loader,
            results_file="predictions",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )
```

### 5.4 ValidateTask 验证任务

```python
@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self) -> None:
        """执行验证"""
        assert self.trainer.val_loader is not None, "Val dataset required"
        assert self.config["checkpoint"], "Checkpoint required"
        
        self.trainer.validate(
            split="val",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )
```

---

## 6. 完整调用链流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           main()                                     │
│  - setup_logging()                                                   │
│  - 解析命令行参数                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       build_config()                                 │
│  - load_config() → 递归加载 YAML + includes                          │
│  - 合并命令行参数                                                      │
│  - 返回 config 字典                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    runner_wrapper(config)                            │
│  或 elastic_launch(..., runner_wrapper)(config)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Runner()(config)                               │
│  with new_trainer_context(config) as ctx:                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐
    │   new_trainer_context()   │   │                           │
    │   - setup_imports()       │   │   registry.get_trainer()  │
    │   - distutils.setup()     │   │   → MLIPTrainer(...)      │
    │   - registry.get_task()   │   │   → TrainTask(config)     │
    └───────────────────────────┘   └───────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    task.setup(trainer)                               │
│  - 设置检查点路径                                                      │
│  - 加载已有检查点（如果有）                                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       task.run()                                     │
│  - TrainTask: trainer.train()                                        │
│  - PredictTask: trainer.predict()                                    │
│  - ValidateTask: trainer.validate()                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   trainer.train() / predict()                        │
│  - 数据加载                                                           │
│  - 模型前向传播                                                        │
│  - 损失计算 & 反向传播                                                  │
│  - 验证 & 保存检查点                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Registry 注册机制

fairchem 使用注册模式管理所有可扩展组件：

```python
# 注册训练器
@registry.register_trainer("mlip_trainer")
class MLIPTrainer(OCPTrainer):
    ...

@registry.register_trainer("base")
class OCPTrainer(BaseTrainer):
    ...

# 注册任务
@registry.register_task("train")
class TrainTask(BaseTask):
    ...

@registry.register_task("predict")
class PredictTask(BaseTask):
    ...

# 注册模型
@registry.register_model("esen_backbone")
class eSEN_Backbone(nn.Module):
    ...

# 获取注册的类
trainer_cls = registry.get_trainer_class("mlip_trainer")  # → MLIPTrainer
task_cls = registry.get_task_class("train")               # → TrainTask
model_cls = registry.get_model_class("esen_backbone")     # → eSEN_Backbone
```

---

## 8. 命令行使用示例

```bash
# 训练
python -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode train \
    --num-gpus 4

# 从检查点继续训练
python -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode train \
    --checkpoint checkpoints/checkpoint.pt

# 预测
python -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode predict \
    --checkpoint checkpoints/best_model.pt

# 验证
python -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode validate \
    --checkpoint checkpoints/best_model.pt

# 提交到 Slurm 集群
python -m fairchem.core._cli \
    --config-yml configs/eSEN_31M_omt24_direct_force.yml \
    --mode train \
    --submit \
    --slurm-partition gpu \
    --num-nodes 2 \
    --num-gpus 8
```

---

## 9. 关键文件索引

| 文件 | 主要内容 |
|------|----------|
| `src/fairchem/core/_cli.py` | main()、Runner 类 |
| `src/fairchem/core/common/utils.py` | build_config()、load_config()、new_trainer_context() |
| `src/fairchem/core/common/registry.py` | Registry 注册表 |
| `src/fairchem/core/common/distutils.py` | 分布式训练工具 |
| `src/fairchem/core/tasks/task.py` | BaseTask、TrainTask、PredictTask 等 |
| `src/fairchem/core/common/flags.py` | 命令行参数定义 |