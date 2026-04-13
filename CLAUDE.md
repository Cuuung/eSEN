# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fairchem is FAIR Chemistry's centralized repository for materials science ML. The core task is training models to predict energies, forces, and relaxed structures for atomic/molecular systems (the Open Catalyst Project). It supports tasks like S2EF (Structure to Energy and Forces), IS2RE, and IS2RS.

## Installation

```bash
pip install -e packages/fairchem-core
```

## Common Commands

### Training

**Single GPU (legacy CLI):**
```bash
python main.py --mode train --config-yml configs/my_config.yml
```

**Multi-GPU (torchrun):**
```bash
torchrun --nproc_per_node=N main_torchrun.py --config-yml configs/my_config.yml --mode train
```

**SLURM submission:**
```bash
python main.py --mode train --config-yml configs/my_config.yml --submit --num-nodes N --num-gpus N
```

**New Hydra-based CLI:**
```bash
fairchemv2 --config-name my_config
```

### Testing

```bash
pytest tests/                     # all tests
pytest tests/core/                # core module tests
pytest tests/core/test_foo.py     # single test file
pytest tests/core/test_foo.py::test_bar  # single test
```

### Linting & Formatting

```bash
pre-commit run --all-files        # run all pre-commit hooks
ruff check src/                   # lint
ruff format src/                  # format
```

## Architecture

### Registry Pattern

All major components are registered in a central registry (`src/fairchem/core/common/registry.py`) and instantiated from YAML config keys. This is the core design pattern:

```python
@registry.register_model("my_model")
class MyModel(nn.Module): ...

@registry.register_trainer("forces")
class OCPTrainer(BaseTrainer): ...
```

Config keys (`model.name`, `trainer`, `task.dataset`) map to registered class names.

### Execution Flow

```
main.py / main_torchrun.py
└── _cli.py / _cli_hydra.py
    └── Runner()(config)
        ├── instantiate Trainer from registry
        ├── instantiate Task from registry
        ├── create Datasets from config
        └── task.setup(trainer) → task.run()
```

### Key Abstractions

**Tasks** (`src/fairchem/core/tasks/task.py`):
- `BaseTask.setup(trainer)` loads checkpoints and initializes trainer
- `TrainTask`, `ValidateTask`, `PredictTask`, `RelaxationTask` implement `run()`

**Trainers** (`src/fairchem/core/trainers/`):
- `BaseTrainer` handles device management, checkpointing, distributed training, logging, EMA
- `OCPTrainer` (in `ocp_trainer.py`) adds S2EF/IS2RE/relaxation support

**Models** (`src/fairchem/core/models/`):
- Models inherit from `nn.Module` and use `GraphModelMixin` for graph/PBC utilities
- Available architectures: SchNet, DimeNet++, PaiNN, ESCN, EquiformerV2 (v1/v2), GemNet, GemNet-OC, UMA, eSEN, SCN
- Prediction heads implement `HeadInterface`

**Datasets** (`src/fairchem/core/datasets/`):
- `LmdbDataset` — primary dataset format for large-scale OCP data
- `AseReadDataset`, `AseDBDataset` — ASE format files
- `create_dataset(config, split)` — factory function used throughout

### YAML Config Structure

```yaml
trainer: forces           # registry key for trainer
task:
  dataset: trajectory_lmdb
model:
  name: equiformer_v2    # registry key for model
  # model hyperparameters...
dataset:
  train: {src: /path, ...}
  val: {src: /path, ...}
optim:
  batch_size: 8
  lr_initial: 1e-4
  num_workers: 8
loss_functions: [...]
evaluation_metrics: [...]
logger: wandb
```

Configs live in `configs/`. See `src/fairchem/core/TRAIN.md` for detailed config documentation.

### Distributed Training

`src/fairchem/core/common/distutils.py` abstracts single-GPU, multi-GPU, and multi-node setups. Trainers call `distutils` helpers for rank, world size, and collective operations. Submitit integration (`_cli.py`) handles SLURM preemption with automatic checkpoint/restore.

### eSEN Model

The eSEN (Equivariant Scalable Equivariant Network) model lives in `src/fairchem/core/models/esen/`. It has its own trainer at `src/fairchem/core/models/esen/trainers/trainer.py` and uses configs under `configs/eSEN_*.yml`. It supports both `auto_grad` (force via autograd) and `direct_force` prediction modes.

### Inference via ASE

```python
from fairchem.core import OCPCalculator
calc = OCPCalculator(model_name="EquiformerV2-31M-S2EF-OC20-All+MD")
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

Relaxation utilities are in `src/fairchem/core/common/relaxation/`.
