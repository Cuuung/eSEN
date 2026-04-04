"""
多节点入口：torchrun 负责进程启动，fairchem 读取环境变量初始化分布式。
"""
import os
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import build_config, setup_logging
from fairchem.core._cli import Runner

setup_logging()

parser = flags.get_parser()
args, override_args = parser.parse_known_args()

config = build_config(args, override_args)

# 让 fairchem 读 torchrun 注入的环境变量
config["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
config["distributed_backend"] = "nccl"
config["submit"] = False
config["summit"] = False

Runner()(config)
