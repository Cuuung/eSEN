"""
sAlex 跨数据集验证脚本（联合校正版）

评估流程：
1. 用 MPtrj normalizer + MPtrj refs 得到原始预测 E_pred_raw
2. 在 5000 个测试样本的「预测值 vs 真实值」之间拟合 per-element 校正量 δ
3. E_pred_corrected = E_pred_raw + δ·c，计算校正后 MAE
   → 排除两个数据集 DFT 设置不同带来的系统偏差，得到纯模型误差
"""

import random
import logging

import numpy as np
import torch
from functools import partial
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 注册所有需要的模型
from fairchem.core.models.base import HydraModel
from fairchem.core.common.registry import registry
from fairchem.core.models.esen import esen, esen_dens
from fairchem.core.models.esen.nn import rank2

# ── 路径 ──────────────────────────────────────────────────────────────────────
CHECKPOINT = (
    "/mnt/afs/share/checkpoint/eSEN/checkpoints/"
    "2026-04-14-15-51-28-eSEN_3.2M_directforce/checkpoint_epoch100_step275000.pt"
)
SALEX_VAL_DIR = "/mnt/afs/share/dataset/periodicSystem/sAlex/val/"

N_EVAL      = 5_000
BATCH_SZ    = 32
NUM_WORKERS = 0
SEED        = 42
MAX_ELEMENTS = 119   # 原子序数 1-118

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ── 1. 加载 sAlex 数据集，随机抽 5000 ─────────────────────────────────────────
from fairchem.core.datasets import AseDBDataset, data_list_collater

logging.info("Loading sAlex val dataset...")
full_dataset = AseDBDataset({
    "src": SALEX_VAL_DIR,
    "a2g_args": {"r_energy": True, "r_forces": True},
})
logging.info(f"Total sAlex val structures: {len(full_dataset)}")

all_indices  = list(range(len(full_dataset)))
random.shuffle(all_indices)
eval_indices = all_indices[:N_EVAL]
eval_subset  = Subset(full_dataset, eval_indices)

# ── 2. 加载模型（MPtrj 训练的权重） ──────────────────────────────────────────
from fairchem.core.modules.normalization.normalizer import create_normalizer

logging.info("Loading checkpoint...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

model_cfg    = ckpt["config"]["model"]
model_kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
model        = HydraModel(**model_kwargs)

raw_sd = ckpt["state_dict"]
if all(k.startswith("module.") for k in raw_sd):
    raw_sd = {k[len("module."):]: v for k, v in raw_sd.items()}
model.load_state_dict(raw_sd)
model = model.to(device).eval()
logging.info(f"Model loaded, {sum(p.numel() for p in model.parameters()):,} params")

# MPtrj normalizer 和 element refs（训练时的，直接用）
mptrj_normalizer = create_normalizer(state_dict=ckpt["normalizers"]["energy"]).to(device)
from fairchem.core.modules.normalization.element_references import create_element_references
mptrj_refs = create_element_references(state_dict=ckpt["elementrefs"]["energy"]).to(device)
logging.info(
    f"MPtrj normalizer: mean={mptrj_normalizer.mean.item():.4f}, "
    f"rmsd={mptrj_normalizer.rmsd.item():.4f}"
)

# ── 3. 推理：得到原始预测 E_pred_raw ─────────────────────────────────────────
eval_loader = DataLoader(
    eval_subset,
    batch_size=BATCH_SZ,
    shuffle=False,
    collate_fn=partial(data_list_collater, otf_graph=True),
    num_workers=NUM_WORKERS,
)

all_pred_raw  = []   # E_pred_raw [eV]
all_true      = []   # E_true     [eV]
all_natoms    = []   # N_atoms
# 组成矩阵：每行是一个结构的原子计数向量，用于拟合 per-element 校正
composition_rows = []

logging.info(f"Running inference on {N_EVAL} structures...")
with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Inference"):
        batch = batch.to(device)
        out   = model(batch)

        # MPtrj 空间：denorm → 加 MPtrj refs → 原始预测绝对能量
        e_norm = out["energy"].squeeze(-1)                         # [B]
        e_res  = mptrj_normalizer.denorm(e_norm)                   # 残差空间
        e_raw  = mptrj_refs(e_res, batch, reshaped=False)          # 加 MPtrj refs

        all_pred_raw.extend(e_raw.cpu().tolist())
        all_true.extend(batch.energy.cpu().tolist())
        all_natoms.extend(batch.natoms.cpu().tolist())

        # 逐结构统计组成向量
        for data in batch.to_data_list():
            counts = torch.bincount(
                data.atomic_numbers.int(), minlength=MAX_ELEMENTS
            ).float()
            composition_rows.append(counts.cpu())

# ── 4. 拟合 per-element 校正量 δ ─────────────────────────────────────────────
# 求解：min Σ|E_pred_raw_i + δ·c_i - E_true_i|²
# 即：  min Σ|(E_pred_raw_i - E_true_i) - (-δ·c_i)|²
# 等价：min ||C·δ - residual||²，其中 residual = E_true - E_pred_raw

logging.info("Fitting per-element correction δ on test predictions vs true values...")

pred_raw = np.array(all_pred_raw)   # [N]
true_val = np.array(all_true)       # [N]
natoms   = np.array(all_natoms)     # [N]

residual = true_val - pred_raw      # E_true - E_pred_raw，即需要被补偿的系统偏移

# 组成矩阵，只保留出现过的元素列
C = torch.stack(composition_rows).numpy()          # [N, 119]
nonzero_mask = C.sum(axis=0) != 0                  # 只拟合出现过的元素
C_reduced    = C[:, nonzero_mask]                  # [N, n_elem]

# 最小二乘求解
delta_reduced, _, _, _ = np.linalg.lstsq(C_reduced, residual, rcond=None)

delta = np.zeros(MAX_ELEMENTS)
delta[nonzero_mask] = delta_reduced

logging.info("Per-element corrections δ (eV/atom)，典型元素：")
elem_names = {1:"H", 6:"C", 8:"O", 26:"Fe", 28:"Ni", 29:"Cu"}
for z, sym in elem_names.items():
    logging.info(f"  {sym:2s} (Z={z:3d}): {delta[z]:.4f} eV")

# ── 5. 计算校正后 MAE ─────────────────────────────────────────────────────────
# E_pred_corrected = E_pred_raw + δ·c
delta_per_struct = C @ delta                          # [N]，每个结构的校正量
pred_corrected   = pred_raw + delta_per_struct        # [N]

err_total   = np.abs(pred_corrected - true_val)       # [N]  eV
err_peratom = err_total / natoms                      # [N]  eV/atom

# 无校正的 MAE（用 MPtrj refs，作为对比基线）
err_raw_peratom = np.abs(pred_raw - true_val) / natoms

print("\n" + "="*55)
print(f"  sAlex 跨数据集验证结果（联合校正，n={N_EVAL}）")
print("="*55)
print(f"  [无校正]   Energy MAE (per-atom): {err_raw_peratom.mean()*1000:.1f} meV/atom")
print(f"  [联合校正] Energy MAE (total)   : {err_total.mean():.4f} eV")
print(f"  [联合校正] Energy MAE (per-atom): {err_peratom.mean()*1000:.1f} meV/atom")
print(f"  [联合校正] Energy Median        : {np.median(err_peratom)*1000:.1f} meV/atom")
print("="*55)
