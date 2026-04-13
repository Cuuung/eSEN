"""
phonon_eval.py
==============
从 sAlex .aselmdb 数据集抽取平衡态结构（frame_idx=0），评估：

  1. Energy MAE (eV/atom)：ML 预测 vs DFT ground truth，在所有抽样结构上计算
  2. 声子热力学量（phonopy 有限位移法）：
       - Max frequency (THz)
       - Free energy (kJ/mol)  @ 给定温度
       - Heat capacity Cv (J/K/mol)  @ 给定温度
     默认输出 mean ± std（预测分布）。
     若提供 --phonon-ref 文件（JSON），则额外计算 MAE。

说明：
  - 平衡态结构通过 sid 末尾为 `_0` 来识别（frame_idx=0，DFT 已收敛）
  - 声子计算不需要再驰豫（frame_idx=0 的 max_force < 0.005 eV/Å）
  - 5000 个结构的 Energy MAE 约需几分钟；每个结构的声子计算约需 1-5 分钟

依赖安装：
    pip install phonopy

用法示例：
    python scripts/phonon_eval.py \
        --checkpoint checkpoints/2026-03-31-10-12-16/best_checkpoint.pt \
        --config configs/eSEN_31M_omat24_direct_force.yml \
        --data-dir share/dataset/periodicSystem/sAlex/train \
        --n-samples 5000 \
        --phonon-subset 50 \
        --temperature 300 \
        --supercell 2 2 2 \
        --output results/phonon_eval_sAlex
"""

import argparse
import json
import logging
import random
import time
import zlib
from pathlib import Path

import lmdb
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 读取 .aselmdb（LMDB + zlib JSON 格式）
# ─────────────────────────────────────────────

def _open_env(path: Path) -> tuple[lmdb.Environment, int]:
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    n = env.stat()["entries"]
    with env.begin() as txn:
        if txn.get(b"nextid") is not None:
            n -= 1
    return env, n


def _read_raw(env: lmdb.Environment, key: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(f"{key}".encode("ascii"))
    return json.loads(zlib.decompress(raw))


def _to_atoms(data: dict) -> Atoms:
    atoms = Atoms(
        numbers=data["numbers"],
        positions=data["positions"],
        pbc=data["pbc"],
        cell=data["cell"],
    )
    calc_kw = {}
    if "energy" in data:
        calc_kw["energy"] = float(data["energy"])
    if "forces" in data:
        calc_kw["forces"] = np.array(data["forces"])
    if "stress" in data:
        calc_kw["stress"] = np.array(data["stress"])
    if calc_kw:
        atoms.calc = SinglePointCalculator(atoms, **calc_kw)
    atoms.info["sid"] = data.get("data", {}).get("sid", data.get("unique_id", ""))
    return atoms


def _is_equilibrium(sid: str) -> bool:
    """sid 末尾为 _0 表示 frame_idx=0（DFT 收敛的平衡态）。"""
    return sid.endswith("_0")


# ─────────────────────────────────────────────
# 建立平衡态索引（二次扫描，仅读 sid）
# ─────────────────────────────────────────────

def build_equilibrium_index(data_dir: Path) -> list[tuple[Path, int]]:
    """
    扫描 data_dir 下所有 .aselmdb，返回所有平衡态（frame_idx=0）条目的
    (文件路径, key) 列表。
    """
    db_files = sorted(data_dir.rglob("*.aselmdb"))
    assert db_files, f"未在 {data_dir} 找到 .aselmdb 文件"
    log.info(f"找到 {len(db_files)} 个 .aselmdb 文件，扫描平衡态条目 ...")

    index = []
    for db_path in db_files:
        env, n = _open_env(db_path)
        with env.begin() as txn:
            for key in range(1, n + 1):
                raw = txn.get(f"{key}".encode("ascii"))
                if raw is None:
                    continue
                data = json.loads(zlib.decompress(raw))
                sid = data.get("data", {}).get("sid", data.get("unique_id", ""))
                if _is_equilibrium(sid):
                    index.append((db_path, key))
        env.close()

    log.info(f"平衡态条目总数：{len(index)}")
    return index


# ─────────────────────────────────────────────
# 1. Energy MAE
# ─────────────────────────────────────────────

def compute_energy_mae(ml_calc, entries: list, n_samples: int) -> float:
    """ML 预测能量 vs DFT 能量，计算 MAE（eV/atom）。"""
    log.info(f"计算 Energy MAE：{n_samples} 个平衡结构 ...")
    sampled = random.sample(entries, min(n_samples, len(entries)))

    errors = []
    for i, (db_path, key) in enumerate(sampled):
        data = _read_raw(lmdb.open(str(db_path), subdir=False, readonly=True, lock=False), key)
        atoms = _to_atoms(data)
        gt_energy = atoms.calc.results["energy"]    # DFT 参考值（eV）

        atoms.calc = ml_calc
        pred_energy = atoms.get_potential_energy()  # ML 预测（eV）

        errors.append(abs(pred_energy - gt_energy) / len(atoms))
        if (i + 1) % 100 == 0:
            log.info(f"  进度：{i+1}/{len(sampled)}，当前 MAE={np.mean(errors):.4f} eV/atom")

    mae = float(np.mean(errors))
    log.info(f"Energy MAE：{mae:.4f} eV/atom（n={len(errors)}）")
    return mae


# ─────────────────────────────────────────────
# 2. 声子计算（phonopy 有限位移法）
# ─────────────────────────────────────────────

def run_phonon(atoms: Atoms, ml_calc, supercell_matrix, temperature: float,
               displacement: float = 0.01) -> dict | None:
    """
    对单个平衡结构做声子计算。
    atoms 须已在平衡态（frame_idx=0 满足条件，无需再驰豫）。
    """
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        raise ImportError("请先安装 phonopy：pip install phonopy")

    ph_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell().array,
        scaled_positions=atoms.get_scaled_positions(),
    )
    phonon = Phonopy(ph_atoms, supercell_matrix=supercell_matrix)
    phonon.generate_displacements(distance=displacement)
    supercells_disp = phonon.supercells_with_displacements

    # 有限位移：用 ML 预测每个位移超胞的力
    forces_list = []
    for sc in supercells_disp:
        sc_atoms = Atoms(symbols=sc.symbols, positions=sc.positions, cell=sc.cell, pbc=True)
        sc_atoms.calc = ml_calc
        forces_list.append(sc_atoms.get_forces())

    phonon.forces = forces_list
    phonon.produce_force_constants()

    # 声子色散 → 最大频率
    phonon.run_mesh([20, 20, 20], with_eigenvectors=False)
    max_freq = float(np.max(phonon.get_mesh_dict()["frequencies"]))   # THz

    # 热力学性质（谐振近似）
    phonon.run_thermal_properties(t_min=temperature, t_max=temperature, t_step=1)
    tp = phonon.get_thermal_properties_dict()
    free_energy   = float(tp["free_energy"][0])     # kJ/mol
    heat_capacity = float(tp["heat_capacity"][0])   # J/K/mol

    return {
        "max_freq_THz":          max_freq,
        "free_energy_kJ_mol":    free_energy,
        "heat_capacity_J_K_mol": heat_capacity,
    }


def compute_phonon_properties(
    ml_calc, entries: list, n_phonon: int,
    supercell_matrix, temperature: float,
    phonon_ref: dict | None = None,
) -> dict | None:
    """
    对随机抽取的 n_phonon 个结构做声子计算。

    phonon_ref: 可选，格式为 {sid: {"max_freq_THz": ..., "free_energy_kJ_mol": ...,
                                    "heat_capacity_J_K_mol": ...}}
                若提供，则额外计算 MAE。
    """
    log.info(f"声子计算：{n_phonon} 个结构，T={temperature} K ...")
    sampled = random.sample(entries, min(n_phonon, len(entries)))

    results = []
    for i, (db_path, key) in enumerate(sampled):
        data = _read_raw(lmdb.open(str(db_path), subdir=False, readonly=True, lock=False), key)
        atoms = _to_atoms(data)
        sid = atoms.info["sid"]
        log.info(f"  [{i+1}/{len(sampled)}] {sid}  ({len(atoms)} 原子)")

        try:
            atoms.calc = ml_calc
            props = run_phonon(atoms, ml_calc, supercell_matrix, temperature)
            props["sid"] = sid
            results.append(props)
            log.info(
                f"    max_freq={props['max_freq_THz']:.2f} THz  "
                f"F={props['free_energy_kJ_mol']:.3f} kJ/mol  "
                f"Cv={props['heat_capacity_J_K_mol']:.3f} J/K/mol"
            )
        except Exception as e:
            log.warning(f"    {sid} 失败：{e}")

    if not results:
        log.warning("所有声子计算均失败。")
        return None

    keys3 = ["max_freq_THz", "free_energy_kJ_mol", "heat_capacity_J_K_mol"]
    summary = {"n_success": len(results)}
    for k in keys3:
        vals = [r[k] for r in results]
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"]  = float(np.std(vals))

    # 若有 DFT 参考，计算 MAE
    if phonon_ref:
        for k in keys3:
            paired = [(r[k], phonon_ref[r["sid"]][k])
                      for r in results if r["sid"] in phonon_ref]
            if paired:
                preds, refs = zip(*paired)
                summary[f"mae_{k}"] = float(np.mean(np.abs(np.array(preds) - np.array(refs))))
                log.info(f"  MAE {k}: {summary[f'mae_{k}']:.4f}  (n={len(paired)})")

    summary["details"] = results
    return summary


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="sAlex 平衡态结构 Energy MAE + 声子性质评估")
    p.add_argument("--checkpoint",    required=True, help="训练好的 .pt checkpoint 路径")
    p.add_argument("--config",        required=True, help="YAML 配置文件路径")
    p.add_argument("--data-dir",      required=True, help="sAlex .aselmdb 数据目录")
    p.add_argument("--n-samples",     type=int, default=5000,
                   help="Energy MAE 抽样数量（同时作为声子计算的候选池）")
    p.add_argument("--phonon-subset", type=int, default=50,
                   help="从 n-samples 中抽多少个做声子计算（每个约1-5分钟）")
    p.add_argument("--temperature",   type=float, default=300.0, help="热力学温度 (K)")
    p.add_argument("--supercell",     type=int, nargs=3, default=[2, 2, 2],
                   help="超胞对角元（默认：2 2 2）")
    p.add_argument("--phonon-ref",    default=None,
                   help="（可选）DFT 声子参考 JSON 文件，用于计算声子 MAE\n"
                        "格式：{sid: {max_freq_THz, free_energy_kJ_mol, heat_capacity_J_K_mol}}")
    p.add_argument("--cpu",           action="store_true", help="使用 CPU（默认 GPU）")
    p.add_argument("--seed",          type=int, default=42, help="随机种子")
    p.add_argument("--output",        default=None, help="结果保存路径前缀（.npz）")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 加载 ML 模型
    from fairchem.core import OCPCalculator
    log.info(f"加载模型：{args.checkpoint}")
    ml_calc = OCPCalculator(
        config_yml=args.config,
        checkpoint_path=args.checkpoint,
        cpu=args.cpu,
    )

    # 建立平衡态索引
    index = build_equilibrium_index(Path(args.data_dir))
    assert index, "未找到任何平衡态条目（frame_idx=0）"

    # 从平衡态索引中抽取 n_samples 个
    n_samples = min(args.n_samples, len(index))
    sampled_pool = random.sample(index, n_samples)
    log.info(f"抽取 {n_samples} 个平衡结构用于评估")

    # 可选：加载 DFT 声子参考数据
    phonon_ref = None
    if args.phonon_ref:
        with open(args.phonon_ref) as f:
            phonon_ref = json.load(f)
        log.info(f"加载 DFT 声子参考：{len(phonon_ref)} 条记录")

    supercell_matrix = np.diag(args.supercell)
    t0 = time.time()

    # ── 1. Energy MAE ────────────────────────────
    energy_mae = compute_energy_mae(ml_calc, sampled_pool, n_samples)

    # ── 2. 声子性质 ──────────────────────────────
    phonon_summary = compute_phonon_properties(
        ml_calc=ml_calc,
        entries=sampled_pool,
        n_phonon=args.phonon_subset,
        supercell_matrix=supercell_matrix,
        temperature=args.temperature,
        phonon_ref=phonon_ref,
    )

    # ── 打印汇总 ─────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("评估结果汇总")
    print("=" * 65)
    print(f"数据集：       {args.data_dir}")
    print(f"平衡态结构：   {len(index)} 个（frame_idx=0）")
    print(f"抽样数量：     {n_samples}")
    print(f"Energy MAE:    {energy_mae:.4f} eV/atom")

    if phonon_summary:
        n = phonon_summary["n_success"]
        print(f"\n声子热力学量（T={args.temperature} K，成功 {n}/{args.phonon_subset} 个结构）:")
        for label, key in [
            ("Max frequency (THz)", "max_freq_THz"),
            ("Free energy (kJ/mol)", "free_energy_kJ_mol"),
            ("Heat capacity Cv (J/K/mol)", "heat_capacity_J_K_mol"),
        ]:
            mean = phonon_summary[f"mean_{key}"]
            std  = phonon_summary[f"std_{key}"]
            line = f"  {label:<28} {mean:.3f} ± {std:.3f}"
            if f"mae_{key}" in phonon_summary:
                line += f"   MAE={phonon_summary[f'mae_{key}']:.3f}"
            print(line)

        if not phonon_ref:
            print("\n  注：未提供 --phonon-ref，仅输出预测分布（非 MAE）。")
            print("      若需计算声子 MAE，请提供 DFT 声子参考 JSON 文件。")

    print(f"\n总耗时：{elapsed:.1f} s")
    print("=" * 65)

    # ── 保存结果 ─────────────────────────────────
    if args.output:
        save = {"energy_mae_eV_per_atom": energy_mae}
        if phonon_summary:
            save.update({k: v for k, v in phonon_summary.items() if k != "details"})
        out_path = args.output if args.output.endswith(".npz") else args.output + ".npz"
        np.savez(out_path, **save)
        log.info(f"结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
