"""
Energy MAE with element re-referencing for cross-dataset evaluation.

Usage:
    python scripts/eval_energy_reref.py \
        --config-yml configs/eval_eSEN_3.2M_sAlex.yml \
        --checkpoint /data/share/checkpoint/eSEN/checkpoints/.../checkpoint.pt \
        --out results_3.2M_sAlex.npz

Steps:
    1. Load model + dataset via fairchem trainer
    2. Run inference, collect (E_pred, E_DFT, natoms, composition)
    3. Fit per-element linear references separately on E_pred and E_DFT
    4. Compute MAE on residuals (per atom)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Make sure fairchem is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairchem.core.common.utils import build_config, new_trainer_context, setup_logging
from fairchem.core.common.flags import flags

MAX_ATOMIC_NUM = 119  # H=1 ... Og=118, index 0 unused


def collect_predictions(trainer):
    """Run inference over val_loader, return arrays of predictions and targets."""
    trainer.model.eval()
    if trainer.ema is not None:
        trainer.ema.store()
        trainer.ema.copy_to()

    e_pred_list = []
    e_dft_list  = []
    natoms_list = []
    comp_list   = []  # composition vectors shape (N, MAX_ATOMIC_NUM)

    device = trainer.device

    with torch.no_grad():
        for batch in tqdm(trainer.val_loader, desc="Inference"):
            batch = batch.to(device)
            out   = trainer._forward(batch)

            # denormalise energy prediction
            e_pred = trainer._denorm_preds("energy", out["energy"], batch)
            e_pred = e_pred.squeeze(-1).cpu().numpy()  # (batch_size,)

            e_dft  = batch.energy.cpu().numpy()         # (batch_size,)
            natoms = batch.natoms.cpu().numpy()          # (batch_size,)

            # build per-structure composition vectors
            atomic_numbers = batch.atomic_numbers.cpu().numpy()  # (total_atoms,)
            split_idx      = np.cumsum(natoms)[:-1]
            atom_groups    = np.split(atomic_numbers, split_idx)

            for nums in atom_groups:
                comp = np.zeros(MAX_ATOMIC_NUM, dtype=np.float64)
                for z in nums:
                    if 0 < z < MAX_ATOMIC_NUM:
                        comp[z] += 1
                comp_list.append(comp)

            e_pred_list.append(e_pred)
            e_dft_list.append(e_dft)
            natoms_list.append(natoms)

    if trainer.ema is not None:
        trainer.ema.restore()

    e_pred = np.concatenate(e_pred_list)   # (N,)
    e_dft  = np.concatenate(e_dft_list)    # (N,)
    natoms = np.concatenate(natoms_list)   # (N,)
    comps  = np.stack(comp_list)           # (N, MAX_ATOMIC_NUM)

    return e_pred, e_dft, natoms, comps


def fit_element_refs(energies: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """
    Fit per-element reference energies by least squares:
        energies ≈ comps @ refs
    Returns refs array of shape (MAX_ATOMIC_NUM,).
    Only elements present in the dataset are fitted; others are zero.
    """
    # Only keep columns (elements) that appear at least once
    present = comps.sum(axis=0) > 0
    A = comps[:, present]
    refs_partial, _, _, _ = np.linalg.lstsq(A, energies, rcond=None)
    refs = np.zeros(MAX_ATOMIC_NUM)
    refs[present] = refs_partial
    return refs


def compute_reref_mae(e_pred, e_dft, natoms, comps):
    """
    Fit a per-element correction to the prediction error delta = e_dft - e_pred,
    then report the residual per-atom MAE.

    delta_i ≈ comps_i @ correction_refs  (least squares)
    e_pred_corrected = e_pred + comps @ correction_refs
    MAE = mean(|e_pred_corrected - e_dft| / natoms)
    """
    delta = e_dft - e_pred                           # (N,) total energy error per structure
    correction_refs = fit_element_refs(delta, comps)  # per-element correction (eV/atom)

    e_pred_corrected = e_pred + comps @ correction_refs
    per_atom_errors  = np.abs(e_pred_corrected - e_dft) / natoms
    mae = per_atom_errors.mean()
    return mae, per_atom_errors, correction_refs


def main():
    setup_logging()

    # Extract --out before handing args to fairchem's flag parser
    out_path = None
    argv = sys.argv[1:]
    if "--out" in argv:
        idx = argv.index("--out")
        out_path = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]
    sys.argv = [sys.argv[0]] + argv

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()

    # Point to val split
    args.mode = "validate"

    config = build_config(args, override_args)
    config["is_debug"] = False  # required for predict

    with new_trainer_context(config=config) as ctx:
        trainer = ctx.trainer

        # Load checkpoint (normalizers + model weights)
        assert config.get("checkpoint"), "Pass --checkpoint path/to/checkpoint.pt"
        trainer.load_checkpoint(config["checkpoint"])

        logging.info("Collecting predictions...")
        e_pred, e_dft, natoms, comps = collect_predictions(trainer)

    n = len(e_pred)
    logging.info(f"Collected {n} samples")

    # Raw MAE (no referencing) for comparison
    raw_mae = (np.abs(e_pred - e_dft) / natoms).mean()
    logging.info(f"Raw energy MAE (no ref):        {raw_mae*1000:.4f} meV/atom")

    # Re-referenced MAE
    mae, per_atom_errors, correction_refs = compute_reref_mae(e_pred, e_dft, natoms, comps)
    logging.info(f"Re-referenced energy MAE:       {mae*1000:.4f} meV/atom")
    logging.info(f"  (min={per_atom_errors.min()*1000:.2f}  "
                 f"max={per_atom_errors.max()*1000:.2f}  "
                 f"median={np.median(per_atom_errors)*1000:.2f} meV/atom)")

    # Show largest per-element corrections (eV/atom)
    import ase.data
    nonzero = np.where(np.abs(correction_refs) > 0.01)[0]
    corrections_sorted = sorted(nonzero, key=lambda z: abs(correction_refs[z]), reverse=True)
    logging.info("Top per-element corrections (eV/atom):")
    for z in corrections_sorted[:10]:
        logging.info(f"  {ase.data.chemical_symbols[z]:2s} (Z={z:3d}): {correction_refs[z]:+.3f} eV/atom")

    if out_path:
        np.savez(out_path, e_pred=e_pred, e_dft=e_dft, natoms=natoms, comps=comps)
        logging.info(f"Saved arrays to {out_path}")


if __name__ == "__main__":
    main()
