"""
Compute energy/forces/stress mean & stdev from omat24 aselmdb files,
and generate metadata.npz (natoms array) for each subdirectory.

Uses multiprocessing to process subdirectories in parallel.

Usage:
    python data_process/normalizer_aselmdb.py [--workers N] [--split train|val|all]
"""

import argparse
import glob
import logging
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import ase.db
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

BASE = "/data/home/liuxiangjia/fairchem/share/dataset/periodicSystem/omat24"

SPLITS = {
    "train": f"{BASE}/training",
    "val":   f"{BASE}/validation",
}


def process_subdir(args):
    """Process a single subdirectory: write metadata.npz and return partial stats."""
    subdir, files = args
    pid = os.getpid()

    natoms_list = []
    energy_sum = energy_sq_sum = 0.0
    force_sum  = force_sq_sum  = 0.0
    stress_sum = stress_sq_sum = 0.0
    n_energy = n_force = n_stress = 0

    for fpath in sorted(files):
        try:
            db = ase.db.connect(fpath, readonly=True)
        except Exception as e:
            logging.warning(f"[{pid}] Cannot open {fpath}: {e}")
            continue

        try:
            for row in db.select():
                natoms_list.append(int(row.natoms))

                atoms = row.toatoms(add_additional_information=True)
                res   = atoms.calc.results

                e = float(res["energy"])
                energy_sum    += e
                energy_sq_sum += e * e
                n_energy      += 1

                f = np.asarray(res["forces"]).flatten()
                force_sum     += f.sum()
                force_sq_sum  += (f ** 2).sum()
                n_force       += len(f)

                s = np.asarray(res["stress"]).flatten()
                stress_sum    += s.sum()
                stress_sq_sum += (s ** 2).sum()
                n_stress      += len(s)

        except Exception as e:
            logging.warning(f"[{pid}] Error reading {fpath}: {e}")
        finally:
            db.close()

    # Write per-subdir metadata.npz
    if natoms_list:
        natoms_arr = np.array(natoms_list, dtype=np.int32)
        out_path = f"{subdir}/metadata.npz"
        np.savez(out_path, natoms=natoms_arr)
        logging.info(f"[{pid}] Saved {out_path}  ({len(natoms_arr)} samples)")

    return {
        "n_samples": len(natoms_list),
        "energy_sum": energy_sum, "energy_sq_sum": energy_sq_sum, "n_energy": n_energy,
        "force_sum":  force_sum,  "force_sq_sum":  force_sq_sum,  "n_force":  n_force,
        "stress_sum": stress_sum, "stress_sq_sum": stress_sq_sum, "n_stress": n_stress,
    }


def process_split(split: str, root_dir: str, num_workers: int):
    logging.info(f"=== {split} : {root_dir} ===")

    all_files = sorted(glob.glob(f"{root_dir}/**/*.aselmdb", recursive=True))
    if not all_files:
        logging.warning(f"No .aselmdb files found under {root_dir}")
        return

    logging.info(f"  Found {len(all_files)} .aselmdb files")

    subdir_files: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        subdir_files[str(Path(f).parent)].append(f)

    tasks = sorted(subdir_files.items())
    logging.info(f"  {len(tasks)} subdirectories, using {num_workers} workers")

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_subdir, tasks)

    # Aggregate stats across all subdirs
    agg = {k: 0.0 for k in results[0]}
    for r in results:
        for k, v in r.items():
            agg[k] += v

    n_energy = int(agg["n_energy"])
    n_force  = int(agg["n_force"])
    n_stress = int(agg["n_stress"])

    if n_energy == 0:
        logging.warning("No data found.")
        return

    e_mean  = agg["energy_sum"] / n_energy
    e_stdev = np.sqrt(max(agg["energy_sq_sum"] / n_energy - e_mean ** 2, 0.0))

    f_mean  = agg["force_sum"] / n_force
    f_stdev = np.sqrt(max(agg["force_sq_sum"] / n_force - f_mean ** 2, 0.0))

    s_mean  = agg["stress_sum"] / n_stress
    s_stdev = np.sqrt(max(agg["stress_sq_sum"] / n_stress - s_mean ** 2, 0.0))

    print(f"\n[{split}] {n_energy} structures")
    print(f"  energy  mean={e_mean:.6f}  stdev={e_stdev:.6f}")
    print(f"  forces  mean={f_mean:.6f}  stdev={f_stdev:.6f}")
    print(f"  stress  mean={s_mean:.6f}  stdev={s_stdev:.6f}")
    print(f"""
  ── config normalizer ({split}) ──
    normalizer:
      energy:
        mean: {e_mean:.6f}
        stdev: {e_stdev:.6f}
      forces:
        mean: {f_mean:.6f}
        stdev: {f_stdev:.6f}
      stress:
        mean: {s_mean:.6f}
        stdev: {s_stdev:.6f}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
    for split, root_dir in splits.items():
        process_split(split, root_dir, num_workers=args.workers)
