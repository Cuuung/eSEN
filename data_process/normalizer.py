import glob
import pickle

import lmdb
import numpy as np

DIRS = {
    "train": "/data/home/liuxiangjia/fairchem/share/dataset/periodicSystem/MPtrj/lmdb/train",
    "val":   "/data/home/liuxiangjia/fairchem/share/dataset/periodicSystem/MPtrj/lmdb/val",
}

for split, dirpath in DIRS.items():
    print(f"\n=== {split} ===")

    natoms_list = []

    energy_sum    = energy_sq_sum    = 0.0
    force_sum     = force_sq_sum     = 0.0
    stress_sum    = stress_sq_sum    = 0.0

    n_energy = n_force = n_stress_elem = 0

    for lmdb_path in sorted(glob.glob(f"{dirpath}/data.*.lmdb")):
        print(f"  reading {lmdb_path} ...", flush=True)
        db = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False)
        with db.begin() as txn:
            length = pickle.loads(txn.get(b"length"))
            for i in range(length):
                data = pickle.loads(txn.get(f"{i}".encode()))

                # natoms
                natoms_list.append(int(data.natoms))

                # energy（per-system）
                e = float(data.energy)
                energy_sum    += e
                energy_sq_sum += e * e
                n_energy      += 1

                # forces（所有分量打平）
                f = np.asarray(data.forces).flatten()
                force_sum     += f.sum()
                force_sq_sum  += (f ** 2).sum()
                n_force       += len(f)

                # stress（3×3 所有 9 分量打平）
                s = np.asarray(data.stress).flatten()  # shape (9,)
                stress_sum    += s.sum()
                stress_sq_sum += (s ** 2).sum()
                n_stress_elem += len(s)

        db.close()

    # ── metadata.npz ──
    natoms_arr = np.array(natoms_list, dtype=np.int32)
    np.savez(f"{dirpath}/metadata.npz", natoms=natoms_arr)
    print(f"  Saved metadata.npz: {len(natoms_arr)} samples")

    # ── 统计量 ──
    e_mean  = energy_sum / n_energy
    e_stdev = np.sqrt(max(energy_sq_sum / n_energy - e_mean ** 2, 0.0))

    f_mean  = force_sum / n_force
    f_stdev = np.sqrt(max(force_sq_sum / n_force - f_mean ** 2, 0.0))

    s_mean  = stress_sum / n_stress_elem
    s_stdev = np.sqrt(max(stress_sq_sum / n_stress_elem - s_mean ** 2, 0.0))

    print(f"\n  [energy]  mean={e_mean:.6f}  stdev={e_stdev:.6f}")
    print(f"  [forces]  mean={f_mean:.6f}  stdev={f_stdev:.6f}")
    print(f"  [stress]  mean={s_mean:.6f}  stdev={s_stdev:.6f}")

    print(f"""
  ── 填入 config normalizer ({split}) ──
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
