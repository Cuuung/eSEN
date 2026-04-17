import glob, pickle
import lmdb
import numpy as np

lmdb_dir = "/mnt/afs/share/dataset/periodicSystem/MPtrj/lmdb/train"

MAX_NATOMS   = 300   # natoms 超过此值
MIN_CELL_LEN = 2.0   # 晶格向量长度小于此值（Å）
MAX_FORCE    = 50.0  # 单个力分量绝对值超过此值（eV/Å），可能是异常数据
MAX_STRESS   = 1000.0  # 应力分量绝对值超过此值（eV/Å³）

suspicious = []

for path in sorted(glob.glob(f"{lmdb_dir}/data.*.lmdb")):
    print(f"scanning {path} ...", flush=True)
    db = lmdb.open(path, readonly=True, subdir=False, lock=False)
    with db.begin() as txn:
        length = pickle.loads(txn.get(b"length"))
        for i in range(length):
            data = pickle.loads(txn.get(f"{i}".encode()))
            reasons = []

            # 1. natoms 过大
            natoms = int(data.natoms)
            if natoms > MAX_NATOMS:
                reasons.append(f"large_natoms={natoms}")

            # 2. 晶格向量过小（会导致 radius_graph_pbc 搜索大量镜像）
            cell = np.asarray(data.cell).reshape(3, 3)
            lengths = np.linalg.norm(cell, axis=1)
            min_len = lengths.min()
            if min_len < MIN_CELL_LEN:
                reasons.append(f"small_cell={min_len:.3f}Å")

            # 3. 晶胞体积过小（接近奇异矩阵，会导致数值不稳定）
            vol = abs(np.linalg.det(cell))
            if vol < 1.0:
                reasons.append(f"small_vol={vol:.4f}Å³")

            # 4. 力的异常值（数据标注错误或单位错误）
            forces = np.asarray(data.forces)
            max_force = np.abs(forces).max()
            if max_force > MAX_FORCE:
                reasons.append(f"large_force={max_force:.1f}")

            # 5. 应力异常值
            stress = np.asarray(data.stress)
            max_stress = np.abs(stress).max()
            if max_stress > MAX_STRESS:
                reasons.append(f"large_stress={max_stress:.1f}")

            # 6. NaN / Inf 检查
            for name, arr in [("energy", [float(data.energy)]),
                               ("forces", forces.flatten()),
                               ("stress", stress.flatten())]:
                arr = np.array(arr)
                if not np.isfinite(arr).all():
                    reasons.append(f"nan_inf_in_{name}")

            if reasons:
                suspicious.append({
                    "sid":     getattr(data, "sid", f"{path}:{i}"),
                    "natoms":  natoms,
                    "min_len": min_len,
                    "reasons": reasons,
                })
    db.close()

print(f"\n共发现 {len(suspicious)} 个可疑结构")
for s in sorted(suspicious, key=lambda x: x["min_len"])[:50]:
    print(f"  sid={s['sid']}  natoms={s['natoms']}  min_cell={s['min_len']:.3f}Å  {s['reasons']}")
