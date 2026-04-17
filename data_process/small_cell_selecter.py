import glob, pickle, os
import lmdb
import numpy as np

SRC_DIR = "/mnt/afs/share/dataset/periodicSystem/MPtrj/lmdb/train"
DST_PATH = "/mnt/afs/share/dataset/periodicSystem/MPtrj/lmdb/small_cell_test/data.0.lmdb"
MIN_CELL_LEN = 2.0

os.makedirs(os.path.dirname(DST_PATH), exist_ok=True)

kept = []
for path in sorted(glob.glob(f"{SRC_DIR}/data.*.lmdb")):
    db = lmdb.open(path, readonly=True, subdir=False, lock=False)
    with db.begin() as txn:
        length = pickle.loads(txn.get(b"length"))
        for i in range(length):
            data = pickle.loads(txn.get(f"{i}".encode()))
            cell = np.asarray(data.cell).reshape(3, 3)
            min_len = np.linalg.norm(cell, axis=1).min()
            if min_len < MIN_CELL_LEN:
                kept.append(data)
    db.close()

dst_db = lmdb.open(DST_PATH, subdir=False, map_size=1024**3)
with dst_db.begin(write=True) as txn:
    for i, data in enumerate(kept):
        txn.put(f"{i}".encode(), pickle.dumps(data))
    txn.put(b"length", pickle.dumps(len(kept)))
dst_db.close()

# metadata.npz
natoms_list = [int(d.natoms) for d in kept]
np.savez(os.path.dirname(DST_PATH) + "/metadata.npz", natoms=np.array(natoms_list, dtype=np.int32))

print(f"共写入 {len(kept)} 个小晶胞结构 → {DST_PATH}")
