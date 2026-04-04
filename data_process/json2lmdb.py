"""
Fast JSON -> lmdb converter for MPtrj dataset.
Loads entire JSON into memory with orjson, then parallel-writes lmdb.

Install: pip install orjson
"""

import orjson
import ijson
import numpy as np
import random
import lmdb
import pickle
import os
import types
import torch
from multiprocessing import Pool
from tqdm import tqdm
from ase.data import atomic_numbers as ase_atomic_numbers

JSON_PATH  = "/data/home/liuxiangjia/fairchem/share/dataset/periodicSystem/MPtrj/MPtrj_2022.9_full.json"
TRAIN_DIR  = "/data/home/liuxiangjia/fairchem/share/dataset/periodicSystem/MPtrj/lmdb/train"
VAL_DIR    = "/data/home/liuxiangjia/fairchem/share/dataset/periodicSystem/MPtrj/lmdb/val"

NUM_WORKERS = 16   # 根据 CPU 核数调整

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ── 全局变量，fork 后子进程通过 COW 共享，不额外复制内存 ──
ALL_DATA = None


def init_worker(data):
    global ALL_DATA
    ALL_DATA = data


def process_chunk(args):
    """每个子进程处理一批 mp_id，分别写 train / val lmdb。"""
    worker_id, mp_ids, train_keys = args
    global ALL_DATA

    train_path = os.path.join(TRAIN_DIR, f"data.{worker_id}.lmdb")
    val_path   = os.path.join(VAL_DIR,   f"data.{worker_id}.lmdb")

    train_env = lmdb.open(train_path, map_size=1099511627776, subdir=False, meminit=False, map_async=True)
    val_env   = lmdb.open(val_path,   map_size=1099511627776, subdir=False, meminit=False, map_async=True)

    train_txn = train_env.begin(write=True)
    val_txn   = val_env.begin(write=True)
    train_idx = val_idx = 0
    COMMIT = 10000

    for mp_id in mp_ids:
        frames = ALL_DATA[mp_id]
        is_train = mp_id in train_keys

        for frame_id, frame in frames.items():
            struct   = frame["structure"]
            symbols  = [s["species"][0]["element"] for s in struct["sites"]]
            pos      = np.array([[float(x) for x in s["xyz"]] for s in struct["sites"]], dtype=np.float32)
            cell     = np.array([[float(x) for x in row] for row in struct["lattice"]["matrix"]], dtype=np.float32)

            n = len(symbols)
            data = types.SimpleNamespace(
                pos            = torch.tensor(pos),
                atomic_numbers = torch.tensor([ase_atomic_numbers[s] for s in symbols], dtype=torch.long),
                cell           = torch.tensor(cell).unsqueeze(0),   # (1,3,3)
                pbc            = torch.tensor([True, True, True]),
                natoms         = torch.tensor(n),
                fixed          = torch.zeros(n, dtype=torch.long),  # MPtrj 无固定原子
                energy         = torch.tensor(float(frame["uncorrected_total_energy"]), dtype=torch.float64),
                forces         = torch.tensor(np.array(frame["force"],  dtype=np.float32)),
                stress         = torch.tensor(np.array(frame["stress"], dtype=np.float32)),
                sid            = f"{mp_id}_{frame_id}",
            )
            buf = pickle.dumps(data, protocol=-1)

            if is_train:
                train_txn.put(str(train_idx).encode(), buf)
                train_idx += 1
                if train_idx % COMMIT == 0:
                    train_txn.commit(); train_txn = train_env.begin(write=True)
            else:
                val_txn.put(str(val_idx).encode(), buf)
                val_idx += 1
                if val_idx % COMMIT == 0:
                    val_txn.commit(); val_txn = val_env.begin(write=True)

    train_txn.commit()
    val_txn.commit()

    with train_env.begin(write=True) as txn:
        txn.put(b"length", pickle.dumps(train_idx, protocol=-1))
    with val_env.begin(write=True) as txn:
        txn.put(b"length", pickle.dumps(val_idx, protocol=-1))

    train_env.close()
    val_env.close()
    return worker_id, train_idx, val_idx


TEST_MODE  = False   # 测试完改成 False
TEST_KEYS  = 20     # 测试时只读前 N 个 key


if __name__ == "__main__":
    # ── Step 1: 加载 JSON ──
    if TEST_MODE:
        print(f"[TEST] Loading first {TEST_KEYS} keys with ijson ...")
        ALL_DATA = {}
        with open(JSON_PATH, "rb") as f:
            for mp_id, frames in ijson.kvitems(f, ""):
                ALL_DATA[mp_id] = frames
                if len(ALL_DATA) >= TEST_KEYS:
                    break
    else:
        print("Loading JSON into memory with orjson ...")
        with open(JSON_PATH, "rb") as f:
            ALL_DATA = orjson.loads(f.read())
    print(f"Loaded {len(ALL_DATA)} materials")

    # ── Step 2: 划分 train / val ──
    all_keys = list(ALL_DATA.keys())
    random.seed(42)
    random.shuffle(all_keys)
    split = int(len(all_keys) * 0.9)
    train_keys = set(all_keys[:split])
    print(f"train: {len(train_keys)}, val: {len(all_keys) - len(train_keys)}")

    # ── Step 3: 分 chunk，多进程写 lmdb ──
    chunks = [all_keys[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    args   = [(i, chunk, train_keys) for i, chunk in enumerate(chunks)]

    print(f"Writing with {NUM_WORKERS} workers ...")
    with Pool(NUM_WORKERS, initializer=init_worker, initargs=(ALL_DATA,)) as pool:
        results = list(tqdm(pool.imap(process_chunk, args), total=NUM_WORKERS))

    total_train = sum(r[1] for r in results)
    total_val   = sum(r[2] for r in results)
    print(f"Done. train: {total_train} frames, val: {total_val} frames")
    print(f"Train lmdb files: {TRAIN_DIR}/data.*.lmdb")
    print(f"Val   lmdb files: {VAL_DIR}/data.*.lmdb")
