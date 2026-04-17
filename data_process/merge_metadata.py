"""
Merge per-file metadata_db_N.npz into a single metadata.npz per subdirectory.

The merge order matches AseLmdbDataset's file loading order:
  sorted(glob(f"{src}/**/*.aselmdb", recursive=True))

Usage:
    python data_process/merge_metadata.py [--split train|val|all] [--dry-run]
"""

import argparse
import glob
import logging
import re
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

BASE = "/mnt/afs/share/dataset/periodicSystem/omat24"

SPLITS = {
    "train": f"{BASE}/training",
    "val":   f"{BASE}/validation",
}


def merge_subdir(subdir: str, dry_run: bool):
    subdir = Path(subdir)
    aselmdb_files = sorted(subdir.glob("*.aselmdb"))
    if not aselmdb_files:
        return

    natoms_parts = []
    missing = []
    for db_file in aselmdb_files:
        # db_N.aselmdb → metadata_db_N.npz
        n = re.search(r"db_(\d+)\.aselmdb$", db_file.name)
        if n is None:
            logging.warning(f"Unexpected filename: {db_file}, skipping")
            continue
        meta_file = subdir / f"metadata_db_{n.group(1)}.npz"
        if not meta_file.is_file():
            missing.append(str(meta_file))
            continue
        natoms_parts.append(np.load(meta_file, allow_pickle=True)["natoms"])

    if missing:
        logging.warning(f"{subdir}: missing metadata for {len(missing)} files: {missing[:3]}...")

    if not natoms_parts:
        logging.warning(f"{subdir}: no metadata found, skipping")
        return

    merged = np.concatenate(natoms_parts).astype(np.int32)
    out_path = subdir / "metadata.npz"

    if dry_run:
        logging.info(f"[DRY RUN] Would write {out_path}  ({len(merged)} samples)")
        return

    np.savez(out_path, natoms=merged)
    logging.info(f"Wrote {out_path}  ({len(merged)} samples)")


def merge_split(root_dir: str, dry_run: bool):
    root = Path(root_dir)
    subdirs = sorted({f.parent for f in root.glob("**/*.aselmdb")})
    logging.info(f"{root_dir}: {len(subdirs)} subdirectories")
    for subdir in subdirs:
        merge_subdir(str(subdir), dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["train", "val", "all"])
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
    for split, root_dir in splits.items():
        logging.info(f"=== {split} : {root_dir} ===")
        merge_split(root_dir, dry_run=args.dry_run)
