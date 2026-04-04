"""
Convert XYZ/extxyz files to LMDB format for fairchem training.
Usage:
    python xyz2lmdb.py --input share/dataset/periodicSystem/pet-mad/mad-val.xyz --output data/pet-med-lmdb/train/mad-val.lmdb
"""

import argparse
import lmdb
import pickle
import ase.io
from tqdm import tqdm
from fairchem.core.preprocessing import AtomsToGraphs


def convert(input_path: str, output_path: str, radius: float, max_neigh: int) -> None:
    a2g = AtomsToGraphs(
        max_neigh=max_neigh,
        radius=radius,
        r_energy=True,
        r_forces=True,
        r_fixed=True,
        r_distances=False,
        r_edges=False,
    )

    print(f"Reading {input_path} ...")
    frames = ase.io.read(input_path, ":")
    print(f"Total structures: {len(frames)}")

    db = lmdb.open(
        output_path,
        map_size=1 * (1024 ** 3),  # 1 GB
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for idx, atoms in enumerate(tqdm(frames, desc="Writing LMDB")):
        data = a2g.convert(atoms)
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    # Write total length
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(frames), protocol=-1))
    txn.commit()

    db.sync()
    db.close()
    print(f"Done! Saved to {output_path}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert XYZ to LMDB for fairchem")
    parser.add_argument("--input", required=True, help="Path to input .xyz file")
    parser.add_argument("--output", required=True, help="Path to output .lmdb file")
    parser.add_argument("--radius", type=float, default=6.0, help="Cutoff radius (default: 6.0)")
    parser.add_argument("--max-neigh", type=int, default=50, help="Max neighbors (default: 50)")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    convert(args.input, args.output, args.radius, args.max_neigh)
