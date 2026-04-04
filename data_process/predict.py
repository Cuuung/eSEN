from ase.io import read
from fairchem.core import OCPCalculator

local_ckpt_path = "/data/home/liuxiangjia/fairchem/share/model/esen/esen_30m_mptrj.pt"

calc = OCPCalculator(checkpoint_path=local_ckpt_path, cpu=False, seed=42)

molecules = read("/data/home/liuxiangjia/fairchem/share/dataset/periodicSystem/pet-mad/mad-test.xyz", index=':')
print(f"total number:{len(molecules)}")
for atoms in molecules[:100]:
    atoms.calc = calc
    print(atoms.get_potential_energy())
