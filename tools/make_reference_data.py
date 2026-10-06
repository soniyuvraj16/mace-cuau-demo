"""STAND-IN labeller: label data/structures.xyz with ASE's EMT potential.

Produces data/train.xyz and data/holdout.xyz in the same format the VASP
collector (tools/vasp/collect.py) produces, so the demo can be timed before the
DFT runs finish. Not DFT.
"""

import sys
from pathlib import Path

from ase.calculators.emt import EMT
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import DATA  # noqa: E402


def main():
    frames = read(DATA / "structures.xyz", ":")
    for at in frames:
        at.calc = EMT()
        at.info["REF_energy"] = float(at.get_potential_energy())
        at.arrays["REF_forces"] = at.get_forces().copy()
        at.info["source"] = "EMT-standin"
        at.calc = None
    for split in ("train", "holdout"):
        sub = [at for at in frames if at.info["split"] == split]
        write(DATA / f"{split}.xyz", sub, format="extxyz")
        print(f"wrote {len(sub)} frames -> data/{split}.xyz (EMT stand-in)")


if __name__ == "__main__":
    main()
