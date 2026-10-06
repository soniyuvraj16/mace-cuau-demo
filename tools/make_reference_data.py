"""STAND-IN labeller: label data/structures.xyz with ASE's EMT potential.

Produces data/train.xyz and data/holdout.xyz in the same format the VASP
collector (tools/vasp/collect.py) produces, so the demo can be timed before the
DFT runs finish. Not DFT. Refuses to overwrite files that hold real labels
unless --force is given.
"""

import argparse
import sys
from pathlib import Path

from ase.calculators.emt import EMT
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import DATA  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="overwrite even if data/*.xyz hold non-EMT labels")
    args = p.parse_args()

    for split in ("train", "holdout"):
        path = DATA / f"{split}.xyz"
        if path.exists() and not args.force:
            sources = {at.info.get("source") for at in read(path, ":")}
            if sources - {"EMT-standin"}:
                sys.exit(f"{path} holds {sorted(sources)} labels; not overwriting with EMT (use --force)")

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
