"""Collect finished VASP runs into data/train.xyz and data/holdout.xyz.

Reads vasp/calcs/<name>/OUTCAR for every structure in data/structures.xyz,
takes energy(sigma->0) and the final forces, maps forces back to the
structure's site order, and writes REF_energy / REF_forces. Unconverged or
missing runs are reported and skipped.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import DATA, bits_of  # noqa: E402

E_SIGMA0 = re.compile(r"energy\(sigma->0\)\s*=\s*([-\d.Ee+]+)")


def parse(d):
    outcar = d / "OUTCAR"
    if not outcar.exists():
        return None, "missing OUTCAR"
    text = outcar.read_text(errors="ignore")
    if "aborting loop because EDIFF is reached" not in text:
        return None, "SCF not converged"
    energies = E_SIGMA0.findall(text)
    if not energies:
        return None, "no energy(sigma->0) in OUTCAR"
    forces_poscar = read(outcar, format="vasp-out").get_forces()
    perm = np.array(json.loads((d / "meta.json").read_text())["perm"])
    forces = np.empty_like(forces_poscar)
    forces[perm] = forces_poscar
    return (float(energies[-1]), forces), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calcs", default=str(ROOT / "vasp" / "calcs"))
    args = p.parse_args()
    calcs = Path(args.calcs)

    frames = read(DATA / "structures.xyz", ":")
    done, problems = [], []
    for at in frames:
        result, why = parse(calcs / at.info["name"])
        if result is None:
            problems.append(f"{at.info['name']}: {why}")
            continue
        energy, forces = result
        at.info["REF_energy"] = energy
        at.arrays["REF_forces"] = forces
        at.info["source"] = "VASP-PBE"
        done.append(at)

    for split in ("train", "holdout"):
        sub = [at for at in done if at.info["split"] == split]
        write(DATA / f"{split}.xyz", sub, format="extxyz")
        print(f"wrote {len(sub)} frames -> data/{split}.xyz")
    if problems:
        print(f"\n{len(problems)} of {len(frames)} runs skipped:")
        print("\n".join("  " + s for s in problems))
    pure_done = {sum(bits_of(at)) for at in done if len(at) == 8 and at.info["kind"] == "ideal"}
    missing_pure = [el for el, n_au in (("Cu8", 0), ("Au8", 8)) if n_au not in pure_done]
    if missing_pure:
        print(f"\nWARNING: pure 8-atom references {missing_pure} are missing; formation energies need them")


if __name__ == "__main__":
    main()
