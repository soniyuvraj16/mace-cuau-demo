"""Collect finished VASP runs into train.xyz and holdout.xyz.

Reads <calcs>/<name>/OUTCAR for every structure in structures.xyz, takes
energy(sigma->0) and the final forces, maps forces back to the structure's
site order, and writes REF_energy / REF_forces. Unconverged or missing runs
are reported and skipped.

Works both inside the repo (tools/vasp/collect.py, data under ../../data) and
inside the uploaded bundle (vasp/tools/collect.py, data under ../data).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
for p in (HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(p))
from common import bits_of  # noqa: E402

E_SIGMA0 = re.compile(r"energy\(sigma->0\)\s*=\s*([-\d.Ee+]+)")


def first_existing(*candidates):
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


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
    p.add_argument("--structures", default=None)
    p.add_argument("--calcs", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    structures = Path(args.structures) if args.structures else first_existing(
        HERE.parents[1] / "data" / "structures.xyz", HERE.parent / "data" / "structures.xyz")
    calcs = Path(args.calcs) if args.calcs else first_existing(
        HERE.parents[1] / "vasp" / "calcs", HERE.parent / "calcs")
    out_dir = Path(args.out_dir) if args.out_dir else structures.parent

    frames = read(structures, ":")
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

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "holdout"):
        sub = [at for at in done if at.info["split"] == split]
        write(out_dir / f"{split}.xyz", sub, format="extxyz")
        print(f"wrote {len(sub)} frames -> {out_dir / f'{split}.xyz'}")
    if problems:
        print(f"\n{len(problems)} of {len(frames)} runs skipped:")
        print("\n".join("  " + s for s in problems))
    pure_done = {sum(bits_of(at)) for at in done if len(at) == 8 and at.info["kind"] == "ideal"}
    missing_pure = [el for el, n_au in (("Cu8", 0), ("Au8", 8)) if n_au not in pure_done]
    if missing_pure:
        print(f"\nWARNING: pure 8-atom references {missing_pure} are missing; formation energies need them")


if __name__ == "__main__":
    main()
