"""Author-side script: build the reference ("DFT") dataset.

STAND-IN MODE: labels come from ASE's EMT potential, not DFT. The output format
(extxyz with REF_energy / REF_forces) is exactly what the demo consumes, so a
real DFT run only has to replace the `label()` function below.

Outputs
  data/train.xyz    4-atom cells (volume scans + rattles) and a random 40% of
                    the 8-atom orderings (ideal + rattles), pure Cu8/Au8 always
  data/holdout.xyz  the other 60% of the 8-atom orderings, never trained on
"""

import sys
from pathlib import Path

import numpy as np
from ase.calculators.emt import EMT
from ase.eos import EquationOfState
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import DATA, build, distinct_orderings, vegard_a  # noqa: E402

rng = np.random.default_rng(0)
N_TRAIN_MIXED_8ATOM = 8


def label(atoms, config_type, kind):
    atoms = atoms.copy()
    atoms.calc = EMT()
    atoms.info["REF_energy"] = float(atoms.get_potential_energy())
    atoms.arrays["REF_forces"] = atoms.get_forces().copy()
    atoms.info["config_type"] = config_type
    atoms.info["kind"] = kind
    atoms.info["source"] = "EMT-standin"
    atoms.calc = None
    return atoms


def relaxed_lattice_constant(bits):
    a0 = vegard_a(bits)
    frames, vols, es = [], [], []
    for s in np.linspace(0.95, 1.05, 7):
        fr = label(build(bits, a=a0 * s), "eos4", "ideal")
        frames.append(fr)
        vols.append(fr.get_volume())
        es.append(fr.info["REF_energy"])
    v0, _, _ = EquationOfState(vols, es).fit()
    return (v0 / len(bits) * 4) ** (1 / 3), frames


def rattled(bits, repeat, a, sigma, config_type):
    at = build(bits, a=a, repeat=repeat)
    at.rattle(stdev=sigma, seed=int(rng.integers(1 << 31)))
    return label(at, config_type, "rattled")


def four_atom_frames():
    frames = []
    for bits in distinct_orderings((1, 1, 1)):
        a_eq, eos = relaxed_lattice_constant(bits)
        frames += eos
        for scale in (0.97, 1.0, 1.03):
            for sigma in (0.05, 0.10):
                for _ in range(2):
                    frames.append(rattled(bits, (1, 1, 1), a_eq * scale, sigma, "rattle4"))
        print(f"  4-atom ordering {''.join(map(str, bits))}: a_eq={a_eq:.3f} A")
    return frames


def eight_atom_frames(orderings, n_rattle, config_type, in_training):
    frames = []
    for bits in orderings:
        ideal = label(build(bits, repeat=(2, 1, 1)), config_type, "ideal")
        ideal.info["in_training"] = in_training
        frames.append(ideal)
        for _ in range(n_rattle):
            fr = rattled(bits, (2, 1, 1), vegard_a(bits), 0.08, config_type)
            fr.info["in_training"] = in_training
            frames.append(fr)
    return frames


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    all8 = distinct_orderings((2, 1, 1))
    pure = [b for b in all8 if sum(b) in (0, 8)]
    mixed = [b for b in all8 if 0 < sum(b) < 8]
    pick = rng.choice(len(mixed), size=N_TRAIN_MIXED_8ATOM, replace=False)
    train8 = pure + [mixed[i] for i in sorted(pick)]
    hold8 = [b for i, b in enumerate(mixed) if i not in set(pick)]

    print("Building train set...")
    train = four_atom_frames() + eight_atom_frames(train8, 3, "train8", True)
    write(DATA / "train.xyz", train, format="extxyz")
    print(f"  {len(train8)} of {len(all8)} distinct 8-atom orderings in training")
    print(f"  wrote {len(train)} frames -> data/train.xyz")

    print("Building holdout set...")
    holdout = eight_atom_frames(hold8, 2, "holdout8", False)
    write(DATA / "holdout.xyz", holdout, format="extxyz")
    print(f"  {len(hold8)} held-out 8-atom orderings, wrote {len(holdout)} frames -> data/holdout.xyz")
    print(f"\nreference single points in total: {len(train) + len(holdout)}")
