"""Build the reference structure set (no labels) -> data/structures.xyz.

Every labeller (VASP via tools/vasp, or the EMT stand-in) works from this one
file, so the demo's train/holdout split is fixed here and nowhere else.

  train    4-atom cells: 5 distinct orderings x (7-point volume scan at the
           Vegard lattice constant + 12 rattled snapshots at 3 volumes)
           8-atom cells: pure Cu8, Au8 and 8 random mixed orderings, each
           ideal + 3 rattled
  holdout  the other 15 mixed 8-atom orderings, each ideal + 2 rattled
"""

import sys
from pathlib import Path

import numpy as np
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import DATA, build, distinct_orderings, vegard_a  # noqa: E402

rng = np.random.default_rng(0)
N_TRAIN_MIXED_8ATOM = 8


def tag(atoms, split, config_type, kind, in_training):
    atoms.info.update(split=split, config_type=config_type, kind=kind, in_training=in_training)
    return atoms


def rattled(bits, repeat, a, sigma):
    at = build(bits, a=a, repeat=repeat)
    at.rattle(stdev=sigma, seed=int(rng.integers(1 << 31)))
    return at


def four_atom():
    frames = []
    for bits in distinct_orderings((1, 1, 1)):
        a0 = vegard_a(bits)
        for s in np.linspace(0.95, 1.05, 7):
            frames.append(tag(build(bits, a=a0 * s), "train", "eos4", "ideal", True))
        for scale in (0.97, 1.0, 1.03):
            for sigma in (0.05, 0.10):
                for _ in range(2):
                    frames.append(tag(rattled(bits, (1, 1, 1), a0 * scale, sigma), "train", "rattle4", "rattled", True))
    return frames


def eight_atom(orderings, n_rattle, split, config_type, in_training):
    frames = []
    for bits in orderings:
        frames.append(tag(build(bits, repeat=(2, 1, 1)), split, config_type, "ideal", in_training))
        for _ in range(n_rattle):
            frames.append(tag(rattled(bits, (2, 1, 1), vegard_a(bits), 0.08), split, config_type, "rattled", in_training))
    return frames


def main():
    all8 = distinct_orderings((2, 1, 1))
    pure = [b for b in all8 if sum(b) in (0, 8)]
    mixed = [b for b in all8 if 0 < sum(b) < 8]
    pick = set(rng.choice(len(mixed), size=N_TRAIN_MIXED_8ATOM, replace=False).tolist())
    train8 = pure + [b for i, b in enumerate(mixed) if i in pick]
    hold8 = [b for i, b in enumerate(mixed) if i not in pick]

    frames = four_atom() + eight_atom(train8, 3, "train", "train8", True) + eight_atom(hold8, 2, "holdout", "holdout8", False)
    for i, fr in enumerate(frames):
        fr.info["name"] = f"{i:03d}_{fr.info['config_type']}_{fr.info['ordering']}_{fr.info['kind']}"

    DATA.mkdir(exist_ok=True)
    write(DATA / "structures.xyz", frames, format="extxyz")
    n_train = sum(fr.info["split"] == "train" for fr in frames)
    print(f"wrote {len(frames)} structures -> data/structures.xyz ({n_train} train, {len(frames) - n_train} holdout)")
    print(f"8-atom orderings: {len(train8)} in training, {len(hold8)} held out, {len(all8)} distinct in total")


if __name__ == "__main__":
    main()
