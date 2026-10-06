"""Build the reference structure set (no labels) -> data/structures.xyz.

Every labeller (VASP via tools/vasp, or the EMT stand-in) works from this one
file, so the demo's train/holdout split is fixed here and nowhere else.

  train    4-atom cells: 5 distinct orderings x (7-point volume scan at the
           Vegard lattice constant + 12 rattled snapshots at 3 volumes)
           8-atom cells: pure Cu8, Au8 and 8 mixed orderings, each ideal + 3 rattled
  holdout  the other 17 mixed 8-atom orderings, each ideal + 2 rattled

The 8-atom lists are written out explicitly. The first 25 orderings and the
8 training picks are the ones the DFT labels were computed for; the last two
orderings (00010101, 01010111) were found later, after a bug in the symmetry
reduction that had merged each of them with a look-alike, and are appended so
the first 180 structures keep their names and geometries.
"""

import sys
from pathlib import Path

import numpy as np
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import DATA, build, canonical, distinct_orderings, vegard_a  # noqa: E402

TRAIN8 = ["00000000", "11111111", "00000001", "00000011", "00010001",
          "00010010", "00010111", "00011101", "00110011", "01111111"]
HOLDOUT8 = ["00000101", "00000111", "00001111", "00010011", "00010110",
            "00011110", "00011111", "00110111", "00111111", "01010110",
            "01011010", "01011011", "01011111", "01110111", "01111011",
            "00010101", "01010111"]

rng = np.random.default_rng(0)
# The training picks above came from this draw in the first version of the
# script; it is kept so the rattle seeds that follow reproduce the labelled
# structures exactly.
rng.choice(23, size=8, replace=False)


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
    for s in orderings:
        bits = tuple(int(c) for c in s)
        frames.append(tag(build(bits, repeat=(2, 1, 1)), split, config_type, "ideal", in_training))
        for _ in range(n_rattle):
            frames.append(tag(rattled(bits, (2, 1, 1), vegard_a(bits), 0.08), split, config_type, "rattled", in_training))
    return frames


def main():
    keys = {canonical(tuple(int(c) for c in s), (2, 1, 1)) for s in TRAIN8 + HOLDOUT8}
    all_keys = {canonical(b, (2, 1, 1)) for b in distinct_orderings((2, 1, 1))}
    assert keys == all_keys and len(keys) == len(TRAIN8) + len(HOLDOUT8), "8-atom lists must cover every distinct ordering exactly once"

    frames = four_atom() + eight_atom(TRAIN8, 3, "train", "train8", True) + eight_atom(HOLDOUT8, 2, "holdout", "holdout8", False)
    for i, fr in enumerate(frames):
        fr.info["name"] = f"{i:03d}_{fr.info['config_type']}_{fr.info['ordering']}_{fr.info['kind']}"

    DATA.mkdir(exist_ok=True)
    write(DATA / "structures.xyz", frames, format="extxyz")
    n_train = sum(fr.info["split"] == "train" for fr in frames)
    print(f"wrote {len(frames)} structures -> data/structures.xyz ({n_train} train, {len(frames) - n_train} holdout)")
    print(f"8-atom orderings: {len(TRAIN8)} in training, {len(HOLDOUT8)} held out, {len(all_keys)} distinct in total")


if __name__ == "__main__":
    main()
