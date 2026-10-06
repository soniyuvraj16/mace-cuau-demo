"""Shared helpers: Cu-Au FCC ordering builders, fingerprints, formation energies."""

import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

A_CU = 3.63  # PBE lattice constants
A_AU = 4.16
FINGERPRINT_A = 4.0
MODEL_NAME = "cuau_ft"


def device():
    forced = os.environ.get("MACE_DEVICE")
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def vegard_a(bits):
    x = sum(bits) / len(bits)
    return (1 - x) * A_CU + x * A_AU


def build(bits, a=None, repeat=(1, 1, 1)):
    """Conventional 4-site FCC cell repeated, decorated by bits (0=Cu, 1=Au)."""
    n_sites = 4 * int(np.prod(repeat))
    assert len(bits) == n_sites, f"need {n_sites} bits, got {len(bits)}"
    if a is None:
        a = vegard_a(bits)
    atoms = bulk("Cu", "fcc", a=a, cubic=True).repeat(repeat)
    atoms.set_chemical_symbols(["Au" if b else "Cu" for b in bits])
    atoms.info["ordering"] = "".join(str(b) for b in bits)
    return atoms


def fingerprint(bits, repeat):
    """Symmetry-invariant key: pair-type/distance histogram on a fixed lattice."""
    atoms = build(bits, a=FINGERPRINT_A, repeat=repeat)
    i, j, d = neighbor_list("ijd", atoms, cutoff=1.3 * FINGERPRINT_A)
    z = atoms.numbers
    pairs = Counter(
        (min(z[a], z[b]), max(z[a], z[b]), round(float(dist), 2))
        for a, b, dist in zip(i, j, d)
    )
    return str(sorted(pairs.items()))


def distinct_orderings(repeat):
    """All symmetry-distinct 0/1 decorations of the repeated 4-site cell."""
    n_sites = 4 * int(np.prod(repeat))
    seen = {}
    for bits in itertools.product((0, 1), repeat=n_sites):
        fp = fingerprint(bits, repeat)
        if fp not in seen:
            seen[fp] = bits
    return list(seen.values())


def bits_of(atoms):
    return tuple(int(s == "Au") for s in atoms.get_chemical_symbols())


def formation_energy_per_atom(energy, n_cu, n_au, e_cu, e_au):
    """energy is total; e_cu/e_au are per-atom pure-element references."""
    n = n_cu + n_au
    return (energy - n_cu * e_cu - n_au * e_au) / n


def lower_hull(xs, ys):
    """Lower convex hull of (x, y) points; returns indices of hull vertices."""
    pts = sorted(range(len(xs)), key=lambda k: (xs[k], ys[k]))
    hull = []
    for k in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = (xs[hull[-2]], ys[hull[-2]]), (xs[hull[-1]], ys[hull[-1]])
            x3, y3 = xs[k], ys[k]
            if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(k)
    return hull


def reference_ideals_8atom():
    """bits -> (REF_energy, in_training) for every ideal 8-atom cell on disk."""
    from ase.io import read

    out = {}
    for name in ("train.xyz", "holdout.xyz"):
        for fr in read(DATA / name, ":"):
            if len(fr) == 8 and fr.info.get("kind") == "ideal":
                out[bits_of(fr)] = (fr.info["REF_energy"], bool(fr.info.get("in_training", False)))
    return out


def pure_per_atom(energy_of_bits, n_sites):
    """Per-atom pure Cu and Au energies from a bits -> total-energy function."""
    return energy_of_bits(tuple([0] * n_sites)) / n_sites, energy_of_bits(tuple([1] * n_sites)) / n_sites


def mace_calculator(model_path=None, dev=None, dtype="float32"):
    from mace.calculators import MACECalculator, mace_mp

    dev = dev or device()
    if model_path is None:
        return mace_mp(model="small", device=dev, default_dtype=dtype)
    return MACECalculator(model_paths=str(model_path), device=dev, default_dtype=dtype)


def banner(text):
    print("\n" + "=" * 72, file=sys.stderr)
    print(text, file=sys.stderr)
    print("=" * 72, file=sys.stderr)
