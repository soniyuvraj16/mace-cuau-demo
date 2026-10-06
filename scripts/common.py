"""Shared helpers: Cu-Au FCC ordering builders, fingerprints, formation energies."""

import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
from functools import lru_cache
from pathlib import Path

import numpy as np
from ase.build import bulk

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

A_CU = 3.63  # PBE lattice constants
A_AU = 4.16
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


def _cubic_point_ops():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            ops.append(m)
    return ops


@lru_cache(maxsize=None)
def _equivalence_maps(repeat):
    """For every symmetry op of the empty FCC crystal (48 point ops x FCC
    translations mod the supercell): the site maps needed to (a) test whether
    a decoration is periodic under the rotated supercell lattice and (b) pull
    the rotated decoration back onto the supercell sites. Positions in units
    of the lattice constant."""
    lat = bulk("Cu", "fcc", a=1.0, cubic=True).repeat(repeat)
    pos = lat.positions
    box = np.array(repeat, dtype=float)

    def site(p):
        frac = np.round((p / box) % 1.0, 6) % 1.0
        return index[tuple(frac)]

    index = {tuple(np.round((p / box) % 1.0, 6) % 1.0): i for i, p in enumerate(pos)}
    basis = [np.array(v, dtype=float) for v in np.diag(repeat)]
    maps = set()
    for g in _cubic_point_ops():
        ginv = g.T
        periodicity = tuple(tuple(site(p + ginv @ l) for p in pos) for l in basis)
        for t in pos:
            pull = tuple(site(ginv @ (p - t)) for p in pos)
            maps.add((periodicity, pull))
    return tuple(maps)


def canonical(bits, repeat):
    """Canonical label of a decoration: the lexicographically smallest decoration
    of the same supercell that describes the same crystal (any orientation)."""
    bits = tuple(bits)
    best = None
    for periodicity, pull in _equivalence_maps(tuple(repeat)):
        if any(bits[m[i]] != bits[i] for m in periodicity for i in range(len(bits))):
            continue
        img = tuple(bits[pull[j]] for j in range(len(bits)))
        if best is None or img < best:
            best = img
    return "".join(map(str, best))


def distinct_orderings(repeat):
    """One representative per symmetry-distinct 0/1 decoration of the repeated 4-site cell."""
    n_sites = 4 * int(np.prod(repeat))
    seen = {}
    for bits in itertools.product((0, 1), repeat=n_sites):
        seen.setdefault(canonical(bits, repeat), bits)
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
