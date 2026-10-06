"""Step 3: evolutionary search for the most stable Cu-Au ordering, scored by the
fine-tuned MACE instead of DFT. The champion is then graded against reference
data for the same ordering.

Search space: all 2^8 decorations of an 8-site FCC supercell (25 distinct
orderings after symmetry). Small enough to also brute-force, which shows how
many evaluations the genetic algorithm actually needed.
"""

import json
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, MODELS, MODEL_NAME, RESULTS, banner, build, device, fingerprint, formation_energy_per_atom, lower_hull, mace_calculator, pure_per_atom, reference_ideals_8atom  # noqa: E402

REPEAT = (2, 1, 1)
N_SITES = 8
rng = np.random.default_rng(42)


class Oracle:
    """MACE energy oracle with a symmetry-aware cache."""

    def __init__(self, calc):
        self.calc = calc
        self.cache = {}
        self.calls = 0
        self.e_cu, self.e_au = pure_per_atom(self.total, N_SITES)

    def total(self, bits):
        fp = fingerprint(bits, REPEAT)
        if fp not in self.cache:
            at = build(bits, repeat=REPEAT)
            at.calc = self.calc
            self.cache[fp] = float(at.get_potential_energy())
            self.calls += 1
        return self.cache[fp]

    def formation(self, bits):
        n_au = sum(bits)
        return formation_energy_per_atom(self.total(bits), N_SITES - n_au, n_au, self.e_cu, self.e_au)


def genetic_search(oracle, name_of, pop_size=12, generations=8, p_mut=0.15):
    pop = [tuple(rng.integers(0, 2, N_SITES)) for _ in range(pop_size)]
    history = []
    for gen in range(generations):
        scored = sorted(pop, key=oracle.formation)
        best = scored[0]
        history.append({"generation": gen, "best_ordering": name_of(best),
                        "best_formation_meV": round(oracle.formation(best) * 1000, 2),
                        "unique_evaluations_so_far": oracle.calls})
        print(f"  gen {gen:2d}: best {name_of(best)}  "
              f"E_form = {oracle.formation(best) * 1000:7.2f} meV/atom  "
              f"(unique MACE calls: {oracle.calls})")
        elite = scored[: pop_size // 3]
        children = list(elite)
        while len(children) < pop_size:
            a, b = elite[rng.integers(len(elite))], elite[rng.integers(len(elite))]
            cut = rng.integers(1, N_SITES)
            child = list(a[:cut] + b[cut:])
            for k in range(N_SITES):
                if rng.random() < p_mut:
                    child[k] ^= 1
            children.append(tuple(child))
        pop = children
    return min(pop, key=oracle.formation), history


def reference_table():
    ref = reference_ideals_8atom()
    e_cu, e_au = pure_per_atom(lambda b: ref[b][0], N_SITES)
    table = {}
    for bits, (e, in_training) in ref.items():
        n_au = sum(bits)
        table[fingerprint(bits, REPEAT)] = {
            "ordering": "".join(map(str, bits)),
            "x_au": n_au / N_SITES,
            "ref_formation": formation_energy_per_atom(e, N_SITES - n_au, n_au, e_cu, e_au),
            "in_training": in_training,
        }
    return table


def main():
    dev = device()
    banner(f"Step 3 - evolutionary search over 8-site Cu-Au orderings with fine-tuned MACE ({dev})")
    oracle = Oracle(mace_calculator(MODELS / f"{MODEL_NAME}.model", dev))
    ref = reference_table()

    t0 = time.time()
    champion, history = genetic_search(oracle, lambda bits: ref[fingerprint(bits, REPEAT)]["ordering"])
    ga_calls, ga_time = oracle.calls, time.time() - t0

    t0 = time.time()
    rows = []
    for r in ref.values():
        bits = tuple(int(c) for c in r["ordering"])
        rows.append(r | {"mace_formation": oracle.formation(bits)})
    brute_time = time.time() - t0

    xs = [r["x_au"] for r in rows]
    mace_hull = {rows[k]["ordering"] for k in lower_hull(xs, [r["mace_formation"] for r in rows])}
    ref_hull = {rows[k]["ordering"] for k in lower_hull(xs, [r["ref_formation"] for r in rows])}
    for r in rows:
        r["on_mace_hull"] = r["ordering"] in mace_hull
        r["on_ref_hull"] = r["ordering"] in ref_hull

    champ = ref[fingerprint(champion, REPEAT)]
    champ_mace = oracle.formation(champion)
    ref_best = min(rows, key=lambda r: r["ref_formation"])
    mixed_err = np.mean([abs(r["mace_formation"] - r["ref_formation"]) for r in rows if 0 < r["x_au"] < 1]) * 1000
    heldout_err = np.mean([abs(r["mace_formation"] - r["ref_formation"]) for r in rows if not r["in_training"]]) * 1000

    seen = "was in the training set" if champ["in_training"] else "was NEVER in the training set"
    print(f"\nGA champion: {champ['ordering']}  (x_Au = {champ['x_au']:.3f}) -- this ordering {seen}")
    print(f"  MACE      E_form = {champ_mace * 1000:7.2f} meV/atom")
    print(f"  reference E_form = {champ['ref_formation'] * 1000:7.2f} meV/atom   "
          f"(error {abs(champ_mace - champ['ref_formation']) * 1000:.2f} meV/atom)")
    gap = (champ["ref_formation"] - ref_best["ref_formation"]) * 1000
    if ref_best["ordering"] == champ["ordering"]:
        verdict = "MATCH: the GA found the reference ground state"
    elif gap < 1.0:
        verdict = f"effectively a MATCH: reference puts the champion only {gap:.2f} meV/atom above its ground state"
    else:
        verdict = f"MISS: champion sits {gap:.1f} meV/atom above the reference ground state"
    print(f"  reference global minimum: {ref_best['ordering']} at {ref_best['ref_formation'] * 1000:.2f} meV/atom")
    print(f"  -> {verdict}")
    print(f"\nconvex hull: MACE puts {len(mace_hull)} orderings on the hull, reference {len(ref_hull)}; "
          f"agreement on {len(mace_hull & ref_hull)}/{len(ref_hull)} reference ground states")
    print(f"mean |MACE - reference| over the {sum(1 for r in rows if not r['in_training'])} never-trained orderings: "
          f"{heldout_err:.1f} meV/atom (all mixed: {mixed_err:.1f})")
    print(f"GA: {ga_calls} unique MACE evaluations in {ga_time:.1f} s; "
          f"brute force over all {len(rows)} distinct orderings took {brute_time:.1f} s more")

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    (RESULTS / "discovery.json").write_text(json.dumps({
        "champion": champ | {"mace_formation": champ_mace},
        "reference_global_minimum": ref_best,
        "ga_history": history,
        "ga_unique_evaluations": ga_calls,
        "ga_seconds": round(ga_time, 2),
        "all_orderings": rows,
    }, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs_arr = np.array(xs)
    ref_arr = np.array([r["ref_formation"] for r in rows]) * 1000
    mace_arr = np.array([r["mace_formation"] for r in rows]) * 1000
    trained = np.array([r["in_training"] for r in rows])
    ax.scatter(xs_arr, ref_arr, s=40, facecolors="none", edgecolors="k", label="reference")
    ax.scatter(xs_arr[trained], mace_arr[trained], s=18, color="#7b1fa2", label="MACE (ordering in training)")
    ax.scatter(xs_arr[~trained], mace_arr[~trained], s=18, color="#c2185b", label="MACE (never seen)")
    for hull, arr, color, ls, lab in ((ref_hull, ref_arr, "k", "--", "reference hull"), (mace_hull, mace_arr, "#c2185b", "-", "MACE hull")):
        idx = sorted([k for k, r in enumerate(rows) if r["ordering"] in hull], key=lambda k: xs[k])
        ax.plot(xs_arr[idx], arr[idx], color=color, ls=ls, lw=1.2, label=lab)
    ax.scatter([champ["x_au"]], [champ_mace * 1000], marker="*", s=220, color="#ffb300", edgecolors="k", zorder=5, label="GA champion")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Au fraction")
    ax.set_ylabel("formation energy (meV/atom)")
    ax.set_title("Cu-Au ordering search: MACE vs reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "hull.png", dpi=140)
    print(f"\nwrote {FIGURES / 'hull.png'} and {RESULTS / 'discovery.json'}")


if __name__ == "__main__":
    main()
