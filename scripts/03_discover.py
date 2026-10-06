"""Step 3: evolutionary search for the most stable Cu-Au ordering, scored by the
fine-tuned MACE instead of DFT. The champion is then graded against reference
data for the same ordering.

Search space: all 2^8 decorations of an 8-site FCC supercell (27 distinct
orderings after symmetry). Small enough to also brute-force, which shows how
many evaluations the genetic algorithm actually needed.
"""

import json
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, MODELS, MODEL_NAME, RESULTS, banner, build, canonical, device, distinct_orderings, formation_energy_per_atom, lower_hull, mace_calculator, progress, pure_per_atom, reference_ideals_8atom  # noqa: E402

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
        key = canonical(bits, REPEAT)
        if key not in self.cache:
            at = build(bits, repeat=REPEAT)
            at.calc = self.calc
            self.cache[key] = float(at.get_potential_energy())
            self.calls += 1
            if hasattr(self, "e_cu"):
                n_au = sum(bits)
                ef = formation_energy_per_atom(self.cache[key], N_SITES - n_au, n_au, self.e_cu, self.e_au)
                progress(type="eval", ordering=key, x_au=n_au / N_SITES, ef=round(ef * 1000, 2), calls=self.calls)
        return self.cache[key]

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
        progress(type="generation", gen=gen, calls=oracle.calls,
                 population=[{"bits": "".join(map(str, b)), "ordering": canonical(b, REPEAT),
                              "ef": round(oracle.formation(b) * 1000, 2), "x_au": sum(b) / N_SITES} for b in scored])
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


def formula(x_au):
    from math import gcd

    n_au = round(x_au * N_SITES)
    n_cu = N_SITES - n_au
    g = gcd(n_cu, n_au) or 1
    cu, au = n_cu // g, n_au // g
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in (("Cu", cu), ("Au", au)) if n)


def reference_table():
    ref = reference_ideals_8atom()
    e_cu, e_au = pure_per_atom(lambda b: ref[b][0], N_SITES)
    table = {}
    for bits, (e, in_training) in ref.items():
        n_au = sum(bits)
        table[canonical(bits, REPEAT)] = {
            "ordering": "".join(map(str, bits)),
            "x_au": n_au / N_SITES,
            "ref_formation": formation_energy_per_atom(e, N_SITES - n_au, n_au, e_cu, e_au),
            "in_training": in_training,
        }
    return table


def main():
    dev = device()
    banner(f"Step 3 - discovery: evolutionary search with the network as the judge ({dev})\n"
           "The question: which way of placing copper and gold atoms on 8 lattice sites\n"
           "is the most stable? Each arrangement is an 8-bit string (0 = Cu, 1 = Au), so\n"
           "the search space is 2^8 = 256 strings, 27 of them genuinely different. A\n"
           "genetic algorithm evolves a population of strings; the fitness function is\n"
           "the fine-tuned network's energy, not the simulator's. At the end the network's\n"
           "best guesses are checked against the simulator - the 'verify' step.")
    progress(type="start", step="discover", n_sites=N_SITES, n_distinct=len(distinct_orderings(REPEAT)))
    oracle = Oracle(mace_calculator(MODELS / f"{MODEL_NAME}.model", dev))
    ref = reference_table()

    def name_of(bits):
        key = canonical(bits, REPEAT)
        return ref[key]["ordering"] if key in ref else key

    t0 = time.time()
    champion, history = genetic_search(oracle, name_of)
    ga_calls, ga_time = oracle.calls, time.time() - t0

    # MACE evaluates every distinct ordering; the reference covers those with a DFT label
    t0 = time.time()
    rows = []
    for rep in distinct_orderings(REPEAT):
        key = canonical(rep, REPEAT)
        r = ref.get(key, {"ordering": key, "x_au": sum(rep) / N_SITES, "ref_formation": None, "in_training": False})
        rows.append(r | {"mace_formation": oracle.formation(rep)})
    brute_time = time.time() - t0
    labelled = [r for r in rows if r["ref_formation"] is not None]
    unlabelled = [r for r in rows if r["ref_formation"] is None]

    xs = [r["x_au"] for r in rows]
    mace_hull = {rows[k]["ordering"] for k in lower_hull(xs, [r["mace_formation"] for r in rows])}
    ref_hull = {labelled[k]["ordering"] for k in lower_hull([r["x_au"] for r in labelled], [r["ref_formation"] for r in labelled])}
    for r in rows:
        r["on_mace_hull"] = r["ordering"] in mace_hull
        r["on_ref_hull"] = r["ordering"] in ref_hull
    progress(type="hull",
             network=sorted([{"x_au": r["x_au"], "ef": round(r["mace_formation"] * 1000, 2)} for r in rows if r["on_mace_hull"]], key=lambda d: d["x_au"]),
             reference=sorted([{"x_au": r["x_au"], "ef": round(r["ref_formation"] * 1000, 2)} for r in labelled if r["on_ref_hull"]], key=lambda d: d["x_au"]),
             reference_points=[{"ordering": canonical(tuple(int(c) for c in r["ordering"]), REPEAT), "x_au": r["x_au"],
                                "ef": round(r["ref_formation"] * 1000, 2), "in_training": r["in_training"]} for r in labelled])

    champ = next(r for r in rows if r["ordering"] == name_of(champion))
    champ_mace = champ["mace_formation"]
    ref_best = min(labelled, key=lambda r: r["ref_formation"])
    heldout = [r for r in labelled if not r["in_training"] and 0 < r["x_au"] < 1]
    heldout_err = np.mean([abs(r["mace_formation"] - r["ref_formation"]) for r in heldout]) * 1000

    seen = "was in the training set" if champ["in_training"] else "was NEVER in the training set"
    print(f"\nGA champion: {champ['ordering']}  = {formula(champ['x_au'])}  (x_Au = {champ['x_au']:.3f}) -- this ordering {seen}")
    print(f"  MACE      E_form = {champ_mace * 1000:7.2f} meV/atom")
    if champ["ref_formation"] is None:
        print("  reference E_form = (no DFT label for this ordering yet) -> the agent would request one now")
    else:
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

    ref_gs_at = {}
    for r in labelled:
        if r["x_au"] not in ref_gs_at or r["ref_formation"] < ref_gs_at[r["x_au"]]["ref_formation"]:
            ref_gs_at[r["x_au"]] = r
    shortlist = sorted([r for r in rows if r["on_mace_hull"] and 0 < r["x_au"] < 1], key=lambda r: r["x_au"])
    print("\nVerify step - the agent sends the vertices of MACE's hull (its predicted ground state at each composition) to DFT:")
    print(f"  {'ordering':9s} {'x_Au':>5s} {'MACE':>8s} {'reference':>10s} {'error':>7s}  (meV/atom)")
    hits = 0
    for r in shortlist:
        if r["ref_formation"] is None:
            print(f"  {r['ordering']:9s} {r['x_au']:5.3f} {r['mace_formation'] * 1000:8.2f} {'--':>10s} {'--':>7s}  no DFT label yet: request it")
            progress(type="verify", ordering=canonical(tuple(int(c) for c in r["ordering"]), REPEAT), x_au=r["x_au"],
                     mace_ef=round(r["mace_formation"] * 1000, 2), ref_ef=None, confirmed=None)
            continue
        is_gs = ref_gs_at[r["x_au"]]["ordering"] == r["ordering"]
        hits += is_gs
        progress(type="verify", ordering=canonical(tuple(int(c) for c in r["ordering"]), REPEAT), x_au=r["x_au"],
                 mace_ef=round(r["mace_formation"] * 1000, 2), ref_ef=round(r["ref_formation"] * 1000, 2), confirmed=bool(is_gs))
        flag = "  confirmed: reference ground state at this composition" if is_gs else \
            f"  rejected: reference prefers {ref_gs_at[r['x_au']]['ordering']} ({ref_gs_at[r['x_au']]['ref_formation'] * 1000:.2f})"
        print(f"  {r['ordering']:9s} {r['x_au']:5.3f} {r['mace_formation'] * 1000:8.2f} {r['ref_formation'] * 1000:10.2f} "
              f"{abs(r['mace_formation'] - r['ref_formation']) * 1000:7.2f}{flag}")
    print(f"  -> {hits}/{len(shortlist)} predicted ground states confirmed with {len(shortlist)} DFT calls instead of {len(rows)}; "
          f"rejected ones go back into training (the 'refine' step)")
    if unlabelled:
        print(f"  ({len(unlabelled)} of {len(rows)} orderings have no DFT label yet: {', '.join(r['ordering'] for r in unlabelled)})")

    champion_is_gs = champ["ref_formation"] is not None and ref_best["ordering"] == champ["ordering"]
    print("\nIn plain terms: the search asked the neural network about "
          f"{ga_calls} arrangements (about {ga_time / max(ga_calls, 1) * 1000:.0f} ms each) and picked "
          f"{formula(champ['x_au'])} in arrangement {champ['ordering']}. "
          + ("The simulator agrees that this is the most stable arrangement of all. "
             if champion_is_gs else "The simulator ranks a different arrangement first - the network's guess was close but wrong, which is exactly why we check. ")
          + f"Of the network's best pick at each mixing ratio, {hits} of {len(shortlist)} were confirmed by the simulator, "
          f"using {len(shortlist)} slow calculations instead of {len(rows)}.")
    progress(type="champion", ordering=canonical(champion, REPEAT), formula=formula(champ["x_au"]), x_au=champ["x_au"],
             mace_ef=round(champ_mace * 1000, 2), ref_ef=None if champ["ref_formation"] is None else round(champ["ref_formation"] * 1000, 2),
             is_ground_state=champion_is_gs, in_training=champ["in_training"], hits=hits, checked=len(shortlist),
             calls=ga_calls, ms_per_call=round(ga_time / max(ga_calls, 1) * 1000))
    summary = {
        "n_orderings": len(rows),
        "ga_unique_evaluations": ga_calls,
        "ga_ms_per_evaluation": round(ga_time / max(ga_calls, 1) * 1000, 1),
        "champion_ordering": champ["ordering"],
        "champion_formula": formula(champ["x_au"]),
        "champion_in_training": champ["in_training"],
        "champion_is_reference_ground_state": champion_is_gs,
        "champion_error_meV": None if champ["ref_formation"] is None else round(abs(champ_mace - champ["ref_formation"]) * 1000, 2),
        "hull_vertices_checked": len(shortlist),
        "hull_vertices_confirmed": hits,
        "hull_agreement": f"{len(mace_hull & ref_hull)}/{len(ref_hull)}",
        "heldout_mean_error_meV": round(float(heldout_err), 1),
    }
    print(f"\nconvex hull: MACE puts {len(mace_hull)} orderings on the hull, reference {len(ref_hull)}; "
          f"agreement on {len(mace_hull & ref_hull)}/{len(ref_hull)} reference ground states")
    print(f"mean |MACE - reference| over the {len(heldout)} never-trained mixed orderings with labels: {heldout_err:.1f} meV/atom")
    print(f"GA: {ga_calls} unique MACE evaluations in {ga_time:.1f} s; "
          f"brute force over all {len(rows)} distinct orderings took {brute_time:.1f} s more")

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    (RESULTS / "discovery.json").write_text(json.dumps({
        "summary": summary,
        "champion": champ | {"mace_formation": champ_mace},
        "reference_global_minimum": ref_best,
        "shortlist": shortlist,
        "ga_history": history,
        "ga_unique_evaluations": ga_calls,
        "ga_seconds": round(ga_time, 2),
        "all_orderings": rows,
    }, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs_arr = np.array(xs)
    ref_arr = np.array([np.nan if r["ref_formation"] is None else r["ref_formation"] for r in rows]) * 1000
    mace_arr = np.array([r["mace_formation"] for r in rows]) * 1000
    trained = np.array([r["in_training"] for r in rows])
    ax.scatter(xs_arr, ref_arr, s=40, facecolors="none", edgecolors="k", label="physics simulator (DFT)")
    ax.scatter(xs_arr[trained], mace_arr[trained], s=18, color="#7b1fa2", label="network (arrangement was in training)")
    ax.scatter(xs_arr[~trained], mace_arr[~trained], s=18, color="#c2185b", label="network (never seen)")
    for hull, arr, color, ls, lab in ((ref_hull, ref_arr, "k", "--", "most stable per ratio: simulator"), (mace_hull, mace_arr, "#c2185b", "-", "most stable per ratio: network")):
        idx = sorted([k for k, r in enumerate(rows) if r["ordering"] in hull], key=lambda k: xs[k])
        ax.plot(xs_arr[idx], arr[idx], color=color, ls=ls, lw=1.2, label=lab)
    ax.scatter([champ["x_au"]], [champ_mace * 1000], marker="*", s=220, color="#ffb300", edgecolors="k", zorder=5, label="search result")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("fraction of gold atoms (0 = pure copper, 1 = pure gold)")
    ax.set_ylabel("stability score: formation energy (meV/atom)\nlower = more stable than the pure metals")
    ax.set_title("Which Cu/Au arrangement is most stable? Network vs simulator, all 27 arrangements", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "hull.png", dpi=140)
    progress(type="done", step="discover")
    print(f"\nwrote {FIGURES / 'hull.png'} and {RESULTS / 'discovery.json'}")


if __name__ == "__main__":
    main()
