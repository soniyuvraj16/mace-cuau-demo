"""Step 2: compare foundation vs fine-tuned MACE against held-out reference data.

Energies are compared as formation energies per atom, so a constant per-element
offset between the reference code and the foundation model cancels out and the
comparison is fair for both models.
"""

import json
import time

import matplotlib
import numpy as np
from ase.io import read

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DATA, FIGURES, MODELS, MODEL_NAME, RESULTS, banner, bits_of, build, device, formation_energy_per_atom, mace_calculator, pure_per_atom, reference_ideals_8atom  # noqa: E402


def formation(frames, energies, e_cu, e_au):
    out = []
    for fr, e in zip(frames, energies):
        bits = bits_of(fr)
        out.append(formation_energy_per_atom(e, len(bits) - sum(bits), sum(bits), e_cu, e_au))
    return np.array(out)


def evaluate(calc, frames):
    energies, forces = [], []
    for fr in frames:
        at = fr.copy()
        at.calc = calc
        energies.append(at.get_potential_energy())
        forces.append(at.get_forces())
    return np.array(energies), np.concatenate(forces)


def model_pure_per_atom(calc):
    def total(bits):
        at = build(bits, repeat=(2, 1, 1))
        at.calc = calc
        return float(at.get_potential_energy())

    return pure_per_atom(total, 8)


def metrics(ef_pred, ef_ref, f_pred, f_ref):
    return {
        "formation_energy_mae_meV_per_atom": float(np.mean(np.abs(ef_pred - ef_ref)) * 1000),
        "force_rmse_meV_per_A": float(np.sqrt(np.mean((f_pred - f_ref) ** 2)) * 1000),
        "force_mae_meV_per_A": float(np.mean(np.abs(f_pred - f_ref)) * 1000),
    }


def main():
    dev = device()
    banner(f"Step 2 - validating on {DATA / 'holdout.xyz'} ({dev})")
    frames = read(DATA / "holdout.xyz", ":")
    ref = reference_ideals_8atom()
    ref_cu, ref_au = pure_per_atom(lambda b: ref[b][0], 8)
    ref_e = np.array([fr.info["REF_energy"] for fr in frames])
    ref_f = np.concatenate([fr.arrays["REF_forces"] for fr in frames])
    ref_ef = formation(frames, ref_e, ref_cu, ref_au)
    n_orderings = len({bits_of(fr) for fr in frames})

    results, preds = {}, {}
    for label, path in (("foundation (MACE-MP-0 small)", None), ("fine-tuned", MODELS / f"{MODEL_NAME}.model")):
        t0 = time.time()
        calc = mace_calculator(path, dev)
        e_cu, e_au = model_pure_per_atom(calc)
        e, f = evaluate(calc, frames)
        ef = formation(frames, e, e_cu, e_au)
        results[label] = metrics(ef, ref_ef, f, ref_f) | {"seconds": round(time.time() - t0, 1)}
        preds[label] = (ef, f)

    print(f"\n{'model':32s} {'E_form MAE (meV/atom)':>22s} {'F RMSE (meV/A)':>16s} {'time (s)':>9s}")
    for label, m in results.items():
        print(f"{label:32s} {m['formation_energy_mae_meV_per_atom']:22.1f} {m['force_rmse_meV_per_A']:16.1f} {m['seconds']:9.1f}")
    fm, bm = results["fine-tuned"], results["foundation (MACE-MP-0 small)"]
    print(f"\nfine-tuning changed the formation-energy error by "
          f"{bm['formation_energy_mae_meV_per_atom'] / max(fm['formation_energy_mae_meV_per_atom'], 1e-9):.1f}x "
          f"and the force error by {bm['force_rmse_meV_per_A'] / max(fm['force_rmse_meV_per_A'], 1e-9):.1f}x "
          f"on {len(frames)} held-out structures ({n_orderings} orderings the model never saw).")

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    (RESULTS / "validation.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = {"foundation (MACE-MP-0 small)": "#999999", "fine-tuned": "#c2185b"}
    for label, (ef, f) in preds.items():
        axes[0].scatter(ref_ef * 1000, ef * 1000, s=22, alpha=0.8, label=label, color=colors[label])
        axes[1].scatter(ref_f.ravel(), f.ravel(), s=6, alpha=0.5, label=label, color=colors[label])
    for ax, title, unit in ((axes[0], "Formation energy", "meV/atom"), (axes[1], "Forces", "eV/Å")):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"reference ({unit})")
        ax.set_ylabel(f"MACE ({unit})")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Held-out 8-atom Cu-Au cells: before vs after fine-tuning")
    fig.tight_layout()
    fig.savefig(FIGURES / "parity.png", dpi=140)
    print(f"\nwrote {FIGURES / 'parity.png'} and {RESULTS / 'validation.json'}")


if __name__ == "__main__":
    main()
