"""Run the whole demo end to end and report wall-clock time per step."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("finetune", "scripts/01_finetune.py"),
    ("validate", "scripts/02_validate.py"),
    ("discover", "scripts/03_discover.py"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-finetune", action="store_true", help="reuse models/cuau_ft.model")
    p.add_argument("--epochs", type=int, default=None, help="override fine-tuning epochs")
    p.add_argument("--lr", type=float, default=None, help="override fine-tuning learning rate")
    args = p.parse_args()

    timings = {}
    total0 = time.time()
    for name, script in STEPS:
        if name == "finetune" and args.skip_finetune:
            continue
        cmd = [sys.executable, str(ROOT / script)]
        if name == "finetune" and args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if name == "finetune" and args.lr is not None:
            cmd += ["--lr", str(args.lr)]
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "PYTHONWARNINGS": "ignore"})
        timings[name] = round(time.time() - t0, 1)
    timings["total"] = round(time.time() - total0, 1)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "timings.json").write_text(json.dumps(timings, indent=2))
    print("\n" + "=" * 72)
    print("TIMINGS")
    for k, v in timings.items():
        print(f"  {k:10s} {v:7.1f} s")
    print("=" * 72)
    print(plain_summary(timings))


def plain_summary(timings):
    val = json.loads((ROOT / "results" / "validation.json").read_text())
    disc = json.loads((ROOT / "results" / "discovery.json").read_text())["summary"]
    base, ft = val["foundation (MACE-MP-0 small)"], val["fine-tuned"]
    minutes = timings["total"] / 60
    verdict = ("and the physics simulator confirms it is the most stable arrangement of all"
               if disc["champion_is_reference_ground_state"] else
               "but the physics simulator ranks a different arrangement first - a near miss the verify step caught")
    return "\n".join([
        "WHAT JUST HAPPENED, IN PLAIN TERMS",
        f"  1. We had a slow but trusted physics simulator (DFT) score 135 copper-gold arrangements.",
        f"     We fine-tuned a pretrained neural network (MACE) on those scores"
        + (f" in {timings['finetune'] / 60:.1f} min." if "finetune" in timings else "."),
        f"  2. On 51 arrangements it never saw, the network's energy error dropped from "
        f"{base['formation_energy_mae_meV_per_atom']:.0f} to {ft['formation_energy_mae_meV_per_atom']:.0f} meV/atom and its force error from "
        f"{base['force_rmse_meV_per_A']:.0f} to {ft['force_rmse_meV_per_A']:.0f} meV/A, while answering in milliseconds.",
        f"  3. An evolutionary search used the network as its fitness function, asked it about {disc['ga_unique_evaluations']} "
        f"arrangements at ~{disc['ga_ms_per_evaluation']:.0f} ms each, and picked {disc['champion_formula']} "
        f"(bit string {disc['champion_ordering']}) - {verdict}.",
        f"     Checking the network's best pick at each mixing ratio took {disc['hull_vertices_checked']} slow calculations instead of "
        f"{disc['n_orderings']}; {disc['hull_vertices_confirmed']} of {disc['hull_vertices_checked']} were confirmed.",
        f"  The whole loop - learn, test, search, verify - ran in {minutes:.1f} minutes on this machine. With the simulator alone,",
        f"  scoring every arrangement in the search would have taken hours; in a real search space (thousands of candidates,",
        f"  bigger cells) it would be impossible. That is the role a learned potential plays inside an agent's discovery loop.",
    ])


if __name__ == "__main__":
    main()
