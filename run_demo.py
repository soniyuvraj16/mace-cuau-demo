"""Run the demo end to end, with a live dashboard, and report wall-clock time
per step.

Two modes:
  quick (default)  load the fine-tuned model shipped in models/cuau_ft.model,
                   replay its recorded training curve on the dashboard (labelled
                   as recorded), then run the test and the search live. About
                   30 s after the install.
  full             fine-tune live first (about 3 min on a laptop CPU, longer on
                   slower machines), then test and search.

The dashboard (dashboard/index.html) animates events the scripts append to
results/progress.jsonl. It is served from a local HTTP server while the demo
runs and opened in the default browser. Afterwards the same page replays the
run from results/progress.js, so `dashboard/index.html` can simply be opened as
a file to show the animation again.
"""

import argparse
import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
STEPS = [
    ("finetune", "scripts/01_finetune.py"),
    ("validate", "scripts/02_validate.py"),
    ("discover", "scripts/03_discover.py"),
]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def start_dashboard():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/dashboard/"
    opened = webbrowser.open(url)
    print(f"live dashboard: {url}" + ("" if opened else "  (open this in a browser)"))
    return server, url


def emit(event):
    event.setdefault("t", round(time.time(), 3))
    with open(RESULTS / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def replay_recorded_training(log):
    """Feed the shipped model's recorded training curve to the dashboard."""
    emit({"type": "start", "step": "finetune", "epochs": log["epochs_planned"], "device": log["device"],
          "recorded": True, "recorded_seconds": log["seconds"], "recorded_date": log["date"]})
    for e in log["epochs"]:
        emit({"type": "epoch", "epoch": e["epoch"], "mae_e": e["mae_e"], "mae_f": e["mae_f"], "elapsed": e["elapsed"], "recorded": True})
    emit({"type": "done", "step": "finetune", "seconds": log["seconds"], "recorded": True})


def write_replay():
    events = [json.loads(line) for line in (RESULTS / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    (RESULTS / "progress.js").write_text("window.PROGRESS = " + json.dumps(events) + ";\n", encoding="utf-8", newline="\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="quick",
                   help="quick: use the shipped fine-tuned model (default); full: fine-tune live")
    p.add_argument("--skip-finetune", action="store_true", help="same as --mode quick (kept for compatibility)")
    p.add_argument("--epochs", type=int, default=None, help="full mode: override fine-tuning epochs")
    p.add_argument("--lr", type=float, default=None, help="full mode: override fine-tuning learning rate")
    p.add_argument("--no-dashboard", action="store_true", help="do not serve/open the live dashboard")
    p.add_argument("--hold", action="store_true", help="keep the dashboard server running until Enter is pressed")
    args = p.parse_args()
    mode = "quick" if args.skip_finetune else args.mode

    model = MODELS / "cuau_ft.model"
    log_path = MODELS / "training_log.json"
    if mode == "quick" and not model.exists():
        print("no shipped model found in models/; switching to --mode full")
        mode = "full"

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "progress.jsonl").write_text("", encoding="utf-8")
    server = url = None
    if not args.no_dashboard:
        server, url = start_dashboard()
        time.sleep(1.5)

    timings = {}
    recorded = None
    total0 = time.time()
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    print(f"mode: {mode}" + (" — using the shipped fine-tuned model, training curve replayed from models/training_log.json" if mode == "quick" else " — fine-tuning live"))
    for name, script in STEPS:
        if name == "finetune" and mode == "quick":
            if log_path.exists():
                recorded = json.loads(log_path.read_text(encoding="utf-8"))
                replay_recorded_training(recorded)
            timings["finetune"] = 0.0
            continue
        cmd = [sys.executable, str(ROOT / script)]
        if name == "finetune" and args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if name == "finetune" and args.lr is not None:
            cmd += ["--lr", str(args.lr)]
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
        timings[name] = round(time.time() - t0, 1)
    timings["total"] = round(time.time() - total0, 1)

    (RESULTS / "timings.json").write_text(json.dumps(timings, indent=2))
    emit({"type": "finished", "timings": timings, "mode": mode,
          "recorded_finetune_seconds": recorded["seconds"] if recorded else None})
    write_replay()

    print("\n" + "=" * 72)
    print("TIMINGS")
    for k, v in timings.items():
        label = f"{v:7.1f} s" if not (k == "finetune" and mode == "quick") else f"   shipped model (trained earlier in {recorded['seconds']:.0f} s)" if recorded else "   shipped model"
        print(f"  {k:10s} {label}")
    print("=" * 72)
    print(plain_summary(timings, mode, recorded))

    if server is not None:
        print(f"\nthe dashboard is still at {url}; open dashboard/index.html any time to replay the run")
        if args.hold:
            input("press Enter to stop the dashboard server... ")
        else:
            time.sleep(4)  # let the page fetch the last events before the server goes away
        server.shutdown()


def plain_summary(timings, mode, recorded):
    val = json.loads((RESULTS / "validation.json").read_text())
    disc = json.loads((RESULTS / "discovery.json").read_text())["summary"]
    base, ft = val["foundation (MACE-MP-0 small)"], val["fine-tuned"]
    minutes = timings["total"] / 60
    verdict = ("and the physics simulator confirms it is the most stable arrangement of all"
               if disc["champion_is_reference_ground_state"] else
               "but the physics simulator ranks a different arrangement first - a near miss the verify step caught")
    if mode == "quick":
        learn = (f"     We fine-tuned a pretrained neural network (MACE) on those scores earlier — a recorded run of "
                 f"{recorded['seconds'] / 60:.1f} min on a laptop CPU — and loaded the saved model today. (--mode full trains it live.)"
                 if recorded else "     We fine-tuned a pretrained neural network (MACE) on those scores earlier and loaded the saved model today.")
    else:
        learn = f"     We fine-tuned a pretrained neural network (MACE) on those scores in {timings['finetune'] / 60:.1f} min."
    return "\n".join([
        "WHAT JUST HAPPENED, IN PLAIN TERMS",
        f"  1. We had a slow but trusted physics simulator (DFT) score 135 copper-gold arrangements.",
        learn,
        f"  2. On 51 arrangements it never saw, the network's energy error dropped from "
        f"{base['formation_energy_mae_meV_per_atom']:.0f} to {ft['formation_energy_mae_meV_per_atom']:.0f} meV/atom and its force error from "
        f"{base['force_rmse_meV_per_A']:.0f} to {ft['force_rmse_meV_per_A']:.0f} meV/A, while answering in milliseconds.",
        f"  3. An evolutionary search used the network as its fitness function, asked it about {disc['ga_unique_evaluations']} "
        f"arrangements at ~{disc['ga_ms_per_evaluation']:.0f} ms each, and picked {disc['champion_formula']} "
        f"(bit string {disc['champion_ordering']}) - {verdict}.",
        f"     Checking the network's best pick at each mixing ratio took {disc['hull_vertices_checked']} slow calculations instead of "
        f"{disc['n_orderings']}; {disc['hull_vertices_confirmed']} of {disc['hull_vertices_checked']} were confirmed.",
        f"  The live part - test, search, verify{', and learn' if mode == 'full' else ''} - ran in {minutes:.1f} minutes on this machine. With the simulator alone,",
        f"  scoring every arrangement in the search would have taken hours; in a real search space (thousands of candidates,",
        f"  bigger cells) it would be impossible. That is the role a learned potential plays inside an agent's discovery loop.",
    ])


if __name__ == "__main__":
    main()
