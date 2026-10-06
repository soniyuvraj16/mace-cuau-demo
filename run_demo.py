"""Run the whole demo end to end, with a live dashboard, and report wall-clock
time per step.

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


def write_replay():
    events = [json.loads(line) for line in (RESULTS / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    (RESULTS / "progress.js").write_text("window.PROGRESS = " + json.dumps(events) + ";\n", encoding="utf-8", newline="\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-finetune", action="store_true", help="reuse models/cuau_ft.model")
    p.add_argument("--epochs", type=int, default=None, help="override fine-tuning epochs")
    p.add_argument("--lr", type=float, default=None, help="override fine-tuning learning rate")
    p.add_argument("--no-dashboard", action="store_true", help="do not serve/open the live dashboard")
    p.add_argument("--hold", action="store_true", help="keep the dashboard server running until Enter is pressed")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "progress.jsonl").write_text("", encoding="utf-8")
    server = None
    if not args.no_dashboard:
        server, url = start_dashboard()
        time.sleep(1.5)

    timings = {}
    total0 = time.time()
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    for name, script in STEPS:
        if name == "finetune" and args.skip_finetune:
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
    with open(RESULTS / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "finished", "timings": timings, "t": round(time.time(), 3)}) + "\n")
    write_replay()

    print("\n" + "=" * 72)
    print("TIMINGS")
    for k, v in timings.items():
        print(f"  {k:10s} {v:7.1f} s")
    print("=" * 72)
    print(plain_summary(timings))

    if server is not None:
        print(f"\nthe dashboard is still at {url}; open dashboard/index.html any time to replay the run")
        if args.hold:
            input("press Enter to stop the dashboard server... ")
        else:
            time.sleep(4)  # let the page fetch the last events before the server goes away
        server.shutdown()


def plain_summary(timings):
    val = json.loads((RESULTS / "validation.json").read_text())
    disc = json.loads((RESULTS / "discovery.json").read_text())["summary"]
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
