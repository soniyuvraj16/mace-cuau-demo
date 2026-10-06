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


if __name__ == "__main__":
    main()
