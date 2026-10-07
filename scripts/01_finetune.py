"""Step 1: fine-tune the MACE-MP-0 (small) foundation model on data/train.xyz."""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

from common import DATA, MODELS, MODEL_NAME, banner, device, progress

EPOCH_LINE = re.compile(r"(Initial|Epoch (\d+)):.*MAE_E_per_atom=\s*([\d.]+) meV, MAE_F=\s*([\d.]+) meV")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--energy-weight", type=float, default=100.0)
    p.add_argument("--forces-weight", type=float, default=10.0)
    p.add_argument("--device", default=device())
    args = p.parse_args()

    banner(f"Step 1 - fine-tune the neural network ({args.device})\n"
           "A slow physics simulator (DFT, minutes per structure) scored 135 Cu-Au\n"
           "arrangements for us: an energy per structure and a force on every atom.\n"
           "We now adapt a pretrained neural network (MACE) to reproduce those numbers,\n"
           "the same way you would fine-tune a pretrained language model on a small\n"
           "domain dataset. Afterwards it answers in milliseconds instead of minutes.")
    MODELS.mkdir(exist_ok=True)
    for sub in ("checkpoints", "logs", "results"):
        shutil.rmtree(MODELS / sub, ignore_errors=True)
    for old in MODELS.glob(f"{MODEL_NAME}*.model"):
        old.unlink()
    progress(type="start", step="finetune", epochs=args.epochs, device=args.device)
    epochs_seen = []

    cmd = [
        sys.executable, "-u", "-m", "mace.cli.run_train",
        "--name", MODEL_NAME,
        "--foundation_model", "small",
        "--multiheads_finetuning", "False",
        "--train_file", str(DATA / "train.xyz"),
        "--valid_fraction", "0.1",
        "--energy_key", "REF_energy",
        "--forces_key", "REF_forces",
        "--E0s", "average",
        "--loss", "weighted",
        "--energy_weight", str(args.energy_weight),
        "--forces_weight", str(args.forces_weight),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--valid_batch_size", "16",
        "--max_num_epochs", str(args.epochs),
        "--ema", "--ema_decay", "0.99",
        "--amsgrad",
        "--default_dtype", "float32",
        "--device", args.device,
        "--num_workers", "0",
        "--seed", "1",
        "--eval_interval", "1",
        "--error_table", "PerAtomMAE",
        "--save_cpu",
        "--model_dir", str(MODELS),
        "--log_dir", str(MODELS / "logs"),
        "--checkpoints_dir", str(MODELS / "checkpoints"),
        "--results_dir", str(MODELS / "results"),
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONUNBUFFERED": "1"})
    for line in proc.stdout:
        print(line, end="")
        m = EPOCH_LINE.search(line)
        if m:
            epoch = -1 if m.group(1) == "Initial" else int(m.group(2))
            rec = dict(epoch=epoch, mae_e=float(m.group(3)), mae_f=float(m.group(4)), elapsed=round(time.time() - t0, 1))
            epochs_seen.append(rec)
            progress(type="epoch", **rec)
    if proc.wait() != 0:
        sys.exit(f"mace_run_train failed with exit code {proc.returncode}")
    model = MODELS / f"{MODEL_NAME}.model"
    if not model.exists():
        sys.exit(f"expected {model} after training")
    seconds = round(time.time() - t0, 1)
    progress(type="done", step="finetune", seconds=seconds)
    (MODELS / "training_log.json").write_text(json.dumps({
        "recorded": True, "date": time.strftime("%Y-%m-%d"), "machine": f"{platform.system()} {platform.machine()}, {args.device}",
        "device": args.device, "epochs_planned": args.epochs, "seconds": seconds,
        "settings": {"foundation_model": "MACE-MP-0 small", "lr": args.lr, "batch_size": args.batch_size,
                     "energy_weight": args.energy_weight, "forces_weight": args.forces_weight, "train_structures": 135},
        "epochs": epochs_seen}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nfine-tuned model: {model}  ({seconds:.0f} s); training curve saved to models/training_log.json")


if __name__ == "__main__":
    main()
