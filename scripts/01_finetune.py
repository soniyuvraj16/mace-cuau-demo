"""Step 1: fine-tune the MACE-MP-0 (small) foundation model on data/train.xyz."""

import argparse
import os
import shutil
import subprocess
import sys
import time

from common import DATA, MODELS, MODEL_NAME, banner, device


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
    if MODELS.exists():
        shutil.rmtree(MODELS)
    MODELS.mkdir()

    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
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
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    model = MODELS / f"{MODEL_NAME}.model"
    if not model.exists():
        sys.exit(f"expected {model} after training")
    print(f"\nfine-tuned model: {model}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
