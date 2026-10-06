# MACE fine-tuning + evolutionary materials discovery, in ~8 minutes

A live, one-shot demo: fine-tune a MACE foundation model on a small set of
reference (DFT) calculations for Cu–Au alloys, check it against held-out
structures it never saw, then let a genetic algorithm search for the most
stable Cu–Au ordering using MACE as the energy oracle instead of DFT.

The point: DFT is too slow to sit inside a discovery loop. A fine-tuned MLIP is
fast enough to be called thousands of times, and accurate enough to be trusted
— and we verify that trust against DFT it was never trained on.

## For an agent (or a human) running this

Do these in order. Each step prints what it did; you don't need to edit anything.

1. Create a virtual environment and install the dependencies (CPU-only torch is
   fine; a CUDA GPU is used automatically if torch finds one):

   ```
   python -m venv .venv
   .venv/Scripts/python -m pip install -r requirements.txt     # Windows
   .venv/bin/python   -m pip install -r requirements.txt       # macOS / Linux
   ```

2. Run the whole pipeline:

   ```
   .venv/Scripts/python run_demo.py        # Windows
   .venv/bin/python   run_demo.py          # macOS / Linux
   ```

   This runs three scripts and prints a timing table at the end:

   | Step | Script | What it does | Output |
   | --- | --- | --- | --- |
   | 1 | `scripts/01_finetune.py` | fine-tunes MACE-MP-0 (small) on `data/train.xyz` | `models/cuau_ft.model` |
   | 2 | `scripts/02_validate.py` | foundation vs fine-tuned model on `data/holdout.xyz` | `figures/parity.png`, `results/validation.json` |
   | 3 | `scripts/03_discover.py` | genetic algorithm over Cu–Au orderings, scored by MACE, graded against held-out reference | `figures/hull.png`, `results/discovery.json` |

3. Report back: the timing table, the validation table from step 2 (error before
   vs after fine-tuning), and the step-3 verdict (did the GA champion match the
   reference ground state, and by how many meV/atom did MACE miss its energy).
   Open `figures/parity.png` and `figures/hull.png`.

## How long it takes

Measured on a Windows laptop (CPU-only torch, no GPU used):

| | time |
| --- | --- |
| `pip install -r requirements.txt` (torch is the bulk of it) | ~4 min |
| step 1 fine-tune, 20 epochs | ~3 min |
| step 2 validate | ~15 s |
| step 3 discover | ~10 s |
| **pipeline total** | **~3.5 min** |

So: install *before* the session if you can, and the live part is about three
and a half minutes. On a laptop GPU (GTX 1650) the pipeline takes about two
minutes.

Knobs, all on `run_demo.py`: `--epochs N` (20 by default; 30 gets the
held-out error from ~5 to ~2 meV/atom for another ~100 s on CPU, 15 is too
few), `--skip-finetune` to reuse `models/cuau_ft.model` from a previous run.

The first run downloads the MACE-MP-0 small foundation model (a few MB) into
`~/.cache/mace`; after that everything is offline.

## Prompt to give the agent

> Clone this repo, read README.md, and run the demo exactly as it describes:
> create the venv, install the requirements, run `run_demo.py`. Then report
> the timing table, the before/after validation table, the step-3 verdict, and
> open the two figures. Don't change any code.

## What is in the data

Everything the demo trains on or is graded against is committed, so nothing
expensive runs live.

- `data/train.xyz` — 135 structures. The 4-atom conventional FCC cell in its 5
  symmetry-distinct Cu/Au decorations (Cu₄, Cu₃Au, L1₀-CuAu, CuAu₃, Au₄), each
  with a 7-point volume scan and 12 rattled snapshots; plus 10 of the 25
  distinct orderings of an 8-atom (2×1×1) supercell (pure Cu₈ and Au₈, and 8
  mixed ones chosen at random), each ideal plus three rattled copies. Labels are
  total energy (`REF_energy`) and per-atom forces (`REF_forces`).
- `data/holdout.xyz` — 45 structures: the other 15 mixed 8-atom orderings,
  each ideal plus two rattled copies. Never used in training.

That is 180 reference single-point calculations in total. Think of it as an
active-learning snapshot: we paid for DFT on 40% of the ordering space, and ask
the model to rank the other 60%.

All 25 orderings together are the answer key for step 3: the GA searches the
same 8-site space, so whichever ordering it lands on, its reference energy is
known, and the script says whether that ordering was in the training set.

> **Stand-in data.** The labels currently in `data/` were generated with ASE's
> EMT potential (`tools/make_reference_data.py`), not DFT, so the pipeline could
> be timed before the DFT runs finished. The file format and keys are what the
> DFT run will produce; swapping in real labels changes nothing downstream.
> Two consequences worth knowing: the foundation model was trained on DFT, so
> judging it against EMT makes its "before fine-tuning" error look worse than
> it will with real DFT labels; and EMT barely favours ordering (Cu₃Au is only
> ~11 meV/atom below the elements, where PBE gives ~40-50), so the real hull
> will be deeper and the story clearer.

## Why Cu–Au

It is a textbook ordering problem with a known answer: Cu₃Au, CuAu (L1₀) and
CuAu₃ form ordered intermetallics. That gives the search something real to
rediscover, and the 4-atom training cells are small enough that the whole
reference dataset is a day of DFT on a laptop.

## Producing the reference data (authors only)

The structures are fixed once, in `data/structures.xyz`, and every labeller
works from that file:

```
python tools/build_structures.py          # -> data/structures.xyz (180 structures)
python tools/make_reference_data.py       # EMT stand-in -> data/train.xyz, data/holdout.xyz
```

### With VASP

1. Make the calculation directories, concatenating POTCARs from a local
   `potpaw_PBE/` (needs `potpaw_PBE/Cu/POTCAR` and `potpaw_PBE/Au/POTCAR`;
   standard PBE `Cu` and `Au`, 11 valence electrons each):

   ```
   python tools/vasp/make_inputs.py --potcar-dir potpaw_PBE
   ```

   This writes `vasp/calcs/<name>/{POSCAR,KPOINTS,INCAR,POTCAR,meta.json}`
   and `vasp/calcs/list.txt`. Settings (`tools/vasp/INCAR`): PBE, ENCUT 400,
   Methfessel-Paxton smearing 0.1 eV, EDIFF 1e-6, Γ-centred 8×8×8 (4-atom
   cells) / 4×8×8 (8-atom cells), single point with forces. Nothing is
   relaxed: the volume scan brackets the equilibrium, and the hull in step 3
   compares MACE and DFT on the same fixed geometries.

2. Copy `vasp/` to the cluster and submit `tools/vasp/submit_array.sh` (a
   SLURM array over `list.txt`; adapt the header, module and launcher).

3. Copy `vasp/calcs/` back and collect:

   ```
   python tools/vasp/collect.py
   ```

   It takes `energy(sigma->0)` and the forces from each OUTCAR, restores the
   structure's own site order, writes `data/train.xyz` and `data/holdout.xyz`,
   and lists any run that is missing or did not converge.

POTCARs and `vasp/calcs/` are gitignored: the potentials are licensed and the
directories are regenerable.
