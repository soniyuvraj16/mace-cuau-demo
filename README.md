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

Knobs, all on `run_demo.py`: `--epochs N` (20 by default, chosen so install
plus run fits in 8 minutes; **if the environment is already installed, use
`--epochs 30`** — held-out error drops from ~6 to ~3.5 meV/atom for another
~90 s on CPU; 15 is too few), `--skip-finetune` to reuse `models/cuau_ft.model`
from a previous run.

The first run downloads the MACE-MP-0 small foundation model (a few MB) into
`~/.cache/mace`; after that everything is offline.

## Prompt to give the agent

> Clone this repo, read README.md, and run the demo exactly as it describes:
> create the venv, install the requirements, run `run_demo.py` (with
> `--epochs 30` if the environment was already installed). Then report the
> timing table, the before/after validation table, the step-3 champion verdict
> and the verify-step shortlist, and open the two figures. Don't change any
> code.

## What is in the data

Everything the demo trains on or is graded against is committed, so nothing
expensive runs live.

- `data/train.xyz` — 135 structures. The 4-atom conventional FCC cell in its 5
  symmetry-distinct Cu/Au decorations (Cu₄, Cu₃Au, L1₀-CuAu, CuAu₃, Au₄), each
  with a 7-point volume scan and 12 rattled snapshots; plus 10 of the 27
  distinct orderings of an 8-atom (2×1×1) supercell (pure Cu₈ and Au₈, and 8
  mixed ones chosen at random), each ideal plus three rattled copies. Labels are
  total energy (`REF_energy`) and per-atom forces (`REF_forces`).
- `data/holdout.xyz` — the other 17 mixed 8-atom orderings, each ideal plus
  two rattled copies (51 structures). Never used in training.

That is 186 reference single-point calculations in total. Think of it as an
active-learning snapshot: we paid for DFT on under 40% of the ordering space,
and ask the model to rank the rest.

All 27 orderings together are the answer key for step 3: the GA searches the
same 8-site space, so whichever ordering it lands on, its reference energy is
known, and the script says whether that ordering was in the training set (or
flags it if no label exists yet — which is what an agent would then request).

> **Labels are real DFT**: VASP, PBE, computed on TAMU HPRC Grace (see below).
> `tools/make_reference_data.py` can regenerate stand-in labels from ASE's EMT
> potential for timing the pipeline without DFT; it refuses to overwrite DFT
> files unless forced. Two orderings (`00010101`, `01010111`) were found after
> the first 180 calculations, through a bug in the symmetry reduction that had
> merged each with a look-alike; their 6 structures are labelled in a top-up
> run. Until that lands, `data/holdout.xyz` has 45 frames and step 3 reports
> those two orderings as "no DFT label yet".

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

### What the DFT jobs are

186 **static single-point calculations** — no relaxation, no MD. Each one
takes a fixed geometry and returns its total energy and the force on every
atom. DFT is not searching for anything here; it is producing the *labels* the
machine-learned potential is trained on and judged against. Four groups:

| group | count | why it is there |
| --- | --- | --- |
| 4-atom cells, 7-point volume scan (±5 %) for each of the 5 orderings | 35 | energy-vs-volume curve per composition — how stiff each alloy is and where its lattice constant sits |
| 4-atom cells, rattled (σ = 0.05 / 0.10 Å) at three volumes | 60 | non-zero forces: the local shape of the energy surface around each ordering |
| 8-atom cells, 10 orderings (Cu₈, Au₈ + 8 random mixed), ideal + 3 rattled | 40 | ordering patterns that do not exist in a 4-atom cell; teaches the composition dependence |
| 8-atom cells, the other 17 mixed orderings, ideal + 2 rattled | 51 | **never trained on** — validation in step 2, and the answer key for step 3 |

The scientific question the demo answers with these labels: *which arrangement
of Cu and Au on the FCC lattice is most stable at each composition?* — the
ordering ground states and the convex hull. The textbook answer is Cu₃Au (L1₂)
and CuAu (L1₀). The point is that MACE, fine-tuned on the 135 training
labels, predicts the other 17 orderings well enough for a genetic algorithm to
find those ground states using MACE instead of DFT, and the held-out labels
prove it.

### Running them with VASP

1. Make the calculation directories, concatenating POTCARs from `Cu/POTCAR`
   and `Au/POTCAR` (standard PBE `Cu` 22Jun2005 and `Au` 04Oct2007, 11
   valence electrons each; put the two folders in the repo root, they are
   gitignored):

   ```
   python tools/vasp/make_inputs.py --potcar-dir .
   ```

   This writes `vasp/calcs/<name>/{POSCAR,KPOINTS,INCAR,POTCAR,meta.json}`,
   `vasp/calcs/list.txt` and `vasp/submit_array.sh` — `vasp/` is then
   self-contained (~70 MB). Settings (`tools/vasp/INCAR`): PBE, ENCUT 400,
   PREC Accurate, Methfessel-Paxton smearing 0.1 eV, EDIFF 1e-6, Γ-centred
   8×8×8 (4-atom cells) / 4×8×8 (8-atom cells), KPAR 4 × NCORE 3 for 12
   ranks. Nothing is relaxed: the volume scan brackets the equilibrium, and the
   hull in step 3 compares MACE and DFT on the same fixed geometries.

2. On the cluster (written for TAMU HPRC Grace: `intel/2022a`, `vasp/6.3.2`,
   `srun vasp_std`; SLURM array of 186 twelve-core tasks, 30 at a time, 2 h
   limit each — expect well under an hour per task):

   ```
   tar xzf cuau_vasp.tar.gz && cd vasp
   mkdir -p logs
   sbatch submit_array.sh
   ```

   Re-submitting is safe: converged directories are skipped.

3. Copy `vasp/calcs/` back (only `OUTCAR` and `meta.json` are needed) and:

   ```
   python tools/vasp/collect.py
   ```

   It takes `energy(sigma->0)` and the forces from each OUTCAR, restores the
   structure's own site order, writes `data/train.xyz` and `data/holdout.xyz`,
   and lists any run that is missing or did not converge. Then `run_demo.py`
   runs unchanged on real DFT labels.

POTCARs, `vasp/calcs/` and the tarball are gitignored: the potentials are
licensed and the directories are regenerable.
