# MACE fine-tuning + evolutionary materials discovery, in ~8 minutes

A live, one-shot demo: fine-tune a MACE foundation model on a small set of
reference (DFT) calculations for Cu–Au alloys, check it against held-out
structures it never saw, then let a genetic algorithm search for the most
stable Cu–Au ordering using MACE as the energy oracle instead of DFT.

The point: DFT is too slow to sit inside a discovery loop. A fine-tuned MLIP is
fast enough to be called thousands of times, and accurate enough to be trusted
— and we verify that trust against DFT it was never trained on.

## If you are not from materials science (most of the room)

You don't need any chemistry. Here is the whole demo in CS terms:

- There is a **slow, trusted simulator** (DFT — quantum mechanics solved
  numerically). Give it an arrangement of atoms, it returns a score (energy:
  lower = more stable) and a gradient (the force on every atom). Each call
  takes minutes to hours of cluster time.
- There is a **neural network** (MACE) pretrained to imitate that simulator
  on a huge general dataset. We **fine-tune** it on 135 simulator outputs for
  our specific problem — exactly like fine-tuning a pretrained language model
  on a small domain dataset. Afterwards it answers in ~100 ms.
- The **problem** is a search: 8 lattice sites, each copper or gold, so every
  candidate is an **8-bit string** (`0` = Cu, `1` = Au). 2⁸ = 256 strings, 27
  of them genuinely different after symmetry. Which one is most stable?
- A **genetic algorithm** evolves a population of bit strings. Its **fitness
  function is the neural network**, not the simulator — that is what makes
  the search affordable.
- Then the **verify step**: the network's best guess at each mixing ratio is
  sent to the slow simulator. If the simulator agrees, trust is earned; if
  not, that structure goes into the training set and the loop repeats
  (**propose → evaluate → decide → refine**). The agent spends 3 slow calls
  instead of 27.

The answer is a real material: Cu₃Au in the "L1₂" arrangement is a known
ordered alloy you can look up. Every script prints an "in plain terms"
paragraph after its numbers, and `run_demo.py` ends with a summary written
for this audience.

| term you will see | what it means here |
| --- | --- |
| DFT, "reference", "simulator" | the slow ground-truth physics calculation |
| MACE, "the network", MLIP | the neural network that imitates it |
| foundation model / MACE-MP-0 | the pretrained network before fine-tuning |
| ordering, arrangement, bit string | which of the 8 sites are Cu and which Au |
| formation energy | stability score: energy of the mixture minus the pure metals, per atom; lower = more stable, 0 = no better than keeping the metals apart |
| meV/atom | the unit of that score; rival arrangements differ by tens of meV/atom |
| ground state | the most stable arrangement (at a given mixing ratio) |
| convex hull / "hull vertices" | the most stable arrangement at each mixing ratio, joined up — the set a search is really trying to find |
| held out | structures the network never saw in training |

## For an agent (or a human) running this

Do these in order. Each step prints what it did; you don't need to edit anything.

1. Create a virtual environment and install the dependencies. Nothing else is
   assumed on the machine beyond Python 3.10–3.13; `requirements.txt` pins the
   exact versions the demo was timed with and pulls CPU-only torch (a CUDA GPU
   is used automatically if a CUDA torch happens to be installed instead). The
   MACE-MP-0 foundation model (a few MB) is downloaded on first run.

   ```
   python -m venv .venv
   .venv/Scripts/python -m pip install -r requirements.txt     # Windows
   .venv/bin/python   -m pip install -r requirements.txt       # macOS / Linux
   ```

2. Ask which mode, then run the pipeline. **Quick is recommended when time is
   short** (a class demo); full is for anyone who wants to watch the training
   happen and can wait a few minutes.

   ```
   .venv/Scripts/python run_demo.py                 # quick (default): shipped model, ~30 s
   .venv/Scripts/python run_demo.py --mode full     # fine-tune live first, ~3 min on a laptop CPU
   ```
   (`.venv/bin/python` on macOS / Linux.)

   - **quick** loads the fine-tuned model shipped in `models/cuau_ft.model`
     and replays its recorded training curve on the dashboard — clearly
     labelled "recorded" — then runs the test and the search live. The shipped
     model was produced by `scripts/01_finetune.py` on this repository's data
     with the pinned versions; its curve and settings are in
     `models/training_log.json`.
   - **full** trains that model in front of you (overwriting the shipped one
     and its log), then runs the same test and search.

   Either way a live dashboard opens in the browser (keep it visible — that is
   what the audience watches), and a timing table and a plain-language summary
   are printed at the end:

   | Step | Script | What it does | Output |
   | --- | --- | --- | --- |
   | 1 | `scripts/01_finetune.py` (full mode only) | fine-tunes MACE-MP-0 (small) on `data/train.xyz` | `models/cuau_ft.model`, `models/training_log.json` |
   | 2 | `scripts/02_validate.py` | foundation vs fine-tuned model on `data/holdout.xyz` | `figures/parity.png`, `results/validation.json` |
   | 3 | `scripts/03_discover.py` | genetic algorithm over Cu–Au orderings, scored by MACE, graded against held-out reference | `figures/hull.png`, `results/discovery.json` |

3. Report back **for an audience of CS and EE students, not chemists**: the
   timing table, the step-2 table (error before vs after fine-tuning), the
   step-3 verdict (did the search's pick match the simulator's ground state,
   how many of the network's per-ratio picks were confirmed), and the "WHAT
   JUST HAPPENED, IN PLAIN TERMS" block at the end. Explain each number the
   way that block does — slow simulator vs fast network, bit strings, fitness
   function, verify step — and open `figures/parity.png` and
   `figures/hull.png`. In quick mode, say plainly that the model was trained
   earlier and loaded.

Full-mode knobs: `--epochs N` (20 by default; 30 gets the held-out energy error
from 7.9 to ~4 meV/atom for another ~90 s on CPU) and `--lr`.

## What you should see

On the DFT labels, held out = 51 structures of 17 orderings the model never
saw (CPU and GPU give the same numbers; training is seeded):

| | foundation MACE-MP-0 | fine-tuned, 20 epochs | fine-tuned, 30 epochs |
| --- | --- | --- | --- |
| formation-energy MAE (meV/atom) | 10.7 | 7.9 | ~4 |
| force RMSE (meV/Å) | 112 | 26 | ~23 |

Step 3, in every configuration: the GA's champion is `00010001` — Cu₃Au in
the L1₂ ordering — and DFT confirms it is the ground state. MACE's predicted
hull has the same 5 vertices as the DFT hull, and the verify step confirms all
three mixed-composition vertices (Cu₃Au, CuAu L1₀, CuAu₃) with 3 DFT calls
instead of 27. The foundation model is already decent on energies here because
it was itself trained on PBE; fine-tuning's big wins are the forces and the
few-meV ranking that the hull needs.

## The live dashboard

`run_demo.py` opens a page in your browser and streams the run into it: the
network's error falling epoch by epoch, the genetic algorithm's population
drawn as little copper/gold lattices evolving generation by generation, the
stability chart filling in as the network scores arrangements, and the
simulator's check marks landing in the verify step. **Put that browser window
on the projector**; the terminal is for the agent.

**No Python at all?** A recorded run ships with the repo: open
`dashboard/index.html` straight from the clone and it replays that run with
the same pacing — the learning curve, the test, the search, the verify step.
Nothing to install. Any `run_demo.py` run overwrites the recording with yours.

If no browser opens, the URL is printed (`live dashboard: http://127.0.0.1:…`).
After the run, opening `dashboard/index.html` as a file replays the whole
animation from the saved results — handy for showing it again, or if you have
to present without running. `--no-dashboard` turns it off; `--hold` keeps the
page served until you press Enter.


## How long it takes

Measured from a fresh clone on a Windows laptop with nothing installed
(CPU-only torch, download cache disabled):

| | time |
| --- | --- |
| `git clone` + `python -m venv` | ~10 s |
| `pip install -r requirements.txt` (torch is the bulk of it) | ~3 min |
| **quick mode:** load shipped model + test + search | **~30 s** |
| full mode: fine-tune 20 epochs + test + search | ~3.5 min |

So a cold machine reaches the result in about four minutes in quick mode;
slower CPUs or networks stretch the install and the full-mode training, not
the quick run. Install *before* the session if you can.

The first run downloads the MACE-MP-0 small foundation model (a few MB) into
`~/.cache/mace`; after that everything is offline.

## Prompt to give the agent

> Clone this repo and read README.md. Ask me one question: quick mode
> (recommended — uses the fine-tuned model shipped in the repo, about 30 s
> after install) or full mode (fine-tunes live first, a few minutes more)? If I
> don't answer, use quick. Then do exactly what the README says: create the
> venv, install the requirements, run `run_demo.py` with that mode. Explain
> what happened to a room of CS and EE students who know machine learning but
> no chemistry: the timing table, the before/after validation table, the
> step-3 verdict and verify-step table, and the plain-terms summary at the
> end. Open the two figures. Don't change any code.

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
> merged each with a look-alike; their 6 structures were labelled in a top-up
> run, so all 27 orderings have DFT labels.

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
