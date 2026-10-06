# Handoff: run the Cu–Au reference DFT set on TAMU HPRC Grace

Written for the Claude Code session on Grace. The user (Yuvraj) planned this
with another Claude session on his laptop; you cannot see that conversation,
so everything you need is here. Treat the user's own words as the authority
if anything here conflicts with what you see on the cluster.

## 1. What this is for (two paragraphs of context)

Yuvraj is giving a class tutorial (ENGR 689, a course on agentic development)
on MACE, a machine-learned interatomic potential, and how it can act as the
fast "evaluate" step of an agentic materials-discovery loop in place of DFT.
The hands-on demo lives in a public GitHub repo (`mace-cuau-demo`): the
audience clones it and an agent runs it end to end in ~4 minutes — fine-tune
the MACE-MP-0 foundation model on a small set of DFT labels for Cu–Au alloys,
validate on held-out structures, then run a genetic algorithm over Cu/Au
orderings using MACE as the energy oracle and check its answer against DFT.

Your job is to produce the **real PBE labels** for exactly these 186
structures so they can replace the stand-in labels before the class. Nothing
about the structures may change; only their energies and forces are being
computed. (If 180 of them are already done from an earlier bundle, see
TOPUP.md: only directories `180_…`–`185_…` are new.)

## 2. What the jobs are

186 **static single-point VASP calculations** (IBRION=-1, NSW=0): fixed
geometry in, total energy and per-atom forces out. No relaxation, no MD, no
cell optimisation. DFT is not searching for anything — it produces the labels
the ML model trains on and is graded against.

| group | dirs named | count | purpose |
| --- | --- | --- | --- |
| 4-atom FCC cells, 7-point volume scan (±5 %) for each of 5 orderings | `*_eos4_*` | 35 | energy–volume curve per composition |
| 4-atom cells, randomly rattled (σ 0.05 / 0.10 Å) at 3 volumes | `*_rattle4_*` | 60 | non-zero forces |
| 8-atom (2×1×1) cells, 10 orderings, ideal + 3 rattled | `*_train8_*` | 40 | ordering patterns a 4-atom cell cannot hold |
| 8-atom cells, the other 17 orderings, ideal + 2 rattled | `*_holdout8_*` | 51 | never trained on: validation + answer key |

The scientific question the labels answer: which arrangement of Cu and Au on
the FCC lattice is most stable at each composition (ordering ground states,
convex hull — textbook: Cu₃Au L1₂, CuAu L1₀). The demo shows MACE fine-tuned
on 135 labels ranks the other 51 well enough to find those ground states.

## 3. What is in the bundle

```
vasp/
  submit_array.sh            SLURM array, one calc per task, 12 ranks each
  calcs/list.txt             186 directory names, one per line (array index = line number)
  calcs/<name>/              POSCAR  KPOINTS  INCAR  POTCAR  meta.json
  tools/collect.py           turns finished OUTCARs into train.xyz / holdout.xyz
  tools/common.py            helper imported by collect.py
  data/structures.xyz        the 180 structures with their metadata (split, ordering, ...)
  HANDOFF_GRACE.md           this file
```

Settings (identical for every calc — do not vary them between calcs):
PBE PAW (`Cu` 22Jun2005, `Au` 04Oct2007), ENCUT 400, PREC Accurate,
ISMEAR 1 / SIGMA 0.1, EDIFF 1e-6, NELM 120, LREAL .FALSE., ADDGRID, ISPIN 1,
Γ-centred 8×8×8 (4-atom cells) / 4×8×8 (8-atom cells), no WAVECAR/CHGCAR.
`KPAR = 4` and `NCORE = 3` assume **exactly 12 MPI ranks**.

Cluster assumptions, copied from the user's own working job script:

```
module purge; module load intel/2022a vasp/6.3.2
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
srun vasp_std
```

(The user's script used `vasp_gam`; ours needs `vasp_std` because we have real
k-meshes. Both come from the same module.)

## 4. What you may and may not change

**Free to change** — anything about parallelism and scheduling: `--ntasks`,
`--mem`, `--time`, `--partition`/account, the array throttle `%30`, `KPAR`,
`NCORE`, how many calcs run per job. Keep `KPAR × NCORE = ranks per calc`
(or drop both and let VASP default).

**Do not change** — anything that changes the physics or breaks consistency
between calcs: POSCARs, KPOINTS meshes, ENCUT, PREC, ISMEAR/SIGMA, EDIFF,
POTCARs, ISPIN, LREAL. If one of these *must* change for everything (e.g. a
convergence problem), change it for **all 180** and say so in the summary.

Only if a specific calc will not converge with the shared settings: try
`ALGO = All` or `AMIX = 0.1; BMIX = 0.0001` for that one, and report it.

## 5. First thing to check: why only 4 jobs ran

The user submitted `submit_array.sh` (array `1-186%30`) and saw only 4 tasks
running, then "something wrong". Diagnose before resubmitting:

```
cd <wherever vasp/ was unpacked>
squeue -u $USER -r -h -o "%t" | sort | uniq -c            # R vs PD counts
squeue -u $USER -r -t PD -h -o "%R" | sort | uniq -c       # pending reasons
ls logs/ | head; tail -n 30 logs/$(ls -t logs | head -1)   # newest log
for d in $(head -5 calcs/list.txt); do echo "== $d"; tail -n 3 calcs/$d/vasp.out 2>/dev/null; done
```

Likely causes and fixes:

- **Pending reason `QOSMaxJobsPerUserLimit` / `AssocMaxJobsLimit`** — the
  account is capped at N concurrent jobs. 180 calcs at 4 at a time is far too
  slow. Use the whole-node packed script in §6 instead (few jobs, each running
  4 calcs concurrently and looping through the list).
- **Pending reason `Priority` / `Resources`** — nothing wrong; more will start.
- **Tasks fail immediately** (log ends with a module/srun/PMI error) — check
  `module avail vasp`, `module avail intel`; adapt the two `module load` names
  and whether `srun` needs `--mpi=pmi2`. The user's AIMD jobs run with exactly
  the lines above, so this is unlikely unless the environment differs by
  partition.
- **`vasp_std: command not found`** — the module may name the binary
  differently (`vasp_std`, `vasp`, `vasp.std`); `which vasp_std` after loading.
- **Each task killed for memory** — raise `--mem` (these calcs need well under
  10 GB; 24 GB is already generous).
- **Partition refuses `--ntasks=12` shared-node jobs** — switch to §6.
- **Convergence**: a healthy OUTCAR contains `aborting loop because EDIFF is
  reached` and a final `energy(sigma->0)`; a 4-atom calc should take minutes,
  an 8-atom one a bit longer, on 12 cores.

`submit_array.sh` skips any directory whose OUTCAR is already converged, so
resubmitting after fixes is safe.

## 6. Fallback: whole-node packed jobs

If the array is throttled or shared nodes are not allowed, use this instead
(one 48-core node per job, 4 calcs at a time, 12 ranks each; `KPAR 4 × NCORE 3`
in the INCARs stays valid). Save as `vasp/submit_packed.sh`, then
`sbatch submit_packed.sh 1 47`, `sbatch submit_packed.sh 48 94`,
`sbatch submit_packed.sh 95 140`, `sbatch submit_packed.sh 141 186`.

```bash
#!/bin/bash
#SBATCH --job-name=cuau-pack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pack_%j.out
set -uo pipefail
FIRST=${1:?first line of list.txt}; LAST=${2:?last line}
module purge; module load intel/2022a vasp/6.3.2
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
ROOT=$PWD
run_one() {
  d="$1"; cd "$ROOT/calcs/$d" || return
  if grep -qa "aborting loop because EDIFF is reached" OUTCAR 2>/dev/null; then echo "skip $d"; return; fi
  srun --exclusive -N1 -n12 vasp_std > vasp.out 2>&1
  echo "$d $(grep -a 'energy(sigma->0)' OUTCAR | tail -1)"
}
export -f run_one; export ROOT
sed -n "${FIRST},${LAST}p" calcs/list.txt | xargs -P 4 -I{} bash -c 'run_one {}'
```

If `srun --exclusive -n12` steps misbehave on this cluster, the simplest
robust alternative is `-P 1` with `srun -n48 vasp_std` and `KPAR=8, NCORE=6`
written into every INCAR (`sed -i 's/^KPAR.*/KPAR = 8/; s/^NCORE.*/NCORE = 6/' calcs/*/INCAR`).

## 7. When the runs are done: collect on Grace

Needs Python with `ase` and `numpy` (`module load` a Python, or
`pip install --user ase`). Then, from `vasp/`:

```
python tools/collect.py
```

It reads every `calcs/<name>/OUTCAR`, takes `energy(sigma->0)` and the final
forces, restores each structure's own site order (the POSCARs were written
Cu-first; `meta.json` holds the permutation), and writes

```
data/train.xyz     135 frames expected
data/holdout.xyz    51 frames expected
```

and lists every run that is missing or unconverged. Rerun the missing ones
until both counts are complete; the two pure 8-atom cells
(`*_train8_00000000_ideal`, `*_train8_11111111_ideal`) are essential because
every formation energy is measured against them.

## 8. What to hand back to the user

1. `data/train.xyz` and `data/holdout.xyz` (about 100 KB together) — these go
   into the demo repo's `data/` folder, replacing the stand-in files.
2. A short summary: how the jobs were run in the end (array or packed, ranks
   per calc), total wall time, any calc that needed different settings, and
   anything that did not converge.
3. Spot-check numbers worth quoting: for the 4-atom pure cells, the volume-scan
   minimum should sit near a = 3.63 Å (Cu) and 4.16 Å (Au); the formation
   energy of the ideal Cu₃Au cell (`*_eos4_0001_*` middle point) relative to
   the pure cells should be roughly −40 to −50 meV/atom with PBE.

Please do not spend effort on anything beyond producing these labels; the
tutorial is on a one-day deadline and the MACE side is already done.
