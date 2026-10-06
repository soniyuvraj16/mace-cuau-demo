#!/bin/bash
# Cu-Au reference set: 180 static single points (energy + forces), one per
# array task, 12 cores each so several calcs share a node.
#
#   cd vasp && mkdir -p logs && sbatch submit_array.sh
#
# KPAR=4 x NCORE=3 in INCAR assumes exactly 12 ranks; change both together.
#SBATCH --job-name=cuau-ref
#SBATCH --array=1-180%30
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
module purge
module load intel/2022a vasp/6.3.2
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" calcs/list.txt)
cd "calcs/$DIR"
echo "=== $DIR  ($(sed -n 7p POSCAR | awk '{s=0;for(i=1;i<=NF;i++)s+=$i;print s}') atoms) ==="

if grep -qa "aborting loop because EDIFF is reached" OUTCAR 2>/dev/null; then
    echo "already converged, skipping"; exit 0
fi
[ -s POTCAR ] || { echo "missing POTCAR in $DIR"; exit 1; }

srun vasp_std > vasp.out
grep -a "energy(sigma->0)" OUTCAR | tail -1
touch DONE
