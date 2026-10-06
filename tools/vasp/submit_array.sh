#!/bin/bash
# TEMPLATE - adapt the #SBATCH lines, module and launcher to your cluster.
# Submit from the directory that contains calcs/ (i.e. vasp/):  sbatch tools/vasp/submit_array.sh
#SBATCH --job-name=cuau-ref
#SBATCH --array=1-180%20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
module load vasp            # adapt

DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" calcs/list.txt)
cd "calcs/$DIR"
if [ -f POTCAR ] && grep -q "aborting loop because EDIFF is reached" OUTCAR 2>/dev/null; then
  echo "already converged: $DIR"; exit 0
fi
srun vasp_std > vasp.out    # adapt launcher / binary
