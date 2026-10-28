#!/bin/bash -l
#SBATCH --job-name=gapp_spd
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/gapp_spd_%j.txt
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=0:30:00
# Dense check that B = GAPP/(4 pi) is SPD on isolated and periodic grids.
set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
echo "host: $(hostname)  commit: $(git rev-parse HEAD)"
OMP_NUM_THREADS=4 /home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python -u \
    results_spectral_revision/check_gapp_spd.py results_spectral_revision/validation/gapp_spd.json
