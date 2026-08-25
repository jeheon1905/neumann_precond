#!/bin/bash -l
#SBATCH --job-name=b12_calib
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/calib_%j.txt
#SBATCH --error=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/calib_err_%j.txt
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:00:00

set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
PY=/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python
echo "host: $(hostname)"
time $PY test.py \
    --filepath data/systems/B12.sdf \
    --precond neumann --outerorder 4 --error_cutoff -0.4 \
    --averaged_sum 1 --weight 0.5 \
    --spacing 0.2 --supercell 1 1 1 --pbc 0 0 0 \
    --phase fixed --temperature 0.0 \
    --scf_energy_tol inf --scf_density_tol 1e-5 --scf_mixing potential \
    --virtual_factor 1.2 --pp_type TM --filtering \
    --threads 20 --warmup 0 \
    --diag_iter 3 --diag_tol 1e-5 --nblock 2 \
    --verbosity 1 --seed 2
