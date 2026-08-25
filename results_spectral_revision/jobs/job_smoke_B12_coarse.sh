#!/bin/bash -l
#SBATCH --job-name=smoke_B12
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/smoke_%j.txt
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
PY="/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python -u"
echo "host: $(hostname)"
# SMOKE TEST ONLY: coarse spacing 0.45, not a manuscript-consistent calculation
time $PY spectral_experiment.py \
    --filepath data/systems/B12.sdf \
    --precond neumann --outerorder 4 --error_cutoff -0.4 \
    --averaged_sum 1 --weight 0.5 \
    --spacing 0.45 --supercell 1 1 1 --pbc 0 0 0 \
    --phase fixed --temperature 0.0 \
    --scf_energy_tol inf --scf_density_tol 1e-5 --scf_mixing potential \
    --virtual_factor 1.2 --pp_type TM --filtering \
    --threads 8 --warmup 0 \
    --diag_iter 1000 --diag_tol 1e-5 --nblock 2 \
    --verbosity 1 --seed 2 \
    --probe_call 3 --state_rule slowest \
    --krylov_dims 20 40 --arnoldi_seeds 0 \
    --outdir results_spectral_revision/smoke --tag B12_coarse_smoke
