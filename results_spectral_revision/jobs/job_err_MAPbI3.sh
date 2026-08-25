#!/bin/bash -l
#SBATCH --job-name=er_MAPbI3
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/err_MAPbI3_%j.txt
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=24:00:00
set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
PY="/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python -u"
echo "host: $(hostname)  commit: $(git rev-parse HEAD)"
time $PY spectral_experiment.py \
    --filepath data/systems/MAPbI3.cif --spacing 0.2 --supercell 2 2 2 --pbc 1 1 1 \
    --precond neumann --outerorder 4 --error_cutoff -0.4 \
    --averaged_sum 1 --weight 0.5 --phase fixed --temperature 0.0 \
    --scf_energy_tol inf --scf_density_tol 1e-5 --scf_mixing potential \
    --virtual_factor 1.2 --pp_type TM --filtering \
    --threads 8 --warmup 0 --diag_iter 1000 --diag_tol 1e-5 --nblock 2 \
    --verbosity 1 --seed 2 \
    --probe_resid_below 1e-3 --exact_shift \
    --state_rules lowest middle homo \
    --skip_global --krylov_dims 40 \
    --error_norm 0 1 2 3 4 5 6 7 8 9 10 --error_norm_dim 30 \
    --outdir results_spectral_revision/error_norm \
    --tag MAPbI3_errnorm
