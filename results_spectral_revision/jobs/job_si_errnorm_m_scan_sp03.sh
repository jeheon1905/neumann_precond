#!/bin/bash -l
#SBATCH --job-name=si_errnorm_m_scan_sp03
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/si_errnorm_m_scan_sp03_%j.txt
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=12:00:00
# Lanczos convergence of ||Pi G_N Pi|| (the error-norm table): the largest Ritz
# value at Krylov dimension m = 10..100 against the exact norm from a dense SVD,
# on a Si cell small enough to build the operator densely.
set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
PY="/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python -u"
echo "host: $(hostname)  commit: $(git rev-parse HEAD)"
time $PY spectral_experiment.py \
    --filepath data/systems/Si_diamond.cif --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --precond neumann --outerorder 4 --error_cutoff -0.4 \
    --averaged_sum 1 --weight 0.5 --phase fixed --temperature 0.0 \
    --virtual_factor 1.2 --pp_type NNLP \
    --threads 8 --warmup 0 --diag_iter 200 --diag_tol 1e-5 --nblock 2 \
    --verbosity 1 --seed 2 \
    --probe_resid_below 1e-3 --exact_shift \
    --state_rules lowest middle homo \
    --skip_global --krylov_dims 20 \
    --error_norm 0 1 2 3 4 5 6 7 8 9 10 --error_norm_dim 30 \
    --error_norm_mdims 10 20 30 40 60 100 \
    --dense --dense_max_ngpts 8000 \
    --outdir results_spectral_revision/error_norm \
    --tag si_errnorm_m_scan_sp03
