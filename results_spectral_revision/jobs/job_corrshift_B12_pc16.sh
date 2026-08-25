#!/bin/bash -l
#SBATCH --job-name=cs_B12_16
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/corrshift_B12_pc16_%j.txt
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=24:00:00 --exclusive
set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
PY="/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python -u"
echo "host: $(hostname)  date: $(date)  probe_call=16  CORRECTED SHIFT (no --exact_shift)"
echo "commit: $(git rev-parse HEAD)  branch: $(git branch --show-current)"
time $PY spectral_experiment.py \
    --filepath data/systems/B12.sdf --spacing 0.2 --supercell 1 1 1 --pbc 0 0 0 \
    --precond neumann --outerorder 4 --error_cutoff -0.4 \
    --averaged_sum 1 --weight 0.5 --phase fixed --temperature 0.0 \
    --scf_energy_tol inf --scf_density_tol 1e-5 --scf_mixing potential \
    --virtual_factor 1.2 --pp_type TM --filtering \
    --threads 8 --warmup 0 --diag_iter 1000 --diag_tol 1e-5 --nblock 2 \
    --verbosity 1 --seed 2 \
    --probe_call 16 \
    --state_rules homo \
    --skip_global --krylov_dims 40 \
    --svd_bounds 0 2 4 6 8 10 --svd_dim 80 --svd_undeflated \
    --error_norm 0 2 4 6 8 10 --error_norm_dim 30 \
    --deflated --deflated_dims 20 40 --cg_tol 1e-8 --cg_maxiter 200 --eta_max_order 10 \
    --outdir results_spectral_revision/corrected_shift/raw \
    --tag B12_corrshift_pc16
