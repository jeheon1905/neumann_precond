#!/bin/bash -l
#SBATCH --job-name=rhomid_C60_4
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/rho_mid_C60_4_%j.txt
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=24:00:00
set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
PY="/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python -u"
echo "host: $(hostname)  commit: $(git rev-parse HEAD)"
# Same invocation as the archived *_rho jobs, with state_rules restricted to the
# middle state -- the only one whose bound Table S6 is missing.
time $PY spectral_experiment.py \
    --filepath data/systems/C60_4.xyz --supercell 1 1 1 --pbc 0 0 0 --spacing 0.2 \
    --precond neumann --outerorder 4 --error_cutoff -0.4 \
    --averaged_sum 1 --weight 0.5 \
    --phase fixed --temperature 0.0 \
    --scf_energy_tol inf --scf_density_tol 1e-5 --scf_mixing potential \
    --virtual_factor 1.2 --pp_type TM --filtering \
    --threads 8 --warmup 0 \
    --diag_iter 1000 --diag_tol 1e-5 --nblock 2 \
    --verbosity 1 --seed 2 \
    --probe_call 13 --state_rules middle --skip_global --krylov_dims 40 \
    --rho_bound 1 2 3 5 8 12 20 --rho_bound_dim 40 \
    --outdir results_spectral_revision/svd_bounds \
    --tag C60_4_rho_middle
