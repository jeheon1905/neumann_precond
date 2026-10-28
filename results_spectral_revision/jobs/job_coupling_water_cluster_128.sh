#!/bin/bash -l
#SBATCH --job-name=cp_water_cluster_128
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/coupling_water_cluster_128_%j.txt
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --time=24:00:00
# Same snapshot as job_err_*.sh.  Damped scheme only: norms of the implemented
# Pi G_N(E) Pi, of the deflated form G_N(Pi E Pi) Pi, and of their difference
# (the coupling through I - Pi that the deflated analysis neglects).
set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
PY="/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python -u"
echo "host: $(hostname)  commit: $(git rev-parse HEAD)"
time $PY spectral_experiment.py \
    --filepath data/systems/water_cluster_128.xyz --spacing 0.2 --supercell 1 1 1 --pbc 0 0 0 \
    --precond neumann --outerorder 4 --error_cutoff -0.4 \
    --averaged_sum 1 --weight 0.5 --phase fixed --temperature 0.0 \
    --scf_energy_tol inf --scf_density_tol 1e-5 --scf_mixing potential \
    --virtual_factor 1.2 --pp_type TM --filtering \
    --threads 8 --warmup 0 --diag_iter 1000 --diag_tol 1e-5 --nblock 2 \
    --verbosity 1 --seed 2 \
    --probe_resid_below 1e-3 --exact_shift \
    --state_rules lowest middle homo \
    --skip_global --krylov_dims 40 \
    --coupling_norm 0 1 2 3 4 5 6 7 8 9 10 --coupling_dim 40 \
    --outdir results_spectral_revision/coupling \
    --tag water_cluster_128_coupling
