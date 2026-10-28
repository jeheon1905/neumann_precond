#!/bin/bash -l
#SBATCH --job-name=st_C60_4
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/state_C60_4_%j.txt
#SBATCH --error=/home/jhwoo1905/Projects/neumann_precond/results_spectral_revision/jobs/state_C60_4_err_%j.txt
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -uo pipefail
cd /home/jhwoo1905/Projects/neumann_precond
PY="/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python -u"
echo "host: $(hostname)  date: $(date)"
echo "commit: $(git rev-parse HEAD)  branch: $(git branch --show-current)"

# State-dependence study (instruction 10.2: "If state dependence is substantial,
# add a small representative set such as lowest/middle/HOMO/slowest").
time $PY spectral_experiment.py \
    --filepath data/systems/C60_4.xyz \
    --precond neumann --outerorder 4 --error_cutoff -0.4 \
    --averaged_sum 1 --weight 0.5 \
    --spacing 0.2 --supercell 1 1 1 --pbc 0 0 0 \
    --phase fixed --temperature 0.0 \
    --scf_energy_tol inf --scf_density_tol 1e-5 --scf_mixing potential \
    --virtual_factor 1.2 --pp_type TM --filtering \
    --threads 8 --warmup 0 \
    --diag_iter 1000 --diag_tol 1e-5 --nblock 2 \
    --verbosity 1 --seed 2 \
    --probe_call 3 --state_rules lowest middle homo slowest \
    --krylov_dims 40 60 --arnoldi_seeds 0 \
    --eta_max_order 11 \
    --outdir results_spectral_revision/state_dependence/raw \
    --tag C60_4_states
