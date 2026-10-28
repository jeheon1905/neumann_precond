#!/bin/bash -l
#SBATCH --job-name=ab_0.9_N8
#SBATCH --output=/home/jhwoo1905/Projects/neumann_precond/results_paper/alpha_ablation/jobs/B12_a0.9_N8_%j.txt
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --time=12:00:00 --exclusive
# One (alpha, order) per job.  The single-mode factor predicts over-correction only (mu = +3.26), i.e. a normal run here
# (see summarize_ablation.py), which is why the wall limit is 12:00:00.
cd /home/jhwoo1905/Projects/neumann_precond
PY=/home/jhwoo1905/miniforge3/envs/neumann_precond/bin/python
echo "host: $(hostname)  commit: $(git rev-parse HEAD)  system: B12  alpha: 0.9  order: 8"
echo "########## B12 alpha=0.9 order=8 ##########"
T0=$SECONDS
$PY -u test.py --filepath data/systems/B12.sdf --spacing 0.2 --supercell 1 1 1 --pbc 0 0 0 \
  --phase fixed --pp_type TM --filtering --threads 8 --warmup 0 \
  --diag_iter 1000 --diag_tol 1e-5 --nblock 2 --verbosity 0 --seed 0 \
  --precond neumann --outerorder 8 --error_cutoff -0.4 \
  --averaged_sum 1 --weight 0.9 \
  --temperature 0.0 --scf_energy_tol inf --scf_density_tol 1e-5 \
  --scf_mixing potential --virtual_factor 1.2 2>&1 | grep -E "^Diag\. Iter\. |^davidson |^Preconditioning "
echo "########## B12 alpha=0.9 order=8 DONE $((SECONDS-T0)) s ##########"
