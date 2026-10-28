# Neumann preconditioner for real-space iterative diagonalization

This repository contains the implementation, input files and analysis scripts of

> H. Ji, J. Woo, and S. Choi, *Efficient Preconditioning for Iterative Diagonalization via
> Neumann Series Expansion of the Inverse of a Shifted Hamiltonian*.

The Neumann preconditioner (NP) approximates $(H - \tilde{\epsilon} I)^{-1}$ by a truncated
Neumann series built on the Gaussian-approximated Poisson preconditioner (GAPP); order 0 is
GAPP itself. Damped-NP averages the last two orders, which stabilizes the convergence over
the orders examined. Both are implemented in `precondition.py` on top of the real-space DFT
code GOSPEL (`vendor/gospel`).

## Repository layout

| Path | Contents |
|---|---|
| `precondition.py` | NP, Damped-NP, GAPP and ISI preconditioners |
| `test.py`, `repeat_test.py` | single run and parameter sweep drivers |
| `configs/paper/` | configurations that reproduce the benchmarks of the paper ([README](configs/paper/README.md)) |
| `configs/smoke/` | miniature configurations for a quick CPU check ([README](configs/smoke/README.md)) |
| `data/systems/` | structures of the benchmark systems |
| `Figures/` | per-iteration timing data and scripts for Figure S1 and Table S2 ([README](Figures/README.md)) |
| `spectral_*.py` | spectral analysis of the preconditioned operator (Supporting Information, Section S) |
| `results_spectral_revision/` | data of the spectral analysis and the SLURM scripts that produced them |
| `results_paper/alpha_ablation/` | runs of the damping-weight sweep |
| `tests/` | unit tests of the spectral analysis tools |
| `vendor/gospel/` | vendored GOSPEL source with the modifications used here |

## Installation
We recommend using conda to manage dependencies.

```bash
# Create and activate a conda environment
conda create -n neumann_precond python=3.10 -y
conda activate neumann_precond

# Install PyTorch (CUDA 11.8 version)
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy==1.26.4     # Torch compatibility (numpy version pinned)
pip install ase               # Atomic Simulation Environment
pip install gitpython         # Git interface for Python
pip install "spglib>=1.16.1"  # Symmetry analysis library
pip install pyyaml

# Install neumann_precond (this repo)
python setup.py develop
```


## GOSPEL dependency

This project vendors a customized version of **GOSPEL** inside the repository
under `vendor/gospel` (using `git subtree`).

The vendored source is provided to ensure reproducibility and to apply
project-specific modifications.
**GOSPEL should be installed from this vendored source.**

Upstream repository:
- https://gitlab.com/jhwoo15/gospel.git

Base Git state of the vendored GOSPEL:
- **Branch**: `multi_gpu`
- **Commit**: `77621e1fdeff01df1d6adeadd557c056aad278b1`

Additional local modifications are applied on top of this commit
to support the Neumann preconditioner workflow.

### Install GOSPEL (from vendored source)
```bash
cd vendor/gospel
python setup.py develop
```

### pylibxc dependency
```bash
# Install pylibxc (for XC functionals)
git clone https://gitlab.com/libxc/libxc.git
cd libxc
git checkout 6.0.0  # Switch to 6.0.0 tag

# Patch CMakeLists.txt for newer CMake (required on some systems)
sed -i 's/^cmake_minimum_required(VERSION 3.1)/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt

conda install -c conda-forge cmake  # Run this if cmake is not installed
python setup.py develop  # or: pip install -e .
```


## Usage examples

Fixed-Hamiltonian diagonalization of a silicon cell with four preconditioners:

```bash
# 1. Neumann Preconditioner
python test.py \
    --filepath data/systems/Si_diamond.cif \
    --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --phase fixed --pp_type NNLP \
    --precond neumann --outerorder dynamic \
    --diag_iter 50 \
    --retHistory History.neumann.pt

# 2. GAPP
python test.py \
    --filepath data/systems/Si_diamond.cif \
    --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --phase fixed --pp_type NNLP \
    --precond gapp \
    --diag_iter 50 \
    --retHistory History.gapp.pt

# 3. Shift-and-invert Preconditioner
python test.py \
    --filepath data/systems/Si_diamond.cif \
    --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --phase fixed --pp_type NNLP \
    --precond shift-and-invert --inner gapp \
    --diag_iter 50 \
    --retHistory History.isi.pt

# 4. Shift-and-invert Preconditioner + Neumann
python test.py \
    --filepath data/systems/Si_diamond.cif \
    --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --phase fixed --pp_type NNLP \
    --precond shift-and-invert --inner neumann --innerorder dynamic \
    --diag_iter 50 \
    --retHistory History.isi_neumann.pt

# Plot the convergence history
for m in neumann gapp isi isi_neumann; do
    python plot_convg_history.py --filepath History.$m.pt --plot residual --convg_tol 1e-7 --num_eig 16 --save History.$m.residual.png
    python plot_convg_history.py --filepath History.$m.pt --plot eigval --convg_tol 1e-14 --num_eig 16 --save History.$m.eigval.png
done
```


## Reproducing the paper

Each configuration in `configs/paper/` is run with

```bash
python repeat_test.py --config configs/paper/<config>.yaml
```

| Result | Configuration or script |
|---|---|
| Figure 1, Table S1 (Block Davidson) | `config.fixed.timing.yaml` |
| Figure 2 | `config.fixed.history.yaml` |
| Figure S1, Table S2 | `config.fixed.periter.yaml`, then `extract_per_iteration_time.py` and the scripts in `Figures/` |
| Figure 3, Table S3 | `config.scf.nolocking.yaml`, `config.scf.locking.yaml` |
| Damping-weight sweep | `config.fixed.alpha_ablation.yaml`; `results_paper/alpha_ablation/summarize_ablation.py` |
| Table S10 | the SCF configurations with `scf_density_tol` set to `1.0e-6` or `1.0e-7` |
| Spectral analysis | `spectral_experiment.py` with the job scripts in `results_spectral_revision/jobs/` |

[`configs/paper/README.md`](configs/paper/README.md) explains the settings and the output
files. Timings were measured on a single NVIDIA GH200, except the B200 columns of Table S1
and Table S10, which were measured on a single NVIDIA B200.

## Spectral analysis

`spectral_experiment.py` attaches a probe, disabled by default, to the production
preconditioner and records the error propagation operator $E = I - BM$ at a chosen Davidson
iteration, where $B$ is the GAPP operator and $M = H - \tilde{\epsilon} I$. Spectra are
estimated by Arnoldi and operator norms by Lanczos; `results_spectral_revision/validation/`
and `results_spectral_revision/error_norm/` hold the dense checks of these estimates, and
`results_spectral_revision/check_gapp_spd.py` checks that $B$ is symmetric positive definite.
The unit tests of these tools run on a CPU:

```bash
PYTHONPATH=. python tests/test_spectral_tools.py
```
