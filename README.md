# Preconditioner development using Neumann expansion

This repository provides implementations and examples for developing preconditioners based on the Neumann series expansion, applied to iterative diagonalization problems in electronic structure calculations.

## Environment setting
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


## Usage example: ...
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


## Reproduce experiments
You can reproduce the experiments described in this repository using the provided configuration file.

```bash
python repeat_test.py --config config.yaml
```

