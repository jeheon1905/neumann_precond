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

# Install GOSPEL (local development mode)
git clone https://gitlab.com/jhwoo15/gospel.git
cd gospel
python setup.py develop

# Install pylibxc (for XC functionals)
git clone https://gitlab.com/libxc/libxc.git
cd libxc
git checkout 6.0.0  # Switch to 6.0.0 tag
conda install -c conda-forge cmake  # Run this if cmake is not installed
python setup.py develop  # or: pip install -e .

# If pylibxc import fails:
# You may need to add libxc.so* to your library path.
# Example: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libxc

# Install neumann_precond (this repo)
cd neumann_precond
python setup.py develop
```


## Usage example: ...
```bash
# 1. Neumann Preconditioner
python test.py \
    --filepath data/systems/Si_diamond.cif \
    --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --phase fixed --pp_type NNLP \
    --use_cuda --warmup 0 \
    --precond neumann --innerorder 3 --outerorder dynamic \
    --diag_iter 50 \
    --retHistory History.neumann.pt

# 2. GAPP
python test.py \
    --filepath data/systems/Si_diamond.cif \
    --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --phase fixed --pp_type NNLP \
    --use_cuda --warmup 0 \
    --precond gapp \
    --diag_iter 50 \
    --retHistory History.gapp.pt

# 3. Shift-and-invert Preconditioner
python test.py \
    --filepath data/systems/Si_diamond.cif \
    --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --phase fixed --pp_type NNLP \
    --use_cuda --warmup 0 \
    --precond shift-and-invert --inner gapp \
    --diag_iter 50 \
    --retHistory History.isi.pt

# Plot the convergence history
python plot_convg_history.py  --filepath History.neumann.pt --plot residual \
    --convg_tol 1e-7 --num_eig 16 --save History.neumann.residual.png
python plot_convg_history.py  --filepath History.neumann.pt --plot eigval \
    --convg_tol 1e-14 --num_eig 16 --save History.neumann.eigval.png

python plot_convg_history.py  --filepath History.gapp.pt --plot residual \
    --convg_tol 1e-7 --num_eig 16 --save History.gapp.residual.png
python plot_convg_history.py  --filepath History.gapp.pt --plot eigval \
    --convg_tol 1e-14 --num_eig 16 --save History.gapp.eigval.png

python plot_convg_history.py  --filepath History.isi.pt --plot residual \
    --convg_tol 1e-7 --num_eig 16 --save History.isi.residual.png
python plot_convg_history.py  --filepath History.isi.pt --plot eigval \
    --convg_tol 1e-14 --num_eig 16 --save History.isi.eigval.png
```
