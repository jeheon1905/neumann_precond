from __future__ import annotations

import argparse
import datetime
import os
import sys
from math import ceil
from typing import Optional, Union

import numpy as np
import torch

from gospel import GOSPEL
from gospel.ParallelHelper import ParallelHelper as PH

# Prefer local preconditioner; fall back to gospel's if not present.
try:
    # from precondition import create_preconditioner  # local
    from precondition_new import create_preconditioner  # local
except Exception:
    print("Warning: using gospel's preconditioner instead of local neumann_precond")
    from gospel.Eigensolver.precondition import create_preconditioner  # fallback

from utils import block_all_print, get_git_commit, set_global_seed, make_atoms


# ------------------------------
# Builders
# ------------------------------
def resolve_upf_files(
    atoms,
    pp_type: str,
    upf_files_arg: Optional[Union[str, list[str]]] = None,
) -> list[str]:
    """
    Keep original behavior:
    - If upf_files is None: use hardcoded base path and suffix rule identical to original.
    - Else: accept string or list.
    """
    if upf_files_arg is not None:
        if isinstance(upf_files_arg, list):
            return upf_files_arg
        elif isinstance(upf_files_arg, str):
            return [upf_files_arg]
        else:
            raise ValueError("upf_files must be a string or a list of strings.")

    # Original default path & suffixes
    assert pp_type in ["SG15", "ONCV", "TM", "NNLP"]
    pp_path = f"./data/pseudopotentials/{args.pp_type}/"

    if pp_type == "SG15":
        pp_suffix = "_ONCV_PBE-1.2.upf"
    elif pp_type == "ONCV":
        pp_suffix = ".upf"
    elif pp_type == "TM":
        pp_suffix = ".pbe-n-nc.UPF"
    elif pp_type == "NNLP":
        pp_suffix = ".nnlp.UPF"
    else:
        raise NotImplementedError

    symbols = set(atoms.get_chemical_symbols())
    upf_files = [os.path.join(pp_path, f"{symbol}{pp_suffix}") for symbol in symbols]
    return upf_files


def build_eigensolver(args: argparse.Namespace) -> dict:
    """
    Mirror the original default: parallel_davidson with configurable maxiter/verbosity.
    Now honors --locking and --fill_block.
    """
    return {
        "type": "parallel_davidson",
        "maxiter": int(args.diag_iter),
        "locking": bool(args.locking),
        "fill_block": bool(args.fill_block),
        "verbosity": int(args.verbosity),
    }


def build_preconditioner(calc: GOSPEL, args: argparse.Namespace) -> None:
    """
    Preserve original branching & options for:
      - Neumann (with order/error_cutoff)
      - shift-and-invert + gapp
      - shift-and-invert + Neumann (inner)
      - else: generic precond with fp only
    """
    precond_type = args.precond
    if precond_type is None:
        calc.eigensolver.preconditioner = None
        return

    def _innerorder_value(v: Union[int, str]) -> Union[int, str]:
        # Keep 'dynamic' string as-is, else cast to int
        # if isinstance(v, str) and v == "dynamic":
        #     return v
        if v == "dynamic":
            return v
        else:
            return int(v)

    def _outerorder_value(v: Union[int, str]) -> Union[int, str]:
        # if isinstance(v, str) and v == "dynamic":
        #     return v
        if v == "dynamic":
            return v
        else:
            return int(v)

    innerorder = _innerorder_value(args.innerorder)
    outerorder = _outerorder_value(args.outerorder)
    # pcg = int(args.pcg_Neumann)  # kept for compatibility with potential future use
    error_cutoff = float(args.error_cutoff)

    if precond_type == "neumann":
        precond_options = {
            "precond_type": "neumann",
            "grid": calc.grid,
            "use_cuda": bool(args.use_cuda),
            "options": {
                "fp": "DP",
                "no_shift_thr": 10,
                "order": outerorder,
                "error_cutoff": error_cutoff,
                "verbosity": args.verbosity,
                # "MAX_ORDER": 20,
            },
        }
    elif precond_type == "shift-and-invert" and args.inner == "gapp":
        precond_options = {
            "precond_type": "shift-and-invert",
            "grid": calc.grid,
            "use_cuda": bool(args.use_cuda),
            "options": {
                "fp": "DP",
                "no_shift_thr": 10,
                "max_iter": int(args.pcg_iter),
                "verbosityLevel": args.verbosity,
                "inner_precond": "gapp",
                "correction_scale": 0.1,
            },
        }
    elif precond_type == "shift-and-invert" and args.inner == "neumann":
        precond_options = {
            "precond_type": "shift-and-invert",
            "grid": calc.grid,
            "use_cuda": bool(args.use_cuda),
            "options": {
                "fp": "DP",
                "no_shift_thr": 10,
                "max_iter": int(args.pcg_iter),
                "verbosityLevel": args.verbosity,
                "inner_precond": "neumann",
                "correction_scale": 0.1,
                "options": {
                    "order": innerorder,
                    "verbosity": args.verbosity,
                    "correction_scale": 0.0,  # to avoid double shift
                },
            },
        }
    else:
        precond_options = {
            "precond_type": precond_type,
            "grid": calc.grid,
            "use_cuda": bool(args.use_cuda),
            "options": {
                "fp": "DP",
            },
        }

    calc.eigensolver.preconditioner = create_preconditioner(**precond_options)
    print("preconditioner:\n", calc.eigensolver.preconditioner)


def compute_nbands(atoms, upf_files, args):
    from gospel.Pseudopotential.UPF import UPF

    upf = UPF(upf_files)
    nelec = 0
    for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
        nelec += upf[i_sym].zval
    nbands = int(ceil(nelec / 2) * args.virtual_factor)
    return nbands


# ------------------------------
# Core
# ------------------------------
def run_once(args: argparse.Namespace) -> None:
    # --- System / atoms ---
    atoms = make_atoms(args.filepath, args.supercell, pbc=args.pbc, vacuum=3.0)
    print(atoms)

    # --- Pseudopotential files ---
    upf_files = resolve_upf_files(atoms, args.pp_type, args.upf_files)
    print(f"Debug: upf_files={upf_files}")

    # Set number of bands if not provided
    if args.nbands is not None:
        nbands = int(args.nbands)
    else:
        nbands = compute_nbands(atoms, upf_files, args)
    print(f"Number of bands: {nbands}")

    # --- Eigensolver / Calculator ---
    eigensolver = build_eigensolver(args)

    calc = GOSPEL(
        use_cuda=bool(args.use_cuda),
        use_dense_kinetic=False,
        grid={"spacing": float(args.spacing)},
        pp={
            "upf": upf_files,
            "use_dense_proj": True,
            "filtering": True,
        },
        print_energies=True,
        xc={"type": "gga_x_pbe + gga_c_pbe"},
        precond_type=None,
        convergence={
            # SCF checks only energy tolerance; others are disabled (inf)
            "scf_maxiter": 100,
            "density_tol": np.inf,
            "orbital_energy_tol": np.inf,
            "energy_tol": float(args.scf_energy_tol),
            # Diagonalization tolerance (also used below for davidson tol)
            "diag_tol": float(args.diag_tol),
        },
        occupation={"smearing": "Fermi-Dirac", "temperature": float(args.temperature)},
        eigensolver=eigensolver,
        nbands=nbands,
    )
    atoms.calc = calc
    calc.initialize(atoms)

    # --- Preconditioner ---
    build_preconditioner(calc, args)

    # --- Phase: SCF or Fixed ---
    if args.phase == "scf":
        _ = atoms.get_potential_energy()
        if args.density_filename is not None:
            torch.save(
                calc.get_density(spin=slice(0, sys.maxsize)), args.density_filename
            )
            print(f"charge density file '{args.density_filename}' is saved.")
    elif args.phase == "fixed":
        # Fixed Hamiltonian diagonalization (preserve original)
        from gospel.Hamiltonian import Hamiltonian
        from gospel.Eigensolver.ParallelDavidson import davidson

        # Initialize density
        if args.density_filename is not None:
            density = torch.load(args.density_filename)
            density = density.reshape(1, -1).to(PH.get_device())
        else:
            print("Initializing the density...")
            density = calc.density.init_density()
        calc.density.set_density(density)

        calc.hamiltonian = Hamiltonian(
            calc.nspins,
            calc.nbands,
            calc.grid,
            calc.kpoint,
            calc.pp,
            calc.poisson_solver,
            calc.xc_functional,
            calc.eigensolver,
            use_dense_kinetic=calc.parameters["use_dense_kinetic"],
            # use_cuda=calc.parameters["use_cuda"],
            device=PH.get_device(),
        )
        calc.hamiltonian.update(calc.density)
        del density, calc.density, calc.kpoint, calc.poisson_solver, calc.xc_functional

        # Diagonalization
        calc.eigensolver._initialize_guess(calc.hamiltonian)
        results = davidson(
            A=calc.hamiltonian[0, 0],
            X=calc.eigensolver._starting_vector[0, 0],
            B=None,
            preconditioner=calc.eigensolver.preconditioner,
            tol=float(args.diag_tol),
            maxiter=int(args.diag_iter),
            nblock=int(args.nblock),
            locking=bool(args.locking),
            fill_block=bool(args.fill_block),
            verbosity=int(args.verbosity),
            retHistory=(args.retHistory is not None),
        )
        del calc

        # Save convergence history if requested
        if args.retHistory is not None:
            eigval, eigvec, eigHistory, resHistory = results
            print("Saving convergence history...")
            torch.save((eigHistory, resHistory), args.retHistory)
            print(f"{args.retHistory} is saved.")
        else:
            eigval, eigvec = results
    else:
        raise NotImplementedError("Only support 'scf' or 'fixed'.")


def main(args: argparse.Namespace) -> None:
    # Init parallel helper & seed
    # PH.init_from_env(args.use_cuda)
    set_global_seed(args.seed)

    # Warm-up (optional)
    if int(args.warmup):
        with block_all_print():
            run_once(args)
        print("Warm-up finished")

    # Actual timed run
    run_once(args)


# ------------------------------
# CLI
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preconditioner benchmark runner (refactored)"
    )
    # Inputs / Structure
    p.add_argument("--filepath", type=str, help="file path (cif or xyz)", required=True)

    # Preconditioner selection & knobs (keep original names)
    p.add_argument(
        "--precond",
        type=str,
        default=None,
        help="preconditioner type",
        choices=["neumann", "shift-and-invert", "gapp", None],
    )
    p.add_argument(
        "--inner", type=str, default=None, help="inner preconditioner in ISI"
    )
    p.add_argument(
        "--innerorder", type=str, default="0", help="Neumann inner order or 'dynamic'"
    )
    p.add_argument(
        "--outerorder", type=str, default="0", help="Neumann outer order or 'dynamic'"
    )
    p.add_argument(
        "--pcg_iter",
        type=str,
        default="5",
        help="pcg iterations within ISI (default 5)",
    )
    p.add_argument(
        "--error_cutoff", type=str, default="-0.4", help="ensure lower error"
    )

    # Discretization / bands
    p.add_argument("--spacing", type=float, default=0.2, help="grid spacing (Angstrom)")
    p.add_argument(
        "--nbands", type=int, default=None, help="number of orbitals to calculate"
    )
    p.add_argument(
        "--supercell", type=int, nargs=3, default=[1, 1, 1], help="supercell for PBC"
    )
    p.add_argument(
        "--pbc",
        type=int,
        nargs=3,
        help="periodic boundary condition of each axis. e.g., --pbc 0 0 1",
        required=True,
    )

    # Run mode
    p.add_argument(
        "--phase",
        type=str,
        default="fixed",
        choices=["scf", "fixed"],
        help="calculation phase",
    )
    p.add_argument(
        "--density_filename",
        type=str,
        default=None,
        help="charge density filename to save (or for initialization)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1160.45,
        help="temperature for smearing (default: 1160.45 K = 0.1 eV)",
    )
    p.add_argument(
        "--scf_energy_tol",
        type=float,
        default=1e-4,
        help="SCF energy tolerance (Hartree/electron); only energy is checked",
    )
    p.add_argument(
        "--virtual_factor",
        type=float,
        default=1.2,
        help="when nbands is not set, use this factor * occupied bands (default 1.2)",
    )

    # Pseudopotential
    p.add_argument(
        "--upf_files", type=str, nargs="+", default=None, help="explicit UPF file(s)"
    )
    p.add_argument(
        "--pp_type", type=str, default="SG15", choices=["SG15", "ONCV", "TM", "NNLP"]
    )

    # Misc / system
    p.add_argument("--use_cuda", action="store_true", help="use GPU (CUDA)")
    p.add_argument(
        "--warmup", type=int, default=1, help="whether to warm up the GPU (0/1)"
    )

    # Davidson / diag control
    p.add_argument(
        "--diag_iter",
        type=int,
        default=1000,
        help="eigensolver maxiter (default: 1000)",
    )
    p.add_argument(
        "--diag_tol",
        type=float,
        default=1e-4,
        help="residual tolerance for diagonalization",
    )
    p.add_argument(
        "--nblock", type=int, default=2, help="number of blocks for Davidson"
    )
    p.add_argument(
        "--locking", action="store_true", help="use locking in Davidson/eigensolver"
    )
    p.add_argument(
        "--fill_block",
        action="store_true",
        help="use fill_block in Davidson/eigensolver",
    )

    # Logging / output
    p.add_argument(
        "--verbosity", type=int, default=1, help="eigensolver/davidson verbosity level"
    )
    p.add_argument(
        "--seed", type=int, default=1, help="random seed (worker-safe with PH)"
    )
    p.add_argument(
        "--retHistory",
        type=str,
        default=None,
        help="filename to save (eigHistory, resHistory)",
    )
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    print(f"datetime: {datetime.datetime.now()}")
    print(f"GOSPEL git commit: {get_git_commit('gospel')}")
    print(f"neumann_precond git commit: {get_git_commit('neumann_precond')}")
    print("args=", args)

    # Threads / device info (lightweight)
    torch.set_num_threads(os.cpu_count() or 1)
    if args.use_cuda and torch.cuda.is_available():
        print(f"Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    main(args)
