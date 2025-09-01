# src/simple_structure_loader.py
"""Super simple structure loader for ASE.

Rules:
1) Look for "{material}.cif" or "{material}.sdf" in a given directory (in that order).
2) If not found, build from a tiny prototype map when *key matches exactly*.

Keep it minimal and explicit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from ase import Atoms  # type: ignore
from ase.io import read 
from ase.build import bulk, molecule, make_supercell  # type: ignore

# Exact-match prototype keys only (keep it simple)
#   key -> (symbol, crystalstructure, a, cubic)
DEFAULT_PROTOTYPES: Dict[str, Union[Tuple[str, str, float, bool], Tuple[str, str, float]]] = {
    "diamond": ("C", "diamond", 3.57, True),
    "silicon": ("Si", "diamond", 5.43, True),
    "Fe_fcc": ("Fe", "fcc", 3.65, True),  # adjust a to your workflow if needed
    "C60": ("molecule", "C60", 10.0)      # fullerene molecule with 10 Å vacuum 
}

def _ensure_nonzero_cell(atoms: Atoms, vacuum: float = 10.0) -> Atoms:
    """
    Ensure cell lengths are non-zero (needed by many calculators, ATP, GSH, Aspirin).
    """
    import numpy as np

    a, b, c = atoms.cell.lengths()
    if np.isclose([a, b, c], 0).any():
        atoms.center(vacuum=vacuum)  # cubic box around molecule
        atoms.pbc = (False, False, False)
    return atoms


def load_atoms(material: str, file_directory: str = "./") -> Atoms:
    """Return ASE Atoms from file or a minimal prototype map.

    Why this shape: deterministic, tiny surface area, no magic name variants.
    """
    base = Path(file_directory).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Directory not found: {base}")

    # 1) Try files: {material}.cif -> {material}.sdf (read failures fall through)
    last_err: Exception | None = None
    for ext in ("cif", "sdf"):
        p = base / f"{material}.{ext}"
        if p.is_file():
            try:
                atoms = read(p.as_posix())
                return _ensure_nonzero_cell(atoms)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue  # try next ext

    # 2) Try exact-match prototype key
    if material in DEFAULT_PROTOTYPES:
        proto = DEFAULT_PROTOTYPES[material]
        if proto[0] == "molecule":
            _, name, vacuum = proto
            atoms = molecule(name)
            return _ensure_nonzero_cell(atoms, vacuum=float(vacuum))
        else:
            sym, cs, a, cubic = proto  # type: ignore
            atoms = bulk(sym, cs, a=a, cubic=cubic)
            return _ensure_nonzero_cell(atoms)

    hint = f" (last read error: {last_err})" if last_err else ""
    raise ValueError(
        "No structure source found. Provide a file or use one of the exact keys: "
        + ", ".join(sorted(DEFAULT_PROTOTYPES.keys())) + hint
    )

import contextlib
import builtins
@contextlib.contextmanager
def block_all_print():
    original_print = builtins.print
    original_stdout = sys.stdout
    with open(os.devnull, "w") as fnull:
        builtins.print = lambda *args, **kwargs: None
        sys.stdout = fnull
        try:
            yield
        finally:
            builtins.print = original_print
            sys.stdout = original_stdout


# --- Optional tiny CLI ---
if __name__ == "__main__":  # pragma: no cover
    from ase.build import make_supercell
    import numpy as np
    import torch
    from gospel import GOSPEL
    import gospel.Hamiltonian
    from ase.build import bulk
    import argparse
    from ase import Atoms
    import datetime
    import os
    import sys
    from gospel.ParallelHelper import ParallelHelper as PH
    # from gospel.Eigensolver.precondition import create_preconditioner
    from precondition import create_preconditioner

    parser = argparse.ArgumentParser(description='calculation setting')
    parser.add_argument('--material',type=str, default = None, help = "calculation material")
    parser.add_argument('--dir',type=str, default = "./", help = "cif, sdf files diractory")
    parser.add_argument('--precond',type=str, default = None, help = "preconditioner type")
    parser.add_argument('--spacing',type=float, default = "0.2", help = "determine grid interval")
    parser.add_argument('--nbands', type= int, default = 20, help = "number of orbitals to calculate")
    parser.add_argument('--supercell' , type = int, nargs=3, default = [1,1,1], help = "setting supercell in PBC")
    parser.add_argument('--inner', type=str, default = None, help = "setting innerpreconditioner in ISI")
    parser.add_argument('--innerorder', type=str, default = 0, help = "When using innerprconditioner to Neumann, determine Neumann order ")
    parser.add_argument('--outerorder', type= str, default = 0, help = "Neumann preconditioner order")
    parser.add_argument('--pcg_Neumann', type=str, default = 2, help = "using Neumann preconditioner in ISI, determine pcg number")
    parser.add_argument('--error_cutoff', type=str, default = -0.4, help = "ensure lower error")
    parser.add_argument('--density_filename', type=str, default=None, help ="charge density save file name")
    parser.add_argument('--phase',type=str, default="fixed", choices=["scf","fixed"], help = "determine calculation types")
    parser.add_argument('--upf_files',type=str, default=None, help = "determine pseudopotential file diractly")
    parser.add_argument('--pp_type',type=str, default= "TM", choices=["SG15", "ONCV", "TM", "NNLP"], help = "pseudopotential type")
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="whether to use GPU (CUDA)",
    )
    # TODO: Implement warmup
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="whether to warm up the GPU (0=False, 1=True)",
    )

    args = parser.parse_args()
    torch.manual_seed(1)
    innerorder = args.innerorder
    outerorder = args.outerorder
    if args.innerorder != "dynamic":
        innerorder = int(args.innerorder)

    if args.outerorder != "dynamic":
        outerorder = int(args.outerorder)

    pcg = int(args.pcg_Neumann)
    error_cutoff = float(args.error_cutoff)

    solver_type = ["parallel_davidson","lobpcg","davidson"][0]

    if solver_type == "parallel_davidson":
        eigensolver={
                "type":"parallel_davidson",
    #            "maxiter": 10,
                "maxiter" : 1000,
    #            "maxiter": 2, # first diagonalization error
                "locking": False,
    #            "locking": True,
    #            'fill_block': True,
                'fill_block': False,
                'verbosity': 1,
        }
    else:
        raise NotImplementedError

    # Generate atoms object
    atoms = load_atoms(args.material, args.dir)
    scale_factors = args.supercell
    prim = np.diag(scale_factors)
    atoms = make_supercell(atoms, prim)
    print(atoms)

    # Set pseudopotential options
    if args.upf_files is None:
        assert args.pp_type in ["SG15", "ONCV", "TM", "NNLP"]
        # pp_path = f"./data_gga/PSEUDOPOTENTIALS_NC/"
        pp_path = f"/home/jeheon/data/sg15_oncv_upf_2020-02-06/"
        if args.pp_type == "SG15":
            pp_prefix = "_ONCV_PBE-1.2.upf"
        elif args.pp_type == "ONCV":
            pp_prefix = ".upf"
        elif args.pp_type == "TM":
            pp_prefix = ".pbe-n-nc.UPF"
        elif args.pp_type == "NNLP":
            pp_prefix = ".nnlp.UPF"
        else:
            raise NotImplementedError
        symbols = set(atoms.get_chemical_symbols())
        upf_files = [pp_path + symbol + pp_prefix for symbol in symbols]
    else:
        if isinstance(args.upf_files, list):
            upf_files = args.upf_files
        elif isinstance(args.upf_files, str):
            upf_files = [args.upf_files]
        else:
            raise ValueError("upf_files must be a string or a list of strings.")
    print(f"Debug: upf_files={upf_files}")

    calc = GOSPEL(
        #force={"deriv_density": True, "deriv_comp":True},
        use_cuda=args.use_cuda,
        use_dense_kinetic= False,
        grid={"spacing": args.spacing},
        #grid={"gpts":(25,25,25)},
        #grid={"spacing": 0.2},
        pp={"upf": upf_files, "use_dense_proj" : True,
            "filtering":True    # default is False
                    },
        print_energies=True,
        xc={"type": "gga_x_pbe + gga_c_pbe"},
        #precond_type = "poisson",
        precond_type = None,
        convergence={
             "density_tol": 1e-5,
             "orbital_energy_tol": 1e-5,
             "energy_tol": 1e-5,
             "scf_maxiter": 1,
             "diag_tol":1e-4,
             },
        #converge added #########################################
        #symmetry=False,
        #kpts=(2, 2, 2, "gamma"),
        #kpts=(2, 1, 1, "gamma"),
        occupation={"smearing": "Fermi-Dirac", "temperature": 0.01},
        eigensolver=eigensolver,
        nbands = args.nbands
    )
    ###test
    atoms.calc = calc
    calc.initialize(atoms)


    if args.precond == "Neumann":
        precond_options = {
                "precond_type": "Neumann",
                "grid": calc.grid,
                "use_cuda": args.use_cuda,
                "options": {
                    #"max_iter": 1000,
                    "fp": "DP",
                    #"verbosityLevel": 1,
                    #"locking": True,
                    "no_shift_thr": 10,
                    #"fill_block":True,
                    "order":f"{outerorder}",
                    "error_cutoff":error_cutoff,
                },
            }
        calc.eigensolver.preconditioner = create_preconditioner(**precond_options)
    elif args.precond == "shift-and-invert" and args.inner == "gapp":
        precond_options = {
                "precond_type": "shift-and-invert",
                "grid": calc.grid,
                "use_cuda": args.use_cuda,

                "options": {
                    "inner_precond": args.inner,
                    #"inner_precond": "gapp",
                    #"max_iter": 5,
                    "fp": "DP",
                    #"verbosityLevel": 1,
                    #"locking": True,
                    "no_shift_thr": 10,
                    #"fill_block":True,

                },
            }
        calc.eigensolver.preconditioner = create_preconditioner(**precond_options)
    elif args.precond == "shift-and-invert" and args.inner == "Neumann":
        precond_options = {
                "precond_type": "shift-and-invert",
                "grid": calc.grid,
                "use_cuda": args.use_cuda,
                "options": {
                    "inner_precond": args.inner,
                    #"inner_precond": "gapp",
                    #"max_iter": pcg,
                    "fp": "DP",
                    #"verbosityLevel": 1,
                    #"locking": True,
                    "no_shift_thr": 10,
                    #"fill_block":True,
                    "order" : innerorder,
                },
            }
        calc.eigensolver.preconditioner = create_preconditioner(**precond_options)
    else:
        print(args.precond)
        precond_options = {
                "precond_type": args.precond,
                "grid": calc.grid,
                "use_cuda": args.use_cuda,
                "options": {
                    #"inner_precond": args.inner,
                    #"max_iter": 1000,
                    "fp": "DP",
                    #"verbosityLevel": 1,
                    #"locking": True,
                    #"no_shift_thr": 10,
                    #"fill_block":True,
                    },
                }
        calc.eigensolver.preconditioner = create_preconditioner(**precond_options)


    if args.phase == "scf":
        energy = atoms.get_potential_energy()

        # Save the converged density
        if args.density_filename is not None:
            torch.save(
                calc.get_density(spin=slice(0, sys.maxsize)), args.density_filename
            )
            print(f"charge density file '{args.density_filename}' is saved.")
    elif args.phase == "fixed":
        # NOTE: Fixed Hamiltonian diagonalization
        from gospel.Hamiltonian import Hamiltonian
        from gospel.Eigensolver.ParallelDavidson import davidson

        # Initialize the electron density
        if args.density_filename is not None:
            device = PH.get_device(calc.parameters["use_cuda"])
            density = torch.load(args.density_filename)
            density = density.reshape(1, -1).to(device)
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
            use_cuda=calc.parameters["use_cuda"],
        )
        calc.hamiltonian.update(calc.density)
        calc.eigensolver._initialize_guess(calc.hamiltonian)
        #energy = atoms.get_potential_energy()
        del density, calc.density, calc.kpoint, calc.poisson_solver, calc.xc_functional

        # Diagonalization
        results = davidson(
            A=calc.hamiltonian[0, 0],
            X=calc.eigensolver._starting_vector[0, 0],
            B=None,
            preconditioner=calc.eigensolver.preconditioner,
            #tol=args.fixed_convg_tol,
            tol=1e-4,
            #maxiter=args.diag_iter,
            maxiter=1000,
            #nblock=args.nblock,
            nblock=2,
            #locking=args.locking,
            #fill_block=args.fill_block,
            locking=False,
            fill_block=False,
            verbosityLevel=1,
            #retHistory=(args.retHistory is not None),
            #skip_init_ortho=False,
            #timing=True,
            #use_MP=(args.fp == "MP"),
            #MP_scheme=args.MP_scheme,
            #debug_recalc_convg_history=args.recalc_convg_history,
        )
        del calc
