import subprocess
import random
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Union
import importlib.util
import numpy as np
import torch

from ase import Atoms  # type: ignore
from ase.io import read 
from ase.build import bulk, molecule


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


def get_git_commit(package_name: str) -> str:
    """
    Retrieve the current Git commit hash for the package's directory if it belongs to a Git repository.

    Args:
        package_name: The name of the package to check.

    Returns:
        str: A message that includes the package name and the git commit hash,
             or an error message if the package is not found, not under git control,
             or if a git error occurs.
    """
    # Find the package specification using importlib
    spec = importlib.util.find_spec(package_name)
    if not spec or not spec.origin:
        return f"{package_name}: package not found"

    # Extract the package installation directory
    package_dir = os.path.dirname(spec.origin)

    # Traverse up the directory tree to search for a .git folder
    current = package_dir
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            try:
                # Retrieve the current git commit hash
                commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=current)
                    .decode()
                    .strip()
                )
                return f"{package_name}: {commit}"
            except subprocess.CalledProcessError:
                return f"{package_name}: no git info (error)"
        next_dir = os.path.dirname(current)
        if next_dir == current:  # Reached the root directory (cross-platform)
            break
        current = next_dir

    return f"{package_name}: not a git repository"


def warm_up(use_cuda=True):
    if use_cuda:
        device = PH.get_device(args.use_cuda)

        ## matmul warm-up
        A = torch.randn(1000, 1000).to(device)
        for i in range(5):
            A @ A
        print("Warm-up matmul")

        ## redistribute warm-up
        A = PH.split(A).to(device)
        _A = PH.redistribute(A, dim0=1, dim1=0)
        A = PH.redistribute(_A, dim0=0, dim1=1)
        print("Warm-up redistribute")
        del A, _A
    else:
        return

def set_global_seed(seed: int) -> None:
    """
    Set global random seed for torch, numpy, and random.

    Args:
        seed (int): Base seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False  # reproducibility
    torch.backends.cudnn.deterministic = True  # reproducibility
