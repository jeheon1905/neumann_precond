"""Build the implemented approximate inverse B = GAPP/(4 pi) densely and check it is SPD.

Both GAPP branches are covered: an isolated cell (pbc 0 0 0) and a periodic one
(pbc 1 1 1).  Under periodic boundary conditions the constant vector is a null
vector of the Laplacian; B is a sum of Gaussian convolutions and is never asked
to invert it, so its smallest eigenvalue, and B applied to the constant vector,
are reported explicitly.

Usage: python results_spectral_revision/check_gapp_spd.py OUT.json
"""
import json
import sys

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

sys.path.insert(0, ".")
from gospel.Grid import Grid  # noqa: E402
from gospel.ParallelHelper import ParallelHelper as PH  # noqa: E402
from precondition import create_preconditioner  # noqa: E402

torch.set_default_dtype(torch.float64)
PH.init_from_env(False)
cases = []
for label, atoms, pbc in (
    ("isolated H2O, 6 A box", Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
                                    cell=[6.0, 6.0, 6.0], pbc=False), (0, 0, 0)),
    ("periodic Si diamond", bulk("Si", "diamond", a=5.43, cubic=True), (1, 1, 1)),
):
    atoms = atoms.copy()
    atoms.pbc = [bool(x) for x in pbc]
    if not any(pbc):
        atoms.center()
    grid = Grid(atoms, spacing=0.6)
    gapp = create_preconditioner("gapp", grid, use_cuda=False)
    n = grid.ngpts
    B = gapp(torch.eye(n)) * (0.25 / np.pi)   # same scaling as PreNeumann.apply_P
    B = B.cpu().numpy()
    asym = float(np.abs(B - B.T).max() / np.abs(B).max())
    ev = np.linalg.eigvalsh(0.5 * (B + B.T))
    one = np.ones(n) / np.sqrt(n)
    rayleigh_const = float(one @ B @ one)
    cases.append({"case": label, "pbc": list(pbc), "ngpts": int(n),
                  "relative_asymmetry": asym, "eig_min": float(ev.min()),
                  "eig_max": float(ev.max()), "constant_mode_rayleigh": rayleigh_const})
    print(cases[-1], flush=True)

with open(sys.argv[1], "w") as f:
    json.dump(cases, f, indent=1)
