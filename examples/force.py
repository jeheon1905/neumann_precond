"""Test dissociation of diatomic molecule
"""
import numpy as np
from ase.build import molecule
from gospel import GOSPEL

upf_dir = "/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/"

cell = [8, 8, 10]
# cell = [10, 10, 12]
spin = "unpolarized"
magmom = 0.0
elements = ["H"]

for element in elements:
    formula = f"{element}2"
    upf_files = [upf_dir + f"{element}_ONCV_PBE-1.2.upf"]

    atoms = molecule(formula)
    atoms.set_cell(cell)
    atoms.center()
    print(f"distance = {atoms.get_distance(0,1)}")
    calc = GOSPEL(
        spin=spin,
        grid={"spacing": 0.2},
        pp={
            "upf": upf_files,
            # 'use_comp': False,
            "use_comp": True,
            "filtering": True,
        },
        xc={"type": "gga_x_pbe + gga_c_pbe"},
        convergence={
            "density_tol": 1e-4,
            "orbital_energy_tol": 1e-6,
            "scf_maxiter": 100,
        },
        occupation={
            "smearing": "Fermi-Dirac",
            "temperature": 1e-9,
            "magmom": magmom,
        },
    )
    atoms.calc = calc
    atoms.get_forces()
    print("positions = \n", atoms.get_positions())
