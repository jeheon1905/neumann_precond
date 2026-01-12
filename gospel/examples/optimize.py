import numpy as np
from ase import Atoms
from gospel import GOSPEL
from ase.build import bulk

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
calc = GOSPEL(
    centered=False,
    # grid={"spacing": 0.15},
    grid={"spacing": 0.2},
    pp={
        # "upf": ["/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.2.upf"],
        # "upf": ["/appl/ACE-Molecule/DATA/ONCV/nc-sr-04_pbe_standard/Si.upf"],
        "upf": ["/appl/ACE-Molecule/DATA/PSEUDOPOTENTIALS_NC/Si.pbe-n-nc.UPF"],
        "filtering": True,
        # "use_dense_proj": True,
    },
    print_energies=True,
    xc={"type": "gga_x_pbe + gga_c_pbe"},
    convergence={
        "density_tol": 1e-5,
        "orbital_energy_tol": 1e-5,
        # "energy_tol": 1e-5,
        "scf_maxiter": 50,
    },
    occupation={
        "smearing": "Fermi-Dirac",
        "temperature": 0.01,
    },
)

atoms.calc = calc

from ase.optimize import BFGS

# relax = BFGS(atoms, trajectory=f"Si.traj", logfile=f"Si.txt")
relax = BFGS(atoms)
relax.run(fmax=0.01)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

ref_energy = -1043.715753
ref_forces = np.array(
    [
        [-1.41887325e-04, -1.99772680e-04, -3.63775153e-04],
        [-3.72350440e-04, -2.56899038e-04, -2.01922970e-04],
        [4.17693267e-05, -2.47543444e-05, 4.46712005e-05],
        [8.65244070e-05, -5.57457744e-05, -8.22596480e-05],
        [-6.78921394e-05, -3.88501167e-05, -3.63720909e-04],
        [-2.66198132e-04, -1.16874351e-05, -1.08385484e-04],
        [-3.15841717e-05, -1.44663774e-04, 6.34591667e-05],
        [7.22309560e-05, -1.83474565e-04, -3.71639677e-05],
    ],
)

assert abs(energy - ref_energy) < 1e-6, print(
    f"Energy error exceed the tolerance. energy={energy}, ref_energy={ref_energy}"
)
assert abs(forces - ref_forces).sum() < 1e-6, print(
    f"Force error exceed the tolerance. forces={forces}, ref_forces={ref_forces}"
)
