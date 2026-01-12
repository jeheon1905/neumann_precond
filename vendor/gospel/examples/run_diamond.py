import numpy as np
from gospel import GOSPEL
from ase.build import bulk

pp_type = ["SG15", "ONCV", "TM"][-1]
if pp_type == "SG15":
    upf_files = ["/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/C_ONCV_PBE-1.2.upf"]
elif pp_type == "ONCV":
    upf_files = ["/appl/ACE-Molecule/DATA/ONCV/nc-sr-04_pbe_standard/C.upf"]
elif pp_type == "TM":
    #upf_files = ["/appl/ACE-Molecule/DATA/PSEUDOPOTENTIALS_NC/C.pbe-n-nc.UPF"]
    upf_files = ['data_gga/PSEUDOPOTENTIALS_NC/C.pbe-nc.UPF']
else:
    raise

atoms = bulk("C", "diamond", a=3.57, cubic=True)
# atoms = bulk("C", "diamond", a=4.0, cubic=True)
calc = GOSPEL(
    # force={"deriv_density": True, "deriv_comp":True},
    # centered=False,
    centered=True,
    use_cuda=True,
    hamiltonian={"use_dense_kinetic": True},
    # grid={"spacing": 0.1},
    grid={"spacing": 0.15},
    # grid={"gpts":(25,25,25)},
    # grid={"spacing": 0.2},
    pp={"upf": upf_files, "filtering":True},
    print_energies=True,
    xc={"type": "gga_x_pbe + gga_c_pbe"},
    # convergence={
    #     "density_tol": 1e-5,
    #     "orbital_energy_tol": 1e-5,
    #     "energy_tol": 1e-5,
    # },
    # symmetry=False,
    #kpts=(2, 2, 2, "gamma"),
    # kpts=(2, 1, 1, "gamma"),
    occupation={"smearing": "Fermi-Dirac", "temperature": 0.01},
)

atoms.calc = calc
forces = atoms.get_forces()
print(f"forces = {forces}")
