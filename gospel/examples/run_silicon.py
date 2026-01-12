#import numpy as np
#from ase import Atoms
#from gospel import GOSPEL
#from ase.build import bulk
#
#pp_type = ["SG15", "ONCV", "TM"][2]
#if pp_type == "SG15":
#    upf_files = ["/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.2.upf"]
#elif pp_type == "ONCV":
#    upf_files = ["/appl/ACE-Molecule/DATA/ONCV/nc-sr-04_pbe_standard/Si.upf"]
#elif pp_type == "TM":
#    upf_files = ["/appl/ACE-Molecule/DATA/PSEUDOPOTENTIALS_NC/Si.pbe-n-nc.UPF"]
#    # ["../tests/DATA/Si.pbe-n-nc.UPF"]
#else:
#    raise
#
#atoms = bulk("Si", "diamond", a=5.43, cubic=True)
#calc = GOSPEL(
#    grid={"spacing": 0.25},
#    pp={"upf": upf_files},
#    print_energies=True,
#    xc={"type": "gga_x_pbe + gga_c_pbe"},
#    convergence={"density_tol": 1e-5, "orbital_energy_tol": 1e-5},
#    kpts=(2, 2, 2, "gamma"),
#    occupation={"smearing": "Fermi-Dirac", "temperature": 0.01},
#)
#
#atoms.calc = calc
#energy = atoms.get_potential_energy()
#
#
# run_silicon.py
import numpy as np
from ase import Atoms
import gospel
from gospel import GOSPEL
from ase.build import bulk
import torch
torch.manual_seed(1)
upf_files = [ f'{gospel.__file__[:-11]}../examples/data_gga/PSEUDOPOTENTIALS_NC/Si.pbe-n-nc.UPF']

#pp_type = ["SG15", "ONCV", "TM"][2]
#if pp_type == "SG15":
#    upf_files = ["/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.2.upf"]
#elif pp_type == "ONCV":
#    upf_files = ["/appl/ACE-Molecule/DATA/ONCV/nc-sr-04_pbe_standard/Si.upf"]
#elif pp_type == "TM":
#    upf_files = ["/appl/ACE-Molecule/DATA/PSEUDOPOTENTIALS_NC/Si.pbe-n-nc.UPF"]
#    # ["../tests/DATA/Si.pbe-n-nc.UPF"]
#else:
#    raise

atoms = bulk("Si", "diamond", a=5.43, cubic=True)

################ Supercell ###################
from ase.build import make_supercell
prim = np.diag([1, 1, 1])
atoms = make_supercell(atoms, prim)
print(atoms)
# exit()
##############################################

##############################################
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--spacing',  type=float, default=0.2, help='Spacing')
parser.add_argument('--num_gapp_precond',  type=int, default=10 , help='The number of gapp preconditioning')
parser.add_argument('--num_inverse_precond',  type=int, default=0 , help='The number of shift-and-inverse preconditioning')

args = parser.parse_args()
max_iter = args.num_inverse_precond+args.num_gapp_precond
if args.num_inverse_precond==0:
    precond_type ='gapp'
else:
    precond_type = [('gapp', args.num_gapp_precond),('inverse', args.num_inverse_precond)]
##############################################

solver_type = ["parallel_davidson", "lobpcg", "davidson"][0]

if solver_type == "parallel_davidson":
    eigensolver={
        "type": "parallel_davidson",
        "maxiter": max_iter,
        "locking": False,
        # "locking": True,
        'fill_block':False,
        'verbosity': 1,
        # 'convg_tol': 1e-4,
    }
elif solver_type == "lobpcg":
    eigensolver={
        "type": "lobpcg",
        "maxiter": max_iter,
        # 'convg_tol':  1e-5,
    }
elif solver_type == "davidson":
    eigensolver={
        # "type": "lobpcg",
        "type": "davidson",
        "maxiter": max_iter,
        # 'convg_tol':  1e-5,
        "locking": False,
        "fill_block": False,
    }
else:
    raise NotImplementedError

calc = GOSPEL(
    use_cuda = torch.cuda.is_available(),
    hamiltonian={"use_dense_kinetic": True},
    precond_type=precond_type,
    eigensolver=eigensolver,
    # grid={"spacing": 0.25},
    grid={"spacing": args.spacing},
    pp={"upf": upf_files, "use_dense_proj": True},
    #print_energies=True,
    xc={"type": "gga_x_pbe + gga_c_pbe"},
    convergence={"density_tol": 1e-5, "orbital_energy_tol": 1e-5, "scf_maxiter": 100,},
    kpts=(2, 2, 2, "gamma"),
    occupation={"smearing": "Fermi-Dirac", "temperature": 0.01},
)

atoms.calc = calc
energy = atoms.get_potential_energy()# run_silicon.py
