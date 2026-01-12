import numpy as np
from ase import Atoms
from gospel import GOSPEL
from ase.build import bulk

atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
calc = GOSPEL(
              grid={'spacing':0.25},
              pp={'upf':["../tests/DATA/Si.pbe-n-nc.UPF"]},
              xc={'type':'gga_x_pbe + gga_c_pbe'},
              convergence={'density_tol':1e-5,
                           'orbital_energy_tol':1e-5,
                           'scf_maxiter':300},
#              kpts = (2,2,2, 'gamma'),
#              eigensolver={'type':'parallel_davidson',
              eigensolver={'type':'cg',
                            'maxiter':20,
                            'convg_tol':1e-6,
                            'relative_tol':1e-5,
                            'orthogonal_to_all':True,
                            'orthogonal_to_prev':True,
#                           'precond_type':'laplacian',
                           },
              occupation={'smearing':'Fermi-Dirac',
                          'temperature':0.01},
              )

atoms.calc = calc
energy = atoms.get_potential_energy()
