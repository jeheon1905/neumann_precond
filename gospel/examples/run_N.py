import numpy as np
from ase import Atoms
from gospel import GOSPEL

atoms = Atoms('N', positions=[[0,0,0]], cell=[6,6,6])
calc = GOSPEL(
              centered=True,
              spin='polarized',
              nbands=11,
              grid={'spacing':0.25},
              pp={'upf':["data_gga/PSEUDOPOTENTIALS_NC/N.pbe-nc.UPF"],
                  #'use_comp':False},
                  'use_comp':True},
              xc={'type':'gga_x_pbe + gga_c_pbe'},
              convergence={'density_tol':1e-5,
                           'orbital_energy_tol':1e-5},
              occupation={'fix_magmom':True,
                          'magmom':3,
                          'smearing':'Fermi-Dirac',
                          'temperature':0.0},
                          #'temperature':1E-9},
              eigensolver={'type':'parallel_davidson','maxiter':1000},
              #iter_type='lbfgs'
              )

atoms.calc = calc
energy = atoms.get_potential_energy()
