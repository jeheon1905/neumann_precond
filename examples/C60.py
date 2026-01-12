from ase.build import molecule
from ase import Atoms
from gospel import GOSPEL
import numpy as np
import sys
import argparse 

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--spacing',  type=float, default=0.16 , help='Spacing')
parser.add_argument('--num_c60',  type=int, default=1 , help='The number of C60')
parser.add_argument('--num_gapp_precond',  type=int, default=0, help='The number of gapp preconditioning')
parser.add_argument('--num_inverse_precond',  type=int, default=2, help='The number of shift-and-inverse preconditioning')
#parser.add_argument('--precond_type',   type=str, default='gapp' , help='Method')

args = parser.parse_args()

max_iter = args.num_inverse_precond+args.num_gapp_precond

if args.num_inverse_precond==0:
    precond_type ='gapp'
elif args.num_gapp_precond==0:
    precond_type ='inverse'
else:
    precond_type = [('gapp', {}, args.num_gapp_precond),('inverse', {}, args.num_inverse_precond)]

upf = ["../tests/DATA/C_ONCV_PBE-1.2.upf"]

gap = 4.0

list_atoms=[]
for i in range(args.num_c60):
    atoms_ = molecule("C60")
    atoms_.set_positions(atoms_.get_positions() + np.array([[gap*(2*i+2), 2*gap, 2*gap]] ))
    list_atoms.append(atoms_)
atoms = sum(list_atoms, Atoms())
atoms.set_cell([(2*args.num_c60+2)*gap, 4*gap, 4*gap])
#for atom in atoms:
#    print(atom.symbol,"\t", atom.position[0], "\t", atom.position[1], "\t",atom.position[2] )
#print(np.max(atoms.get_positions(), axis=0))
#print(np.min(atoms.get_positions(), axis=0))
#print(atoms.get_cell())
##from ase.visualize import view
##view(atoms)
#exit(-1)
#    use_dense_kinetic=True,
calc = GOSPEL(
    print_energies=True,
    use_cuda=True,
    hamiltonian={"use_dense_kinetic": True},
    #eigensolver={"type": "davidson",         "maxiter": max_iter, "locking": False,'fill_block':False, 'verbosity': 1, 'convg_tol':  1e-5},
    eigensolver={"type": "parallel_davidson",         "maxiter": max_iter, "locking": True,'fill_block':False, 'verbosity': 1, 'convg_tol':  1e-5},
    precond_type=precond_type,
    grid={"spacing": args.spacing},
    pp={"upf": upf, "filtering": True,"use_dense_proj":True},
    xc={"type": "gga_x_pbe + gga_c_pbe"},
    convergence={
        "scf_maxiter": 250,
    },
)
atoms.calc = calc
atoms.get_potential_energy()


#calc = GOSPEL(
#    use_cuda=True,
#    use_dense_kinetic= True,
#    #eigensolver={"type": "ssrr",         "maxiter": 3, },
#    #eigensolver={"type": "parallel_davidson2",         "maxiter": 25, "locking": False,'fill_block':False },
#    eigensolver={"type": "parallel_davidson2",         "maxiter": 10, "locking": False,'fill_block':False, 'verbosity': 1},
#    #eigensolver={"type": "davidson",         "maxiter": 25, "locking": True,'fill_block':True , 'verbosity': 1},
#    #eigensolver={"type": "parallel_davidson", "maxiter": 500, "locking": False,'fill_block':False },
#    # eigensolver={"type": "lobpcg", "maxiter": 30},
#    #precond_type='filter',
#    precond_type=[('gapp', 9),('inverse', 1)],
#    #precond_type='gapp',
#    #:rid={"spacing": 0.3},
#    grid={"spacing": 0.16},
#    pp={"upf": upf, "filtering": True,"use_dense_proj":True},
#    xc={"type": "gga_x_pbe + gga_c_pbe"},
#    convergence={
#        "scf_maxiter": 250 #250,
#    },
#)
#atoms.calc = calc
#atoms.get_potential_energy()



#import torch.autograd.profiler as profiler
#with profiler.profile(with_stack=True, profile_memory=True) as prof:
#
#    atoms.get_potential_energy()
#    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
