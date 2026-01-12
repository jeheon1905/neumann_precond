import numpy as np
import torch
from ase.io import read
from gospel import GOSPEL

# import os
# from initialize_data import initialize
# if (not os.path.exists('data')):
#    initialize()


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    ## Warm-up
    _ = torch.randn(10, 10).cuda()
    _ = _ @ _
    _ = _ @ _
    print("Finish Warm-up!")

ranks = (12, 12, 12)
atoms = read("MgO.cif")
calc = GOSPEL(
    spin="unpolarized",
    grid={"gpts": (29, 29, 29)},
    pp={"upf": ["../tests/DATA/O.pbe-n-nc.UPF", "../tests/DATA/Mg.pbe-n-nc.UPF"]},
    xc={"type": "lda_x + lda_c_pz"},
    occupation={"smearing": "Fermi-Dirac", "temperature": 0.01},
    eigensolver={
        "type": "tucker",
        "ignore_kbproj": True,
        "ranks": ranks,
        "check_residue": False,
    },
    use_cuda=USE_CUDA,
)
atoms.calc = calc
atoms.get_potential_energy()
