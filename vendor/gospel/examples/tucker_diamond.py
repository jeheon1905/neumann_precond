import numpy as np
import torch
from gospel import GOSPEL
from ase.build import bulk
from ase.units import Bohr

pp_type = ["SG15", "ONCV", "TM"][0]
if pp_type == "SG15":
    upf_files = ["/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/C_ONCV_PBE-1.2.upf"]
elif pp_type == "ONCV":
    upf_files = ["/appl/ACE-Molecule/DATA/ONCV/nc-sr-04_pbe_standard/C.upf"]
elif pp_type == "TM":
    upf_files = ["/appl/ACE-Molecule/DATA/PSEUDOPOTENTIALS_NC/C.pbe-n-nc.UPF"]
else:
    raise

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    import torch
    ## Warm-up
    _ = torch.randn(10, 10).cuda()
    _ = _ @ _
    _ = _ @ _
    print("Finish Warm-up!")

ranks = (12, 12, 12)
# ranks = (15, 15, 15)

atoms = bulk("C", "diamond", a=3.57, cubic=True)
calc = GOSPEL(
    grid={"spacing": 0.3 * Bohr},
    pp={"upf": upf_files, "filtering":True},
    print_energies=True,
    xc={"type": "gga_x_pbe + gga_c_pbe"},
    kpts=(4, 4, 4, "gamma"),
    eigensolver={
        "type": "tucker",
        "ignore_kbproj": True,
        "ranks": ranks,
        "check_residue": False,
    },
    occupation={"smearing": "Fermi-Dirac", "temperature": 0.01},
    use_cuda=USE_CUDA,
)

atoms.calc = calc
forces = atoms.get_forces()
print(f"forces = {forces}")
