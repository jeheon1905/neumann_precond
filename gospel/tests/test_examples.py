from gospel.calculator import GOSPEL
from ase.build import bulk
from ase.units import Ha
from ase import Atoms

def test_false():
    assert 0

def test_simple():
    assert 1

def test_silicon():
    #atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
    atoms = Atoms('He', positions=[[0,0,0]], cell=[[4,0,0],[0,4,0],[0,0,4] ],pbc=False )
    calc = GOSPEL(
                  grid={'spacing':0.25},
                  pp= {'upf': ['../examples/data_gga/PSEUDOPOTENTIALS_NC/He.pbe-n-nc.UPF']},
                  #pp={'upf':["DATA/Si.pbe-n-nc.UPF"]},
                  xc={'type':'gga_x_pbe + gga_c_pbe'},
                  convergence={'density_tol':1e-5,
                               'orbital_energy_tol':1e-5},
                  # kpts=(2,2,2, 'gamma'),
                  occupation={'smearing':'Fermi-Dirac',
                              'temperature':0.01},
                  use_cuda=True, 
                  ## Make eigensolver option
                  eigensolver = {
                      "type": "parallel_davidson",
                      "locking": False,
                      "fill_block": False,
                      'verbosity': 1 
                  },
                  )
    atoms.calc = calc
    energy = atoms.get_potential_energy().item()
    fermi_level = calc.fermi_level
    band_gap = calc.band_gap[0]
    assert abs(-0.8273621325194571*Ha - energy) < 1e-6
    assert abs(19.42917700731109 - fermi_level * Ha) < 1e-4
    assert abs( 4.7136737287064285 - band_gap ) < 1e-4
#    assert abs(-31.1834133 - energy) < 1e-6
#    assert abs(4.63415316 - fermi_level * Ha) < 1e-4
#    assert abs(0.544 - band_gap * Ha) < 1e-4
    return


test_silicon()
