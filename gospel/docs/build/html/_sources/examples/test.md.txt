# Run examples

-------------------------------------------------------------

## Optimization of H2 molecule

```py
from ase.build import molecule
from gospel import GOSPEL

## Make Atoms object
atoms = molecule("H2")
atoms.center(vacuum=4.0)

## Make calculator
calc = GOSPEL(
    grid={"spacing": 0.2},  # Angstrom
    pp={"upf": [./H.UPF]},  # UPF pseudopotential file list
    xc={"type": "gga_x_pbe + gga_c_pbe"},
    occupation={
        "smearing": "Fermi-Dirac",
        "temperature": 1160.4518,  # Kelvin
    },
)
atoms.calc = calc

## Optimize atoms using BFGS method
from ase.optimize import BFGS
BFGS(atoms).run(fmax=0.01)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

-------------------------------------------------------------

## k-point calculation of silicon

```py
from ase.build import bulk
from gospel import GOSPEL

## Make Atoms object
atoms = bulk("Si", "diamond", a=5.43, cubic=True)

## Make calculator
calc = GOSPEL(
    grid={"spacing": 0.2},  # Angstrom
    kpts=(3, 3, 3),  # Monkhorst-Pack k-point grid
    symmetry=True,
    pp={
        "upf": [./Si.UPF],  # UPF pseudopotential file list
        "use_comp": True,
        "filtering": True,
    },
    xc={"type": "gga_x_pbe + gga_c_pbe"},
    occupation={
        "smearing": "Fermi-Dirac",
        "temperature": 1160.4518,  # Kelvin
    },
)
atoms.calc = calc

energy = atoms.get_potential_energy()  # calculate energy
```

-------------------------------------------------------------
