import numpy as np
import sys
from math import ceil
from datetime import datetime
from typing import Dict, Any
from ase.calculators.calculator import Calculator
from ase.units import Bohr, Ha
from ase.dft.bandgap import bandgap

from gospel.Grid import Grid
from gospel.util import Timer, set_global_seed
from gospel.Poisson import create_poisson
from gospel.Pseudopotential import create_Pseudopotential
from gospel.Occupation import Occupation
from gospel.Eigensolver.precondition import create_preconditioner
from gospel.Eigensolver import create_eigensolver
from gospel.XC import create_xc
from gospel.scf import SCF
from gospel.lbfgs import LBFGS
from gospel.Kpoint import Kpoint
from gospel.Hamiltonian import Hamiltonian
from gospel.Density import Density
from gospel.ParallelHelper import ParallelHelper as PH


class GOSPEL(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    # implemented_properties = ['energy', 'forces', 'stress', 'magmom', 'dipole']
    discard_results_on_any_change = True

    default_parameters: Dict[str, Any] = {
        "spin": "unpolarized",
        "nbands": None,  # default: (num of occ states) * 1.05
        "nelec": None,  # calculated in self.initialize()
        "verbose": 0,
        "print_energies": False,
        "output_file": None,
        "centered": False,
        "grid": {
            "FD_order": 3,
            "spacing": None,  # angstrom
            "gpts": None,
        },
        "hamiltonian": {
            "multi_dtype": None,
            "use_dense_kinetic": False,
        },
        "pp": {
            "upf": None,
            "NLCC": True,
            "use_comp": True,
            "filtering": False,
            "use_dense_proj": False,
            "num_near_cell": 1,
        },
        "kpts": (1, 1, 1, "gamma"),
        "kpts_and_weights": None,
        "group_sym": False,  # consider group symmetry to sample k-point
        "inv_sym": True,  # consider inversion symmetry to sample k-point
        "xc": {
            "type": "gga_x_pbe + gga_c_pbe",
            "weight": None,
        },
        "occupation": {
            "smearing": None,
            "temperature": 1160.4518,  # Kelvin
            "fix_magmom": True,
            "magmom": 0.0,
            "MP_order": 2,
        },
        "init_guess": {
            "density": "rhoatom",
            "orbital": None,
            "occ": None,
        },
        "force": {
            "deriv_density": False,
            "deriv_comp": True,
        },
        "convergence": {
            "scf_maxiter": 100,
            "energy_tol": 1e-6,  # Hartree / electron
            "density_tol": 1e-5,  # / electron
            "orbital_energy_tol": 1e-6,  # Hartree / electron
            "bands": "occupied",  # check convergence for occupied states (default)
        },
        "mixing": {
            "what": "density",
            "scheme": "pulay",
            "history": 3,
            "parameter": 0.2,
            "pulay_beta": 0.3,
        },
        "iter_type": "scf",
        "lbfgs": {
            "alpha": 1.0,
            "beta": 0.5,
            "sigma": 1e-6,  # 1e-4,
            "m": 5,  # 3,
        },
        "poisson": None,
        "eigensolver": None,
        "precond_type": "gapp",
        # "precond_type": "shift-and-invert",
        "use_cuda": False,
        "random_seed": 42,
    }

    def __init__(self, restart=None, label=None, **kwargs):
        self.__name__ = "GOSPEL"
        Calculator.__init__(self, restart=restart, label=label, **kwargs)

        ## Fill parameters with default_parameters
        for key, val in self.default_parameters.items():
            if isinstance(val, dict):
                self.default_parameters[key].update(self.parameters[key])
                self.parameters[key] = self.default_parameters[key]

        # ParallelHelper Init
        PH.init_from_env(self.parameters.get("use_cuda"))
        set_global_seed(self.parameters.get("random_seed") + PH.rank)

        self.initialized = False
        self.nelec = None
        self.nspins = None
        self.nbands = None

        self.grid = None
        self.kpoint = None
        self.pp = None
        self.poisson_solver = None
        self.xc_functional = None
        self.occupation = None
        self.preconditioner = None
        self.eigensolver = None
        self.density = None
        self.scf = None
        self.hamiltonian = None

        self.eigvec = None
        self.eigval = None
        self.fermi_level = None
        self.band_gap = None
        return

    @Timer.timeit
    def initialize(self, atoms):
        """Initialize the objects required for the calculation.
        Objects : Atoms, Grid, Kpoint, Pseudopotential, Poisson_solver, XC_Functional,
                  Occupation, Eigensolver, Density
        """
        self.atoms = atoms.copy()  # atoms is deep-copied to Calculator.atoms

        if self.parameters["output_file"] is not None:
            sys.stdout = open(self.parameters["output_file"], "a")

        atoms = atoms.copy()

        ## Check that the cell is a cubic. (Currently, only cubic cell is supported.)
        assert all(atoms.cell.cellpar()[:3] != np.array([0.0, 0.0, 0.0])), (
            "zero value is detected in atoms.cell. atoms.cell may not be set. Current cell length is "
            + str(atoms.cell.cellpar()[:3])
        )
        assert all(
            abs(atoms.cell.cellpar()[3:] - np.full(3, 90.0)) < 1e-7
        ), f"non-orthogonal cell vectors are not allowed, cellpar={atoms.cell.cellpar()}"

        ## Convert unit of Atoms object to atomic unit. (ang to Bohr)
        atoms.set_positions(atoms.get_positions() / Bohr)
        atoms.set_cell(atoms.cell / Bohr)

        ## Apply centered option to positions in ASE Atoms object.
        if self.parameters.centered:
            atoms.center()

        """Make Objects"""
        ## Grid
        if self.grid is None:
            spacing = self.parameters.grid.get("spacing")
            if spacing is not None:
                spacing /= Bohr  # ang to Bohr
            self.grid = Grid(
                atoms,
                self.parameters.grid.get("gpts"),
                spacing,
                self.parameters.grid.get("FD_order"),
            )
            print(self.grid)

        ## K-point
        if self.kpoint is None:
            self.kpoint = Kpoint(
                atoms,
                self.parameters.get("kpts"),
                self.parameters.get("kpts_and_weights"),
                self.parameters.get("group_sym"),
                self.parameters.get("inv_sym"),
            )
            print(self.kpoint)

        ## Pseudopotential
        if self.pp is None:
            self.pp = create_Pseudopotential(
                self.grid,
                self.parameters.get("pp"),
                kpts=self.kpoint.kpts,
                device=PH.get_device(),
            )
            print(self.pp)

        ## Set 'nspins', 'nbands', 'nelec'
        self.nelec = (
            self.parameters.nelec
            if self.parameters.nelec != None
            else self.pp.num_valence_e
        )
        if self.parameters.nbands == None:
            num_virtual = int(ceil(self.nelec / 2) * 0.05)
            num_virtual = 4 if num_virtual < 4 else num_virtual
            self.nbands = ceil(self.nelec / 2) + num_virtual
        else:
            self.nbands = self.parameters.nbands
        ## Check nbands is sufficient.
        if self.nbands < ceil(self.nelec / 2):
            self.nbands = ceil(self.nelec / 2) + 4
            print(
                f"WARNING: 'nbands' is not sufficient. 'nbands' is set to {self.nbands}"
            )
        self.nspins = 2 if self.parameters.spin == "polarized" else 1

        ## check use_cuda
        if PH.device_count() > 0 and self.parameters["use_cuda"] == False:
            print(
                f"WARNING: current node has GPU but this calcualtion does not utilize it. In most cases, GPU provide overwhelming performance."
            )

        ## Poisson_solver
        if self.poisson_solver is None:
            self.poisson_solver = create_poisson(
                self.grid,
                self.parameters.get("poisson"),
                device=PH.get_device(),
            )
            print(self.poisson_solver)

        ## XC_Functional
        if self.xc_functional is None:
            self.xc_functional = create_xc(self.parameters.get("xc"), self)
            print(self.xc_functional)

        ## Occupation
        if self.occupation is None:
            self.occupation = Occupation(
                self.nelec,
                self.nspins,
                self.nbands,
                nkpts=self.kpoint.nkpts,
                scaled_kpts=self.kpoint.scaled_kpts,
                weights=self.kpoint.weights,
                **self.parameters.get("occupation"),
            )
            print(self.occupation)

        ## Preconditioner
        if self.preconditioner is None:
            self.preconditioner = create_preconditioner(
                self.parameters.get("precond_type"),
                self.grid,
                use_cuda=self.parameters["use_cuda"],
            )
            print(self.preconditioner)

        ## Eigensolver
        if self.eigensolver is None:
            self.eigensolver = create_eigensolver(self.parameters.get("eigensolver"))
            self.eigensolver.preconditioner = self.preconditioner
            print(self.eigensolver)

        ## Density
        if self.density is None:
            self.density = Density(
                self.grid,
                self.pp,
                self.nelec,
                self.nspins,
                self.occupation.magmom,
                self.occupation.fix_magmom,
                init_type=self.parameters.get("init_guess"),
                device=PH.get_device(),
            )
            print(self.density)

        ## Print input options
        from pprint import pprint

        if PH.is_master():
            print(
                "=========================[ Input Parameters ]========================="
            )
            pprint(self.parameters)
            print(f"\n* pbc                      : {atoms.get_pbc()}")
            print(f"\n* Positions of atoms (ang) :")
            for iatom, symbol in enumerate(atoms.get_chemical_symbols()):
                print(f"{symbol} : {atoms.get_positions()[iatom] * Bohr}")
            print(
                "======================================================================"
            )
        self.initialized = True
        return

    @Timer.timeit
    def calculate(self, atoms=None, properties=["energy"], system_changes=["cell"]):
        print("Date : ", datetime.today())

        # if atoms is not None, atoms is deep-copied to self.atoms
        Calculator.calculate(self, atoms, properties)

        if system_changes:
            if system_changes == ["position"]:
                self.pp = self.density = None
            else:
                self.grid = None
                self.kpoint = None
                self.pp = None
                self.poisson_solver = None
                self.xc_functional = None
                self.occupation = None
                self.preconditioner = None
                self.eigensolver = None
                self.density = None
                self.scf = None
                self.hamiltonian = None
        self.initialize(self.atoms)

        self.hamiltonian = Hamiltonian(
            self.nspins,
            self.nbands,
            self.grid,
            self.kpoint,
            self.pp,
            self.poisson_solver,
            self.xc_functional,
            self.eigensolver,
            use_dense_kinetic=self.parameters.hamiltonian.get("use_dense_kinetic"),
            multi_dtype=self.parameters.hamiltonian.get("multi_dtype"),
            device=PH.get_device(),
        )

        ## SCF Cycle
        if self.parameters.iter_type == "scf":
            self.scf = SCF(
                self.parameters.get("convergence"),
                self.occupation,
                self.parameters.get("mixing"),
                self.density,
            )
        elif self.parameters.iter_type == "lbfgs":
            self.scf = LBFGS(
                self.parameters.get("convergence"),
                self.occupation,
                self.parameters.get("lbfgs"),
                self.density,
            )
        else:
            raise ValueError(
                f"iter_type={self.parameters.iter_type} is not supported. Use 'scf' or 'lbfgs'."
            )
        print(self.scf)

        self.scf.iterate(self.hamiltonian, self.parameters.get("print_energies"))

        if self.scf.converged:
            self.results["free_energy"] = self.scf.energy * Ha  # Ha to eV
            ## thermochemical energy ("energy") is not implemented yet.
            self.results["energy"] = self.results["free_energy"]
            self.eigvec = self.scf.eigvec
            self.eigval = self.scf.eigval
            self.fermi_level = self.scf.fermi_level
            print(f"Total Energy: {self.scf.energy.item()} Ha")
            if self.scf.fermi_level:
                print(f"Fermi Level : {self.scf.fermi_level * Ha} eV")

                # ase.dft.bandgap module use IOContext. It yields potential problems. Only master rank include proper bandgap values.
                if PH.is_master():
                    if self.fermi_level < self.eigval.max():
                        self.band_gap = bandgap(
                            self,
                            output="-",
                            efermi=self.scf.fermi_level.cpu().numpy() * Ha,
                        )

            if "forces" in properties:
                self.results["forces"] = self.calc_forces(
                    self.hamiltonian, self.parameters["force"]
                ).numpy() * (Ha / Bohr)
            if "stress" in properties:
                self.results["stress"] = self.calc_stress(atoms)  # (eV / ang**3)
            if "dipole" in properties:
                self.results["dipole"] = self.density.calc_dipole_moment()
        else:
            print("Warning: scf is not converged. energy is set to zero.")
            for p in properties:
                self.results[p] = None
        return

    def calc_forces(self, hamiltonian, options):
        forces = hamiltonian.calc_forces(self.density, **options)

        print("Atomic Forces (eV/ang):")
        print(" index | symbol |  x-axis  |  y-axis  |  z-axis")
        for i_atom, sym in enumerate(self.atoms.get_chemical_symbols()):
            fx, fy, fz = np.round(forces[i_atom] * Ha / Bohr, 5)
            print(f"{i_atom:>{6}} | {sym:>{6}} | {fx:>{8}} | {fy:>{8}} | {fz:>{8}}")
        return forces

    def calc_stress(self, atoms, d=1e-3, voigt=True):
        """calculate stress tensor"""
        return self._numeric_stress(atoms, d, voigt)

    def _numeric_stress(self, atoms, d=1e-3, voigt=True):
        """
        Modify ase.calculators.test.numeric_stress to only support cartesian cell
        Unit: eV / ang**3
        """
        stress = np.zeros((3, 3), dtype=float)

        cell = atoms.cell.copy()
        V = atoms.get_volume()
        for i in range(3):
            x = np.eye(3)
            x[i, i] += d
            # atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            new_cell = np.dot(cell, x)
            new_cell = np.diag(new_cell.diagonal())
            atoms.set_cell(new_cell, scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, i] -= 2 * d
            # atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            new_cell = np.dot(cell, x)
            new_cell = np.diag(new_cell.diagonal())
            atoms.set_cell(new_cell, scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)

            stress[i, i] = (eplus - eminus) / (2 * d * V)
            x[i, i] += d

            j = i - 2
            x[i, j] = d
            x[j, i] = d
            # atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            new_cell = np.dot(cell, x)
            new_cell = np.diag(new_cell.diagonal())
            atoms.set_cell(new_cell, scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, j] = -d
            x[j, i] = -d
            # atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            new_cell = np.dot(cell, x)
            new_cell = np.diag(new_cell.diagonal())
            atoms.set_cell(new_cell, scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)

            stress[i, j] = (eplus - eminus) / (4 * d * V)
            stress[j, i] = stress[i, j]
        atoms.set_cell(cell, scale_atoms=True)

        if voigt:
            return stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            return stress

    @Timer.timeit
    def calculate_band_structure(
        self,
        density=None,
        path=None,
        npoints=None,
        nbands=None,
        bulk=None,
        eigensolver=None,
    ):
        """Calculate band structure.

        :type  density: str of array
        :param density:
            filename or array containing the converged density
        :type  path: str, optional
        :param path:
            band path
        :type  npoints: int, optional
        :param npoints:
            number of points
        :type  nbands: int, optional
        :param nbands:
            the number of bands
        :type  bulk: ase.Atoms, optional
        :param bulk:
            used to represent the k-point path of some crystal structure as an orthorhombic cell.
        :type  eigensolver: gospel.Eigensolver
        :param eigensolver:
            eigensolver for band calculation

        :rtype: ase.spectrum.band_structure.BandStructure
        :return:
            BandStructure object

        **Example**

        >>> atoms = bulk('C', 'diamond', a=3.57, cubic=True)
        >>> calc = GOSPEL(
        ...     pp={'upf':["/home/jhwoo/PP/pbe.0.3.1/PSEUDOPOTENTIALS_NC/C.pbe-nc.UPF"]},
        ...     xc={'type':'gga_x_pbe + gga_c_pbe'},
        ...     kpts=(3,3,3,'gamma'),
        ... )
        >>> atoms.calc = calc
        >>> atoms.get_potential_energy()
        >>> bs = calc.calculate_band_structure()
        >>> bs.plot(filename='band.png', show=True)
        """
        from ase.spectrum.band_structure import BandStructure

        if bulk is None:
            cell = self.atoms.cell.copy()
            cell *= self.atoms.pbc.reshape(-1, 1)
            bandpath = cell.bandpath(path=path, npoints=npoints)
            kpts = bandpath.kpts
        else:
            assert path is not None
            assert np.all(
                abs(self.atoms.cell.cellpar()[3:] - [90, 90, 90]) < 1e-7
            ), "'bulk' is only required when self.atoms.cell is an orthorhombic cell."
            from gospel.Kpoint import represent_kpts_for_orthorhombic_cell

            kpts = bulk.cell.bandpath(path=path, npoints=npoints).kpts
            kpts = represent_kpts_for_orthorhombic_cell(
                kpts, bulk, self.atoms.cell.cellpar()[:3]
            )
            bandpath = bulk.cell.bandpath(path=path, npoints=npoints)

        self.parameters["nelec"] = self.nelec
        self.parameters["nbands"] = nbands
        self.parameters["kpts"] = kpts
        if eigensolver is not None:
            self.parameters["eigensolver"] = eigensolver
        else:
            from gospel.Eigensolver.lobpcg import LOBPCG

            self.parameters["eigensolver"] = LOBPCG(maxiter=100)

        ## Only initialize objects that depend on k-points.
        del (self.kpoint, self.pp, self.occupation, self.eigensolver)
        self.kpoint = self.pp = self.occupation = self.eigensolver = None
        self.initialize(self.atoms)
        hamiltonian = Hamiltonian(
            self.nspins,
            self.nbands,
            self.grid,
            self.kpoint,
            self.pp,
            self.poisson_solver,
            self.xc_functional,
            self.eigensolver,
            use_dense_kinetic=self.parameters.hamiltonian.get("use_dense_kinetic"),
            multi_dtype=self.parameters.hamiltonian.get("multi_dtype"),
            device=PH.get_device(),
        )

        ## Make Density object and update Hamiltonian
        if density is None:
            if self.scf is None:
                print("Warning: SCF is not defined.")
            elif not self.scf.converged:
                print(f"Warning: SCF is not converged.")
            hamiltonian.update(self.density)
        else:
            rho = Density(
                self.grid,
                self.pp,
                self.nelec,
                self.nspins,
                self.parameters.occupation["magmom"],
                self.parameters.occupation["fix_magmom"],
                init_type={"density": density, "orbital": None, "occ": None},
            )
            rho.set_density(rho.init_density())
            hamiltonian.update(rho)

        ## Calculate eigenvalues and band gap.
        self.eigval, _ = hamiltonian.diagonalize()
        _ = self.occupation.get_occupation(self.eigval)
        fermi_level = self.occupation.fermi_level.item() * Ha
        self.band_gap = bandgap(self, output="-", efermi=fermi_level)
        return BandStructure(bandpath, self.eigval * Ha, fermi_level)

    def get_k_point_weights(self):
        return self.kpoint.weights

    def get_number_of_spins(self):
        return self.nspins

    def get_number_of_bands(self):
        return self.nbands

    def get_density(self, spin=0):
        return self.density.get_density().cpu()[spin]

    def get_occupation(self, kpt=0, spin=0):
        return self.occupation.get_occupation(self.eigval.cpu())[spin, kpt]

    def get_eigenvalues(self, kpt=0, spin=0):
        # Unit: eV
        return np.array(self.eigval[spin, kpt].cpu()) * Ha

    def get_eigenvectors(self, kpt=0, spin=0):
        return self.eigvec[spin, kpt].cpu()

    def get_fermi_level(self):
        # Unit: eV
        return self.fermi_level * Ha

    def get_bz_k_points(self):
        ## Return BZ k-points.
        pass

    def get_bz_to_ibz_map(self):
        ## Return indices from BZ to IBZ.
        pass

    def get_ibz_k_points(self):
        ## Return k-points in the irreducible Brillouin zone.
        return self.kpoint.scaled_kpts.copy()
