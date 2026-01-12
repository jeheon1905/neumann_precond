from abc import ABCMeta, abstractmethod
from itertools import product
import torch
from ase.data import chemical_symbols

from gospel.FdOperators import gradient, calc_deriv_1D
from gospel.Pseudopotential.Compensation import Compensation
from gospel.Pseudopotential.Filtering import Filtering
from gospel.util import Timer
from gospel.ParallelHelper import ParallelHelper as PH


class Pseudopotential(metaclass=ABCMeta):
    """
    This class is in charge of ion-ion and ion-electron interactions by using pseudopotential method.
    This class is parent class of several types of pseudopotential classes, e.g., NCPP and LUSP.
    Notice that only NCPP (UPF 2.1 ver) is currently supported.

    :type  grid: gospel.Grid
    :param grid:
        Grid object
    :type  upf: gospel.Pseudopotential.UPF
    :param upf:
        UPF object. It read UPF files and ready to use.
    :type  NLCC: bool
    :param NLCC:
        use non-linear core correction, defaults to True
    :type  use_comp: bool
    :param use_comp:
        whether to use compensation charge, defaults to True
    :type  filtering: bool
    :param filtering:
        use fourier filtering method
    :type  num_near_cell: int
    :param num_near_cell:
        the number of near cells to consider when PBC
    :type  kpts: list or ndarray
    :param kpts:
        list of k-points, e.g., [[0,0,0], [1,0,0], ...]

    **Example**

    >>> pp = Pseudopotential(grid, {'upf':['/upf_filename']})
    >>> E_NN = pp.ion_ion_repulsion_energy
    >>> E_ext, E_NL = pp.calc_external_energy(density, occ, eigvec, eigvec)
    """
    @Timer.timeit
    def __init__(
        self,
        grid,
        upf,
        NLCC=True,
        use_comp=True,
        filtering=False,
        num_near_cell=1,
        kpts=[[0, 0, 0]],
        device=torch.device("cpu"),
    ):
        self._grid = grid
        self._atoms = grid.atoms
        # self._atoms_positions = torch.from_numpy(self._atoms.get_positions())  # TODO: this line should be activated later.
        self._upf = upf
        self._NLCC = NLCC
        self._use_comp = use_comp
        self._filtering = filtering
        self._num_near_cell = num_near_cell
        self._kpts = kpts
        self._has_nonlocal = False
        self.device = device

        ## When use_comp=True, compensate local potential
        if use_comp:
            self._compensation = Compensation(grid, upf, self._num_near_cell, device)
            print(self._compensation)
            upf.local = self._compensation.V_short
            upf.local_cutoff = self._compensation.short_cutoff
            self._ion_ion_repulsion_energy = 0.0
        else:
            self._ion_ion_repulsion_energy = self.calc_ion_ion_repulsion()

        ## When filtering=True, filtering local and KB projectors
        if filtering:
            assert use_comp, "'filtering' is only supported with 'use_comp'=True"
            ## Note that updated cutoff information is only in the UPF object.
            self._filter = Filtering(max(grid.spacings))
            old_local_cutoff, old_beta_cutoff = upf.local_cutoff, upf.beta_cutoff
            upf.local, upf.local_cutoff = self._filter.filtering(upf, "local")
            upf.beta, upf.beta_cutoff = self._filter.filtering(upf, "nonlocal")

            s = (
                "\n======================================================================="
                "\n* Pseudopotential filtering is applied."
                "\n symbol | r_short (Bohr) | r_betas (Bohr)"
            )
            for sym in upf.elements:
                s += (
                    f"\n {sym:>{6}} |"
                    f" {round(old_local_cutoff[sym], 3):<{5}} -> {round(upf.local_cutoff[sym], 3):<{5}} |"
                    f" {old_beta_cutoff[sym]} -> {upf.beta_cutoff[sym]}"
                )
            s += "\n=======================================================================\n"
            print(s)
            del old_local_cutoff, old_beta_cutoff

        ## Calculate the number of valence electrons
        self._num_valence_e = 0
        for i_atom, i_sym in enumerate(self._atoms.get_chemical_symbols()):
            self._num_valence_e += upf[i_sym].zval

        self._V_ext = self.calc_local_pot(grid, upf, use_comp)
        return

    @Timer.timeit
    def calc_local_pot(self, grid, upf, use_comp, fine=None):
        """Read local potential from upf and represent it on grid.

        :type  grid: gospel.Grid
        :param grid:
            Grid class object
        :type  upf: gospel.Pseudopotential.UPF
        :param upf:
            UPF class object
        :type  use_comp: bool
        :param use_comp:
            whether to use compensation charge
        :type  fine: int, optional
        :param fine:
            multiples of finer grid, defaults to None

        :rtype: torch.Tensor
        :return:
            local potential, shape=(ngpts,)

        Eq) V^{\text{loc}}
            = \sum_{i} V^{\text{loc}}_{i}
            = \sum_{i} \int \frac{n^{\text{loc}}_{i}}{|\vec{r} - \vec{r}'} d\vec{r}'

            When compensation charge is used (DOI: 10.1063/1.2193514),
            V^{\text{loc}} + V^{\text{comp}}
            = \sum_{i} ( V^{\text{loc}}_{i} + V^{\text{comp}}_{i} )
            = V^{\text{short}}
            : short-ranged potential
        """
        atoms = grid.atoms  # ase.Atoms object

        if isinstance(fine, int):
            from gospel.Grid import Grid

            original_grid = grid
            grid = Grid(atoms, grid.gpts * fine)

        atomic_numbers = PH.split(torch.from_numpy(self._atoms.get_atomic_numbers()))
        atoms_positions = PH.split(torch.from_numpy(self._atoms.get_positions()), dim=0)

        local_pot = torch.zeros(grid.ngpts, dtype=torch.float64, device=self.device)
        for i_atom, atomic_number in enumerate(atomic_numbers):
            sym = chemical_symbols[atomic_number]
            extplt = None if use_comp else f"-{upf[sym].zval}/r"
            local_pot += grid.spherical_to_grid(
                upf[sym].R,
                upf[sym].local,
                atoms_positions[i_atom],
                extplt=extplt,
                cutoff_radius=upf[sym].local_cutoff,
                num_near_cell=self._num_near_cell,
                device=self.device,
            )
        local_pot = PH.all_reduce(local_pot)

        if isinstance(fine, int):
            from Supersampling import TruncSincSupersampling3D, run

            conv = TruncSincSupersampling3D(
                [fine, fine, fine], [5, 5, 5], original_grid.spacings
            ).to(self.device)
            local_pot = run(conv, local_pot.reshape(grid.gpts), grid.get_pbc())
            local_pot = local_pot.reshape(-1)
            grid = original_grid
            assert len(local_pot) == grid.ngpts
        return local_pot

    def get_nuclear_pot(self):
        """- Z/r potential"""
        nuclear_pot = torch.zeros(self._grid.ngpts, dtype=torch.float64)
        atomic_numbers = torch.from_numpy(self._atoms.get_atomic_numbers())
        atoms_positions = torch.from_numpy(self._atoms.get_positions())
        for i_atom, i_sym in enumerate(self._atoms.get_chemical_symbols()):
            ## Z_i / |r - R_i|
            d_r_R_i = (self._grid.points - atoms_positions[i_atom]).norm(dim=1)
            nuclear_pot -= atomic_numbers[i_atom] / d_r_R_i
        return nuclear_pot

    def init_density(self, device):
        """Guess initial density from PP_RHOATOMs in pseudopotential files."""
        density = torch.zeros(self._grid.ngpts, dtype=torch.float64, device=self.device)

        atomic_numbers = PH.split(torch.from_numpy(self._atoms.get_atomic_numbers()))
        # atoms_positions = PH.split(torch.from_numpy(self._atoms.get_positions()),dim=0).numpy()
        atoms_positions = PH.split(torch.from_numpy(self._atoms.get_positions()), dim=0)
        for i_atom, atomic_number in enumerate(atomic_numbers):
            sym = chemical_symbols[atomic_number]
            density += self._grid.spherical_to_grid(
                self._upf[sym].R,
                self._upf[sym].rhoatom,
                atoms_positions[i_atom],
                cutoff_radius=self._upf[sym].rho_cutoff,
                num_near_cell=self._num_near_cell,
                device=self.device,
            )
        return PH.all_reduce(density)

    def calc_ion_ion_repulsion(self):
        """Calculate ion-ion repulsion energy.
        Eq) E_{ion-ion} = \frac{1}{2} \sum_{i > j} \frac{ Z_{i} Z_{j} }{ r_{ij} }
        """
        ion_ion = 0.0
        atoms = self._atoms
        atoms_positions = torch.from_numpy(atoms.get_positions())
        cell = self._grid.get_cell()
        num_near = [self._num_near_cell if atoms.get_pbc()[i] else 0 for i in range(3)]

        for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
            Z_i = self._upf[i_sym].zval
            R_i = atoms_positions[i_atom]
            for j_atom, j_sym in enumerate(atoms.get_chemical_symbols()[i_atom:]):
                j_atom += i_atom
                Z_j = self._upf[j_sym].zval
                for i_xyz in product(
                    range(-num_near[0], num_near[0] + 1),
                    range(-num_near[1], num_near[1] + 1),
                    range(-num_near[2], num_near[2] + 1),
                ):
                    if i_atom == j_atom and i_xyz == (0, 0, 0):
                        continue
                    i_xyz = torch.tensor(i_xyz, dtype=cell.dtype, device=cell.device)
                    R_j = atoms_positions[j_atom] + i_xyz @ cell
                    R_ij = R_i - R_j  # (R_i - R_j) vector
                    d_ij = R_ij.norm()  # |R_i - R_j|
                    ion_ion += Z_i * Z_j / d_ij
        return ion_ion

    def calc_ion_ion_forces(self):
        atoms = self._atoms
        symbols = atoms.get_chemical_symbols()
        upf = self._upf
        atoms_positions = torch.from_numpy(atoms.get_positions())
        cell = self._grid.get_cell()
        num_near = [self._num_near_cell if atoms.get_pbc()[i] else 0 for i in range(3)]
        F = torch.zeros((len(symbols), 3), dtype=torch.float64)

        for i_atom, i_sym in enumerate(symbols):
            Z_i = upf[i_sym].zval
            R_i = atoms_positions[i_atom]
            for j_atom, j_sym in enumerate(symbols):
                Z_j = upf[j_sym].zval
                for i_xyz in product(
                    range(-num_near[0], num_near[0] + 1),
                    range(-num_near[1], num_near[1] + 1),
                    range(-num_near[2], num_near[2] + 1),
                ):
                    if i_atom == j_atom:
                        continue
                    i_xyz = torch.tensor(i_xyz, dtype=cell.dtype, device=cell.device)
                    R_j = atoms_positions[j_atom] + i_xyz @ cell
                    R_ij = R_i - R_j  # (R_i - R_j) vector
                    d_ij = R_ij.norm()  # |R_i - R_j|
                    F[i_atom] += (Z_i * Z_j / d_ij**3) * R_ij
        return F

    @Timer.timeit
    def calc_NLCC_core_density(self):
        """Represent NLCC core densites on grid from spherical coordinates."""
        NLCC_core_density = torch.zeros(
            self._grid.ngpts, dtype=torch.float64, device=self.device
        )

        atomic_numbers = PH.split(torch.from_numpy(self._atoms.get_atomic_numbers()))
        # atom_positions = PH.split(torch.from_numpy(self._atoms.get_positions()),dim=0).numpy()
        atom_positions = PH.split(torch.from_numpy(self._atoms.get_positions()), dim=0)
        for i_atom, atomic_number in enumerate(atomic_numbers):
            sym = chemical_symbols[atomic_number]
            if self._upf[sym].nlcc is None:
                continue
            else:
                NLCC_core_density += self._grid.spherical_to_grid(
                    self._upf[sym].R,
                    self._upf[sym].nlcc,
                    atom_positions[i_atom],
                    cutoff_radius=self._upf[sym].nlcc_cutoff,
                    num_near_cell=self._num_near_cell,
                    device=self.device,
                )
        NLCC_core_density = PH.all_reduce(NLCC_core_density)
        return NLCC_core_density

    def calc_local_forces(self, density, deriv_density=True):
        r"""Calculate atomic forces from local potential. (2 schemes: derivativeis of density of potential, each True of False)

        :type  density: torch.Tensor
        :param density:
            density values, shape=(ngpts,)
        :type  deriv_density: bool
        :param deriv_density:
            derivatives of density or potential, defaults to True

        :rtype: torch.Tensor
        :return:
            local part forces, shape=(natoms, 3)

        if deriv_density is False
        F^{a}_{loc} = \int \frac{\round V^{a}_{loc}(r)}{\round r} \rho(r)
                      (\vec{r} - \vec{R}_{a}) / |\vec{r} - \vec{R}_{a}| d\vec{r}

        if deriv_density is True
        F^{a}_{loc} = - \int V^{a}_{loc}(r) \nabla \rho(r) d\vec{r}
        """
        symbols = self._atoms.get_chemical_symbols()
        upf = self._upf
        F = torch.zeros((len(symbols), 3), dtype=torch.float64, device=self.device)
        atoms_positions = torch.from_numpy(self._atoms.get_positions()).to(self.device)
        points = self._grid.points.to(self.device)

        if not deriv_density:  # derivate potential
            for i_atom, i_sym in enumerate(symbols):
                R_i = atoms_positions[i_atom]
                dV_loc = self._grid.spherical_to_grid(
                    upf[i_sym].R,
                    calc_deriv_1D(upf[i_sym].R, upf[i_sym].local),
                    R_i,
                    extplt=f"{upf[i_sym].zval}/r**2",
                    cutoff_radius=upf[i_sym].local_cutoff,
                    num_near_cell=self._num_near_cell,
                    device=self.device,
                )
                r_R_i = points - R_i  # r_R_i.shape = (N_g, 3)
                d_r_R_i = r_R_i.norm(dim=1, keepdim=True)  # shape = (N_g, 1)
                r_R_i = r_R_i / d_r_R_i  # (r - R_i)/|r - R_i|
                r_R_i[d_r_R_i.squeeze() == 0, :] = 0.0  # handle divide by zero
                F[i_atom] = self._grid.integrate(density * dV_loc * r_R_i.T).reshape(-1)
        else:  # derivate density instead of potential
            drho = gradient(self._grid, density)
            for i_atom, i_sym in enumerate(symbols):
                R_i = atoms_positions[i_atom]
                V_loc = self._grid.spherical_to_grid(
                    upf[i_sym].R,
                    upf[i_sym].local,
                    R_i,
                    extplt=f"-{upf[i_sym].zval}/r",
                    cutoff_radius=upf[i_sym].local_cutoff,
                    num_near_cell=self._num_near_cell,
                    device=self.device,
                )
                F[i_atom] = -self._grid.integrate(V_loc * drho).reshape(-1)
        return F

    @Timer.timeit
    def calc_external_energy(self, density):
        """Calculate external energy

        :type  density: gospel.Density
        :param density:
            Density class object

        :rtype: float
        :return:
            external energy

        ( Eq )
        E_{ext} = \int \rho(\vec{r}) V^{loc}(\vec{r}) d\vec{r}
        """
        E_ext = self._grid.integrate(density.get_density().sum(0) * self._V_ext)
        if self._use_comp:
            E_ext += self._compensation.energy_correction
        return E_ext

    def get_comp_charge(self):
        if self._use_comp:
            return self._compensation.comp_charge
        else:
            return 0

    def get_S(self):
        return None

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def calc_forces(self):
        pass

    @property
    def num_valence_e(self):
        return self._num_valence_e

    @property
    def use_comp(self):
        return self._use_comp

    @property
    def NLCC(self):
        return self._NLCC

    @property
    def V_ext(self):
        return self._V_ext

    @property
    def ion_ion_repulsion_energy(self):
        return self._ion_ion_repulsion_energy

    @property
    def has_nonlocal(self):
        return self._has_nonlocal
