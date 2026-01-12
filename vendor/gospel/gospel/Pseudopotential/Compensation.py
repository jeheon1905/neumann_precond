import copy
from itertools import product
from math import log, sqrt, pi
from time import time
import torch
import numpy as np
from scipy.special import erf as scipy_erf
from ase import Atoms
from ase.data import chemical_symbols

from gospel.FdOperators import gradient, calc_deriv_1D
from gospel.Grid import Grid
from gospel.Pseudopotential.UPF import get_zero_convg_idx
from gospel.util import Timer
from gospel.ParallelHelper import ParallelHelper as PH


class Compensation:
    """Compensation charge techniques. (DOI: 10.1063/1.2193514)

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  upf: gospel.Pseudopotential.UPF
    :param upf:
        UPF reader
    :type  num_near_cell: int, optional
    :param num_near_cell:
        the number of near cells to consider when PBC, defaults to 1
    """

    @Timer.timeit
    def __init__(self, grid, upf, num_near_cell=1, device=torch.device("cpu")):
        self.__grid = grid
        self.__atoms = grid.atoms
        self.__upf = upf
        self.device = device

        self.__r_gauss = (
            1.0  # the available optimization of this value is not considered.
        )
        self.__num_near_cell = num_near_cell
        self.__num_near = [num_near_cell if grid.get_pbc()[i] else 0 for i in range(3)]
        self.__tol = 1e-6  # tolerance considered to be zero
        # self.__tol = 1e-7  # tolerance considered to be zero

        ## zero cutoff radii of compensation charges
        self.__r_comp = {}
        for i_atom, i_sym in enumerate(self.__atoms.get_chemical_symbols()):
            self.__r_comp[i_sym] = self._get_r_comp(
                upf[i_sym].zval, self.__r_gauss, self.__tol
            )

        self.__comp_charge = self._comp_charges_on_grid(grid, upf)
        self.__V_short, self.__short_cutoff = self._calc_V_short(
            self.__atoms, upf, self.__tol
        )

        # self.__energy_correction = self._calc_energy_correction(grid, upf)
        self.__energy_correction = self._calc_energy_correction_batch(grid, upf)
        return

    def __str__(self):
        s = str()
        s += "\n=========================== [ Compensation ] =========================="
        s += f"\n symbol | r_gauss | r_short (Bohr) | r_comp (Bohr)"
        for sym in self.__upf.elements:
            s += f"\n {sym:>{6}} |"
            s += f" {round(self.__r_gauss, 3):>{7}} |"
            s += f" {round(self.__short_cutoff[sym], 3):>{14}} |"
            s += f" {round(self.__r_comp[sym], 3):>{13}}"
        s += "\n=======================================================================\n"
        return s

    def _calc_V_short(self, atoms, upf, tol=1e-6):
        """Calculate :math:`V^{short} = V^{loc} + V^{comp}

        V_short: short potentials (V^{loc} + V^{comp})
        short_cutoff: cutoff radii of each short potential
        """
        # TODO: UPF class will be replaced to pytorch from numpy implemenation later.
        # TODO: Filtering and KB_form classes will be replaced to pytorch from numpy implemenation later.
        V_short, short_cutoff = {}, {}

        for i_atom, i_sym in enumerate(set(atoms.get_chemical_symbols())):
            # V_comp = self._calc_comp_pot(upf[i_sym].zval, upf[i_sym].R)
            V_comp = self._calc_comp_pot(
                upf[i_sym].zval, torch.from_numpy(upf[i_sym].R)
            )
            # V_short[i_sym] = upf[i_sym].local + V_comp
            V_short[i_sym] = (torch.from_numpy(upf[i_sym].local) + V_comp).numpy()
            zero_convg_idx = get_zero_convg_idx(V_short[i_sym], tol)
            r_cut = upf[i_sym].R[zero_convg_idx]
            assert (
                r_cut < upf[i_sym].R[-1]
            ), f"cutoff radius of V_short is too long. (={r_cut})"
            ## XXX: The radial grid, upf.R, of SG15 pseudopotentials is very short, so it can be problematic when filtering is used.
            short_cutoff[i_sym] = r_cut
        return V_short, short_cutoff

    def _calc_comp_charge(self, Z, r):
        ## Z exp(- (r / r_gauss)^2) / (\sqrt{\pi} r_gauss)^3
        r_g = self.__r_gauss
        R = r / r_g
        return Z * torch.exp(-(R**2)) / (sqrt(pi) * r_g) ** 3

    def _calc_comp_pot(self, Z, r):
        ## V = Z erf( r / r_gauss ) / r
        r_g = self.__r_gauss
        R = r / r_g
        V_c_0 = 2 * Z / r_g / sqrt(pi)
        V_c = Z * torch.erf(R)
        V_c = torch.div(V_c, r)
        V_c[r == 0] = V_c_0
        return V_c

    def _calc_comp_charge_deriv(self, Z, r):
        r_g = self.__r_gauss
        R = r / r_g
        dn_c = -2 * Z * R * torch.exp(-(R**2)) / pi**1.5 / r_g**4
        return dn_c

    def _calc_comp_pot_deriv(self, Z, r):
        r_g = self.__r_gauss
        R = r / r_g
        dV_c = Z * (2 / sqrt(pi) * R * torch.exp(-(R**2)) - torch.erf(R))
        dV_c = torch.div(dV_c, r**2)
        dV_c[r == 0] = 0.0
        return dV_c

    @Timer.timeit
    def _comp_charges_on_grid(self, grid, upf):
        """Represent gaussian compensation charges of atoms on our grid."""
        atoms = grid.atoms
        num_near = self.__num_near
        cell = grid.get_cell().to(self.device)
        comp_charge = torch.zeros(grid.ngpts, dtype=torch.float64, device=self.device)
        points = grid.points.to(self.device)

        is_parallel = True
        is_batch_i_xyz = True
        print(
            f"Debug: _comp_charges_on_grid() with is_parallel={is_parallel}, is_batch_i_xyz={is_batch_i_xyz}"
        )

        if is_parallel:
            atomic_numbers = PH.split(torch.from_numpy(atoms.get_atomic_numbers())).to(
                self.device
            )
            atoms_positions = PH.split(
                torch.from_numpy(atoms.get_positions()), dim=0
            ).to(self.device)
        else:
            atomic_numbers = atoms.get_atomic_numbers()
            atoms_positions = torch.from_numpy(atoms.get_positions()).to(self.device)

        if is_batch_i_xyz:
            # shifts.shape=(nshifts, 1, 3)
            # points_batch.shape=(nshifts, ngpts, 3)
            # R_i_batch.shape=(nshifts, natoms, 3)
            i_xyzs = torch.tensor(
                list(
                    product(
                        range(-num_near[0], num_near[0] + 1),
                        range(-num_near[1], num_near[1] + 1),
                        range(-num_near[2], num_near[2] + 1),
                    )
                ),
                device=self.device,
                dtype=cell.dtype,
            )
            shifts = (i_xyzs @ cell).unsqueeze(dim=1)
            del i_xyzs
            center_point = torch.from_numpy(grid.center_point).to(self.device)

        for i_atom, atomic_number in enumerate(atomic_numbers):
            sym = chemical_symbols[atomic_number]
            r_comp = self.__r_comp[sym]

            if is_batch_i_xyz:
                R_i = atoms_positions[i_atom]
                R_i_batch = R_i.repeat(len(shifts), 1, 1)
                R_i_batch += shifts

                ## cutoff check
                d_R_i_R_c = (R_i_batch - center_point).norm(dim=-1)
                is_overlap_cell = d_R_i_R_c <= grid.get_radius() + r_comp
                R_i_batch = R_i_batch[is_overlap_cell.squeeze(dim=1)]
                nshifts = len(R_i_batch)
                points_batch = points.repeat(nshifts, 1, 1)

                d_r_R_i_batch = (points_batch - R_i_batch).norm(dim=-1)
                del points_batch

                mask = d_r_R_i_batch < r_comp
                # mask = d_r_R_i_batch < (r_comp * 2) # TODO: Now, r_comp is somewhat short.. -> slightly norm broken

                tmp = torch.zeros(
                    nshifts, grid.ngpts, dtype=torch.float64, device=self.device
                )
                tmp[mask] = self._calc_comp_charge(upf[sym].zval, d_r_R_i_batch[mask])
                comp_charge += tmp.sum(dim=0)
                continue

            for i_xyz in product(
                range(-num_near[0], num_near[0] + 1),
                range(-num_near[1], num_near[1] + 1),
                range(-num_near[2], num_near[2] + 1),
            ):
                i_xyz = torch.tensor(i_xyz, dtype=cell.dtype, device=self.device)
                R_i = atoms_positions[i_atom] + i_xyz @ cell  # R_i vector
                d_r_R_i = (points - R_i).norm(dim=1)  # |r - R_i|
                comp_charge += self._calc_comp_charge(upf[sym].zval, d_r_R_i)

        if is_parallel:
            comp_charge = PH.all_reduce(comp_charge)
        return comp_charge

    def _get_r_comp(self, Z, r_g, tol=1e-7):
        """Calculate the radius of the compensation charge being smaller than the tolerance value."""
        val = tol * (sqrt(pi) * r_g) ** 3 / Z
        return sqrt(-log(val)) * r_g

    @Timer.timeit
    def _calc_energy_correction_batch(self, grid, upf):
        num_near = self.__num_near
        atoms = grid.atoms
        device = self.device

        ## This function is not GPU effective, so only supports CPU calculations.
        atoms_positions = torch.from_numpy(atoms.get_positions()).to(device)
        cell = grid.get_cell().to(device)

        # Get chemical symbols and corresponding properties
        chemical_symbols = atoms.get_chemical_symbols()
        Z_values = torch.tensor(
            [upf[sym].zval for sym in chemical_symbols],
            dtype=torch.long,
            device=device,
        )
        r_comp_values = torch.tensor(
            [self.__r_comp[sym] for sym in chemical_symbols],
            dtype=torch.float64,
            device=device,
        )

        # image charges
        i_xyzs = torch.tensor(
            list(product(*(range(-num_near[i], num_near[i] + 1) for i in range(3)))),
            dtype=cell.dtype,
            device=device,
        )
        shifts = (i_xyzs @ cell).unsqueeze(dim=1)  # shape=(nshifts, 1, 3)

        # Generate indices for unique atom pairs
        i_atom = torch.arange(len(atoms), device=device)
        j_atom = i_atom

        # Calculate R_i and R_j for unique atom pairs
        R_i = atoms_positions  # shape=(natoms, 3)
        R_j = atoms_positions

        # R_i = R_i.repeat(len(R_j), 1, 1)
        R_i = R_i.repeat(len(shifts), 1, 1)
        R_j = R_j[None, :, :] + shifts  # shape=(nshifts, natoms, 3)

        # Calculate R_ij vectors and d_ij distances
        # Z_ij: pair of atomic charge product, shape = (natoms, natoms)
        # R_ij: pair of distance vectors, shape = (nshifts, natoms, natoms, 3)
        # d_ij: pair of distances, shape = (nshifts, natoms, natoms)
        Z_ij = Z_values[i_atom, np.newaxis] * Z_values[j_atom]
        R_ij = R_i[:, :, None].permute(1, 2, 3, 0) - R_j.permute(1, 2, 0)
        d_ij = torch.norm(R_ij, dim=2)
        del R_ij
        d_ij = d_ij.permute(2, 0, 1)

        ## Make mask for masking unvalid pairs
        # 1. masking with cutoff radius
        mask = d_ij <= r_comp_values[i_atom, np.newaxis] + r_comp_values[j_atom]
        # 2. masking overlap pairs in real cell, not in image cells
        no_image_cell = (
            (i_xyzs == torch.zeros(3, device=device)).all(dim=-1).nonzero().item()
        )
        mask[no_image_cell][np.tril_indices(len(atoms), k=-1)] = False

        ## Calculate interactions between compensation charges
        tmp1 = torch.zeros_like(d_ij)  # shape=(nshifts, natoms, natoms)
        tmp1[mask] = -self._calc_gau_gau_interaction_batch(d_ij[mask])
        tmp1 *= Z_ij
        tmp1[no_image_cell][np.diag_indices(len(tmp1[no_image_cell]))] *= 0.5

        ## Calculate interactions between nuclear charges
        tmp2 = torch.zeros_like(d_ij)
        tmp2[mask] = 1 / d_ij[mask]
        tmp2 *= Z_ij
        tmp2[no_image_cell][np.diag_indices(len(tmp2[no_image_cell]))] = 0.0

        correction = tmp1.sum() + tmp2.sum()
        return correction

    @Timer.timeit
    def _calc_energy_correction(self, grid, upf):
        """Calculate energy correction by using compensation charge.

        :type  grid: gospel.Grid
        :param grid:
            Grid object
        :type  upf: gospel.Pseudopotential.UPF
        :param upf:
            UPF object
        """
        correction = 0.0
        num_near = self.__num_near
        atoms = grid.atoms

        ## This function is not GPU effective.
        atoms_positions = atoms.get_positions()
        cell = grid.get_cell().numpy()

        for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
            Z_i = upf[i_sym].zval
            R_i = atoms_positions[i_atom]  # R_i vector
            r_comp_i = self.__r_comp[i_sym]
            # for j_atom, j_sym in enumerate(atoms.get_chemical_symbols()[i_atom:]):
            for j_atom, j_sym in enumerate(atoms.get_chemical_symbols()):  # [i_atom:]):
                # j_atom += i_atom
                Z_j = upf[j_sym].zval
                r_comp_j = self.__r_comp[j_sym]
                for i_xyz in product(
                    range(-num_near[0], num_near[0] + 1),
                    range(-num_near[1], num_near[1] + 1),
                    range(-num_near[2], num_near[2] + 1),
                ):
                    if i_atom < j_atom and i_xyz == (0, 0, 0):
                        continue
                    R_j = atoms_positions[j_atom] + i_xyz @ cell
                    R_ij = R_i - R_j  # (R_i - R_j) vector
                    # d_ij = np.linalg.norm(R_ij)  # |R_i - R_j|
                    d_ij = np.linalg.norm(R_ij, keepdims=True)  # |R_i - R_j|

                    val = 0.0
                    if d_ij > r_comp_i + r_comp_j:
                        continue
                    else:
                        # val = -self._calc_gau_gau_interaction(Z_i, Z_j, R_ij)
                        val = -Z_i * Z_j * self._calc_gau_gau_interaction_batch(d_ij)
                    if i_atom == j_atom and i_xyz == (0, 0, 0):
                        correction += val * 0.5
                    else:
                        correction += val + Z_i * Z_j / d_ij
        return correction

    @Timer.timeit
    def calc_short_forces(self, density, deriv_density=True):
        """
        Calculate atomic forces from short potential.
        (2 schemes: derivativeis of density of potential, each True of False)

        :type  density: torch.tensor
        :param density:
            density, shape=(ngpts,)
        :type  deriv_density: bool
        :param deriv_density:
            whether differentiate density or potential

        :rtype: torch.tensor
        :return:
            local part forces, shape=(natoms, 3)
        """
        atoms = self.__atoms
        upf = self.__upf

        st = time()
        ## Make small grids of each atom
        ## each grid point completely overlap with original grid's points.
        small_grids, idx_matching = self._make_small_grids(atoms, self.__short_cutoff)

        _density = density.cpu()
        if deriv_density:
            st = time()
            drho = gradient(self.__grid, density)
            print(f"deriv_density Time = {time() - st} sec")
            _drho = drho.cpu()

        F = torch.zeros((len(self.__atoms), 3), dtype=torch.float64)
        for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
            Z_i = upf[i_sym].zval
            R_i = torch.from_numpy(self.__atoms.get_positions()[i_atom])

            small_grid = small_grids[i_atom]
            idx_i = idx_matching[i_atom]

            if deriv_density:
                V_short = small_grid.spherical_to_grid(
                    upf[i_sym].R,
                    upf[i_sym].local,
                    R_i,
                    # cutoff_radius=self.__short_cutoff[i_sym],
                    cutoff_radius=upf[i_sym].local_cutoff,
                ).numpy()
                V_short = torch.from_numpy(V_short)
                drho_on_small = torch.zeros((3, small_grid.ngpts), dtype=torch.float64)
                for axis in range(3):
                    drho_on_small[axis] = _drho[axis].reshape(*self.__grid.gpts)[
                        idx_i[:, 0], idx_i[:, 1], idx_i[:, 2]
                    ]
                F[i_atom] = -small_grid.integrate(drho_on_small * V_short).reshape(-1)
            else:  ## calculate derivate of short potential instead of density
                r_R_i = small_grid.points - R_i  # (r - R_i) vector
                d_R_i = r_R_i.norm(dim=1, keepdim=True)  # |r - R_i|
                r_R_i = r_R_i / d_R_i  # (r - R_i)/|r - R_i|
                r_R_i[d_R_i.squeeze() == 0, :] = 0.0

                dV_short = small_grid.spherical_to_grid(
                    upf[i_sym].R,
                    calc_deriv_1D(upf[i_sym].R, upf[i_sym].local),
                    R_i,
                    cutoff_radius=upf[i_sym].local_cutoff,
                ).numpy()
                dV_short = torch.from_numpy(dV_short)
                rho_on_small = _density.reshape(*self.__grid.gpts)[
                    idx_i[:, 0], idx_i[:, 1], idx_i[:, 2]
                ]
                F[i_atom] = small_grid.integrate(
                    rho_on_small * dV_short * r_R_i.T
                ).reshape(-1)
        return F

    @Timer.timeit
    def get_comp_forces(self, potential, deriv_comp=True):
        """Calculate force correction by compensation charges.

        :type  potential: torch.tensor
        :param potential:
            poisson (coulombic) potential.
        :type  deriv_comp: bool, optional
        :param deriv_comp:
            whether differentiate compensation charge or potential, defaults to True

        :rtype: torch.tensor
        :return:
            force correction, shape=(natoms, 3)
        """
        atoms = self.__atoms
        upf = self.__upf

        st = time()
        ## Make small grids of each atom
        ## each grid point completely overlap with original grid's points.
        small_grids, idx_matching = self._make_small_grids(atoms, self.__r_comp)

        if not deriv_comp:
            st = time()
            dpotential = gradient(self.__grid, potential)
            print(f"Debug: dpotential Time = {time() - st} sec")

        F = torch.zeros((len(atoms), 3), dtype=torch.float64, device=self.device)
        atoms_positions = torch.from_numpy(atoms.get_positions()).to(self.device)

        for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
            Z_i = upf[i_sym].zval
            R_i = atoms_positions[i_atom]

            small_grid = small_grids[i_atom]
            idx_i = idx_matching[i_atom]
            # r_R_i = small_grid.points - R_i  # (r - R_i) vector
            r_R_i = small_grid.points.to(self.device) - R_i  # (r - R_i) vector
            d_R_i = r_R_i.norm(dim=1, keepdim=True)  # |r - R_i|
            r_R_i = r_R_i / d_R_i
            r_R_i[d_R_i.squeeze() == 0, :] = 0.0

            if deriv_comp:
                V_on_small = potential.reshape(*self.__grid.gpts)[
                    idx_i[:, 0], idx_i[:, 1], idx_i[:, 2]
                ]
                dn_comp_i = self._calc_comp_charge_deriv(Z_i, d_R_i.squeeze())
                F_correct_comp_pot = -small_grid.integrate(
                    dn_comp_i * V_on_small * r_R_i.T
                ).reshape(-1)
            else:  ## derivative of potential instead of compensation charge
                # n_comp_i = _calc_comp_charge(Z_i, d_R_i.squeeze(), self.__r_gauss)
                n_comp_i = self._calc_comp_charge(Z_i, d_R_i.squeeze())
                dV = torch.zeros(
                    (3, small_grid.ngpts), dtype=torch.float64, device=self.device
                )
                for axis in range(3):
                    dV[axis] = dpotential[axis].reshape(*self.__grid.gpts)[
                        idx_i[:, 0], idx_i[:, 1], idx_i[:, 2]
                    ]
                F_correct_comp_pot = small_grid.integrate(n_comp_i * dV).reshape(-1)
            F[i_atom] += F_correct_comp_pot
        return F

    @Timer.timeit
    def get_comp_comp_forces(self):
        """Calculate (dcomp|comp) - (dloc|loc)."""
        atoms = self.__atoms
        upf = self.__upf
        num_near = self.__num_near

        F = torch.zeros((len(atoms), 3), dtype=torch.float64, device=self.device)
        atoms_positions = torch.from_numpy(atoms.get_positions()).to(self.device)
        cell = self.__grid.get_cell().to(self.device)

        for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
            Z_i = upf[i_sym].zval
            R_i = atoms_positions[i_atom]
            r_comp_i = self.__r_comp[i_sym]

            ## Calculate forces between compensation charge and compensation potential.
            for j_atom, j_sym in enumerate(atoms.get_chemical_symbols()):
                Z_j = upf[j_sym].zval
                r_comp_j = self.__r_comp[j_sym]

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

                    if d_ij > r_comp_i + r_comp_j:
                        continue
                    else:
                        F[i_atom] += (
                            self._calc_dgau_gau_interaction(Z_i, Z_j, R_ij)
                            + Z_i * Z_j / d_ij**3 * R_ij
                        )
        return F

    @Timer.timeit
    def get_forces_correction(self, potential, deriv_comp=True):
        return self.get_comp_forces(potential, deriv_comp) + self.get_comp_comp_forces()

    def _calc_gau_gau_interaction_batch(self, d_ij):
        """Calculate an interaction between two spherical gaussian charges."""
        ## using 0th-order Boys function.
        r_g_i = self.__r_gauss
        r_g_j = self.__r_gauss
        alpha = 1 / (r_g_i**2 + r_g_j**2)
        if type(d_ij) == torch.Tensor:
            tmp = torch.erf(sqrt(alpha) * d_ij) / d_ij
        else:
            tmp = scipy_erf(sqrt(alpha) * d_ij) / d_ij
        tmp[d_ij < 1e-7] = 2 * sqrt(alpha / pi)
        return tmp

    def _calc_gau_gau_interaction(self, Z_i, Z_j, R_ij):
        """Calculate an interaction between two spherical gaussian charges."""
        ## using 0th-order Boys function.
        r_g_i = self.__r_gauss
        r_g_j = self.__r_gauss
        alpha = 1 / (r_g_i**2 + r_g_j**2)
        # d_ij = torch.linalg.norm(R_ij)
        d_ij = np.linalg.norm(R_ij)
        if d_ij < 1e-7:
            tmp = 2 * sqrt(alpha / pi)
        else:
            # tmp = torch.erf(sqrt(alpha) * d_ij) / d_ij
            tmp = scipy_erf(sqrt(alpha) * d_ij) / d_ij
        return Z_i * Z_j * tmp

    def _calc_dgau_gau_interaction(self, Z_i, Z_j, R_ij):
        """Calculate (dgau|gau)."""
        ## using 1st-order Boys function.
        r_g_i = self.__r_gauss
        r_g_j = self.__r_gauss
        alpha = 1 / (r_g_i**2 + r_g_j**2)
        d_ij = torch.linalg.norm(R_ij)
        tmp = (
            sqrt(4 * alpha / pi) * torch.exp(-alpha * d_ij**2) / d_ij**2
            - torch.erf(sqrt(alpha) * d_ij) / d_ij**3
        )
        return Z_i * Z_j * tmp * R_ij

    def _make_small_grids_each_element(self, r_cut):
        """
        Make small grids for each atom element.
        Each grid point completely overlap with original grid's points.
        It should be modified to apply to non-cartesian grid.

        :type  r_cut: dict
        :param r_cut:
            dictionary of symbols (key) and cutoff radii (value)
        :rtype: dict
        :return:
            dictionary of symbols (key) and Grid objects (value)
        """
        grid = self.__grid
        assert (
            grid.is_cartesian
        ), "It should be modified to apply to non-cartesian grid."
        cell = grid.get_cell()
        small_grids = {}
        for sym in set(self.__atoms.get_chemical_symbols()):
            gpts = np.ceil(r_cut[sym] / grid.spacings).astype(int) * 2
            small_cell = cell * (gpts / grid.gpts).reshape(-1, 1)
            small_grid = Grid(Atoms(cell=small_cell), gpts=gpts)
            assert np.all(
                abs(small_grid.spacings - grid.spacings) < 1e-7
            ), f"{small_grid.spacings} != {self.__grid.spacings}"
            small_grids[sym] = small_grid
        return small_grids

    @Timer.timeit
    def _make_small_grids(self, atoms, r_cuts):
        """Make small grids for each atom

        :type  r_cut: dict
        :param r_cut:
            dictionary of symbols (key) and cutoff radii (value)
        :rtype: tuple[list]
        :return:
            list of small_grids (list[Atoms]) and list of indices (list[torch.tensor])
        """
        small_grids = []
        small_grids_sym = self._make_small_grids_each_element(r_cuts)
        idx_matching = []
        cell = self.__grid.get_cell()
        cell_inv = cell.inverse()
        atoms_positions = torch.from_numpy(atoms.get_positions())
        gpts = self.__grid.gpts

        for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
            small_grid = copy.deepcopy(small_grids_sym[i_sym])

            R_i = atoms_positions[i_atom]
            nearest_idx = (R_i @ cell_inv * gpts).round()
            C_i = (nearest_idx @ cell) / gpts
            # small_grid.translation_to(C_i)
            small_grid.translation_to(C_i.numpy())  ##################
            small_grids.append(small_grid)

            idx_shift = (
                ((small_grid.points[0] - self.__grid.points[0]) @ cell_inv * gpts)
                .round()
                .to(int)
            )
            indices = small_grid.indices + idx_shift
            indices %= gpts  # PBC (WARNING: assume PBC, but practically not problematic because small_grid is only defined locally near to the specific atom.)
            idx_matching.append(indices)
        return small_grids, idx_matching

    @property
    def comp_charge(self):
        """Compensation charge (torch.tensor), :math:`\rho^{comp}(\bm{r})`"""
        return self.__comp_charge

    @property
    def V_short(self):
        """Short potential (torch.tensor), :math:`V^{short}(\bm{r})`"""
        return self.__V_short

    @property
    def short_cutoff(self):
        return self.__short_cutoff

    @property
    def energy_correction(self):
        return self.__energy_correction

    @property
    def r_gauss(self):
        return self.__r_gauss

    @staticmethod
    def r_gauss():
        return Compensation.__r_gauss


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    """Plot n_c, V_c, dn_c, dV_c"""
    Z = 1.0
    r_g = 1.0
    r = torch.arange(0, 5, 0.01)

    R = r / r_g

    n_c = Z * torch.exp(-(R**2)) / (sqrt(pi) * r_g) ** 3
    V_c_0 = 2 * Z / r_g / sqrt(pi)
    V_c = Z * torch.erf(R)
    V_c = torch.div(V_c, r)
    V_c[r == 0] = V_c_0

    dn_c = -2 * Z * R * torch.exp(-(R**2)) / pi**1.5 / r_g**4
    dV_c = Z * (2 / sqrt(pi) * R * torch.exp(-(R**2)) - torch.erf(R))
    dV_c = torch.div(dV_c, r**2)
    dV_c[r == 0] = 0.0

    tol = 1e-7
    val = tol * (sqrt(pi) * r_g) ** 3 / Z
    r_comp = sqrt(-log(val)) * r_g
    print("r_comp = ", r_comp)

    plt.figure()
    plt.plot(r, n_c, label="n_c")
    plt.plot(r, V_c, label="V_c")
    plt.plot(r, dn_c, label="dn_c")
    plt.plot(r, dV_c, label="dV_c")
    plt.plot(r, Z / r)
    plt.plot(r, -Z / r**2)
    plt.axhline(0)
    plt.ylim(-5, 5)
    plt.legend()
    plt.show()
