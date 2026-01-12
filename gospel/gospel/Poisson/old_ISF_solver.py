import copy
import numpy as np
import math
import torch
from scipy.special import erfcx, roots_legendre
from gospel.util import timer
from gospel.Poisson.Poisson_solver import Poisson_solver
from itertools import product


def create_ISF_exx_solver(grid, kpts=np.array([[0, 0, 0]])):
    """create ISF_exx_solver for given 'k - q' list"""
    kq = [k - q for k, q in product(kpts, kpts)]
    return ISF_exx_solver(grid, kq)


class ISF_solver(Poisson_solver):
    """
    ISF method for calculating exact-exchange (EXX) potential
    DOI: https://doi.org/10.1063/1.3291027

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  kq: array[array]
    :param kq:
        k - q
    :type  ISF_cutoff: int, optional
    :param ISF_cutoff:
        cutoff of the number of near cells
    :type  fp: str, optional
    :param fp:
        floating-point precision, options=['single', 'double'], defaults to 'double'
    """

    @timer
    def __init__(
        self,
        grid,
        kq=[[0, 0, 0]],
        ISF_cutoff=None,
        fp="double",
        num_ts=[12, 12, 12, 12],
    ):
        assert ISF_cutoff is None or isinstance(
            ISF_cutoff, int
        ), "Input variable error. ISF_cutoff is 'None' or 'int' type."
        assert grid.is_cartesian, "ISF method is only supported for cartensian grid."

        super().__init__(grid)
        self.device = torch.device("cpu")

        self.__type = "ISF"
        self.__kq = np.array(kq)
        self.__kq_x = np.unique(self.__kq[:, 0])
        self.__kq_y = np.unique(self.__kq[:, 1])
        self.__kq_z = np.unique(self.__kq[:, 2])

        self.__cell_lengths = self._grid.cell_lengths

        self.__fp = fp
        if fp == "single":
            self.__float, self.__complex = torch.float32, torch.complex64
        elif fp == "double":
            self.__float, self.__complex = torch.float64, torch.complex128
        else:
            raise NotImplementedError("available 'fp' is 'single' or 'double'.")

        ## Set n_cutoff ##
        self.__n_cutoff = ISF_cutoff
        if self.__n_cutoff != None:
            self.__n_cutoff = np.full(3, self.__n_cutoff) * self._pbc
        else:
            if np.sum(self._pbc) == 3:  # P3D
                self.__n_cutoff = np.full(3, 100)
            else:
                self.__n_cutoff = np.round(2000 / self.__cell_lengths).astype(int)
                # Set the minimum value of n_cutoff as 100.
                self.__n_cutoff[self.__n_cutoff < 100] = 100
                self.__n_cutoff *= self._pbc
        print(f"ISF_cutoff is set to {self.__n_cutoff}.")

        ## t sampling ##
        """
        Divide 't (0 ~ inf)' with 4 ranges: [t_i, t_1], [t_1, t_2], [t_2, t_3], [t_3, t_f]
        The number of sampled points for each range is 'num_ts'.
        """
        if np.sum(self._pbc) == 3:  # P3D
            t_i = 0.05
        else:  # P0D, P1D, P2D
            t_i = 0.0
        num_t_1, num_t_2, num_t_3, num_t_4 = num_ts
        # t_1, t_2, t_3, t_f = 0.2, 0.4, 2.0, 500.0
        t_1, t_2, t_3, t_f = 0.2, 0.4, 2.0, 2000.0

        self.__t_f = t_f
        if np.sum(self._pbc) == 0:  # P0D
            t_1, num_t_1 = 0.0, 0
        self.__total_num_t = num_t_1 + num_t_2 + num_t_3 + num_t_4

        ## Linear sampling
        t_val_1, w_t_1 = legendre_t_sampling(t_i, t_1, num_t_1)
        t_val_2, w_t_2 = legendre_t_sampling(t_1, t_2, num_t_2)
        t_val_3, w_t_3 = legendre_t_sampling(t_2, t_3, num_t_3)
        ## Logarithmic sampling
        t_val_4, w_t_4 = legendre_t_sampling(np.log(t_3), np.log(t_f), num_t_4)
        t_val_4 = torch.exp(t_val_4)
        ## Concatenate t values and weights.
        t_val = torch.cat([t_val_1, t_val_2, t_val_3, t_val_4])
        self.__w_t = torch.cat([w_t_1, w_t_2, w_t_3, w_t_4 * t_val_4]) * (
            2.0 / math.sqrt(math.pi)
        )
        ## Construct F matrices ##
        self.__F_xs_kq = self._construct_F_matrix(t_val, self.__kq_x, axis=0)
        self.__F_ys_kq = self._construct_F_matrix(t_val, self.__kq_y, axis=1)
        self.__F_zs_kq = self._construct_F_matrix(t_val, self.__kq_z, axis=2)

        ## single precision (It can be set from 'self.set_single_precision()')##
        self.__use_sp = False  # whether using single precision
        self.__w_t_sp = None
        self.__F_xs_kq_sp = None
        self.__F_ys_kq_sp = None
        self.__F_zs_kq_sp = None

        print("ISF method is used for Poisson")
        print("* [{}, {}] : {} points".format(t_i, t_1, num_t_1))
        print("* [{}, {}] : {} points".format(t_1, t_2, num_t_2))
        print("* [{}, {}] : {} points".format(t_2, t_3, num_t_3))
        print("* [{}, {}] : {} points".format(t_3, t_f, num_t_4))
        print(f"t_val = {t_val}")
        return

    def __str__(self):
        s = str()
        s += "\n=========================== [ ISF_solver ] ============================"
        s += f"\n* type          : {self.__type}"
        s += f"\n* kq            : \n{self.__kq}"
        s += f"\n* ISF_cutoff    : {self.__n_cutoff}"
        s += f"\n* fp            : {self.__fp}"
        s += f"\n* use_sp        : {self.__use_sp}"
        s += "\n=======================================================================\n"
        return s

    def _compute_F(self, t, x_bar, spacing, cell_length, n_cutoff, kq, dtype):
        """Compute F matrices

        :type  t: int
        :param t:
            t value
        :type  x_bar: torch.Tensor
        :param x_bar:
            grid points on an axis
        :type  spacing: float
        :param spacing:
            grid spacing
        :type  cell_length: float
        :param cell_length:
            length of one side of the cell
        :type  n_cutoff: int
        :param n_cutoff:
            the number of neighboring cells to be summed when PBC
        :type  kq: float
        :param kq:
            k-point vector. (k - q)
        :type  dtype: torch.dtype
        :param dtype:
            data type of F matrices

        :rtype: torch.Tensor
        :param:
            F matrix, shape=(Nx, Nx) where Nx is the number of grid points of one axis
        """
        if kq == 0:
            x_diff = x_bar.repeat(2 * n_cutoff + 1, 1)
            x_diff += cell_length * torch.arange(-n_cutoff, n_cutoff + 1).reshape(-1, 1)
            z = torch.pi / 2 / spacing / t + t * x_diff * 1j
            tmp = -((t * x_diff) ** 2)
            ret_val = torch.exp(tmp)
            ret_val -= (torch.exp(tmp - z**2) * erfcx(z)).real
            F = ret_val.sum(0) * spacing
        else:
            x_diff = x_bar.repeat(2 * n_cutoff + 1, 1)
            x_diff += cell_length * torch.arange(-n_cutoff, n_cutoff + 1).reshape(-1, 1)
            z1 = (torch.pi + spacing * kq) / 2 / spacing / t - t * x_diff * 1j
            z2 = (torch.pi - spacing * kq) / 2 / spacing / t + t * x_diff * 1j
            tmp = -(t**2) * x_diff**2 - kq * x_diff * 1j
            ret_val = 2 * torch.exp(tmp)
            ret_val -= torch.exp(tmp - z1**2) * erfcx(z1) + torch.exp(
                tmp - z2**2
            ) * erfcx(z2)
            F = ret_val.sum(0) * spacing * 0.5
        return F

    @timer
    def _construct_F_matrix(self, t_val, kq, axis):
        from scipy.sparse import diags

        t_size = len(t_val)
        points = np.mgrid[0 : self.__cell_lengths[axis] : self._spacings[axis]][
            : self._gpts[axis]
        ]
        points = torch.from_numpy(points)

        F_kq = np.empty(len(kq), dtype=object)
        for i_k in range(len(kq)):
            dtype = self.__float if kq[i_k] == 0 else self.__complex
            F_kq[i_k] = torch.zeros(
                t_size, self._gpts[axis], self._gpts[axis], dtype=dtype
            )
            for i_t in range(t_size):
                F_tmp = self._compute_F(
                    t_val[i_t],
                    points,
                    self._spacings[axis],
                    self.__cell_lengths[axis],
                    self.__n_cutoff[axis],
                    kq[i_k],
                    dtype,
                ).numpy()
                F_kq[i_k][i_t] = torch.from_numpy(
                    diags(
                        np.append(F_tmp[::-1].conj(), F_tmp[1:]),
                        range(-self._gpts[axis] + 1, self._gpts[axis]),
                        shape=(self._gpts[axis], self._gpts[axis]),
                    ).toarray()
                )
        return F_kq

    ## for testing speed comparison
    def numpy_compute_potential(self, density):
        assert isinstance(density, np.ndarray)

        potential = np.zeros(self._gpts, dtype=density.dtype)
        rho = density.reshape(self._gpts)

        kq = [0, 0, 0]
        i_kq_x = np.argwhere(self.__kq_x == kq[0]).item()
        i_kq_y = np.argwhere(self.__kq_y == kq[1]).item()
        i_kq_z = np.argwhere(self.__kq_z == kq[2]).item()

        st = time()
        for F_x, F_y, F_z, w in zip(
            self.__F_xs_kq[i_kq_x],
            self.__F_ys_kq[i_kq_y],
            self.__F_zs_kq[i_kq_z],
            self.__w_t,
        ):
            tmp = np.tensordot(rho, F_x, ([0], [0]))
            tmp = np.tensordot(tmp, F_y, ([0], [0]))
            tmp = np.tensordot(tmp, F_z, ([0], [0]))
            potential += w * tmp
        potential = potential.reshape(-1)
        potential += density / self.__t_f**2
        print(f" numpy_compute_potential Time : {time()-st}")
        return potential

    @timer
    def child_compute_potential(self, density, kq=[0, 0, 0]):
        """Batch version of 'child_compute_potential()'

        :dtype  density: torch.Tensor
        :param  density:
            shape=(nbands, ngpts) or (ngpts,)
        :dtype  kq: list or np.ndarray, optional
        :param  kq:
            k-point vector (k - q), defaults to [0, 0, 0]

        :rtype: torch.Tensor
        :return:
            poisson potential, shape==density.shape
        """
        assert isinstance(density, torch.Tensor)
        ndim = density.ndim
        assert ndim in [1, 2]
        if ndim == 1:
            density = density.unsqueeze(dim=0)
        i_kq_x = np.argwhere(self.__kq_x == kq[0]).item()
        i_kq_y = np.argwhere(self.__kq_y == kq[1]).item()
        i_kq_z = np.argwhere(self.__kq_z == kq[2]).item()
        if self.__use_sp:
            assert self.__w_t_sp is not None
            complex_type = [torch.complex64, torch.complex128]
            float_type = [torch.float32, torch.float64]
            if density.dtype in complex_type:
                sp_dtype = torch.complex64
            elif density.dtype in float_type:
                sp_dtype = torch.float32
            else:
                raise TypeError
            _rho = density.to(sp_dtype)
            F_xs_list = self.__F_xs_kq_sp[i_kq_x]
            F_ys_list = self.__F_ys_kq_sp[i_kq_y]
            F_zs_list = self.__F_zs_kq_sp[i_kq_z]
            w_t_list = self.__w_t_sp
        else:
            _rho = density
            F_xs_list = self.__F_xs_kq[i_kq_x]
            F_ys_list = self.__F_ys_kq[i_kq_y]
            F_zs_list = self.__F_zs_kq[i_kq_z]
            w_t_list = self.__w_t

        ## density.shape = (nkpts * nbands,)---(ngpts,)
        # potential = np.empty(len(density), dtype=object)  # shape=(nbands,)---(ngpts,)
        if not _rho.is_contiguous():
            _rho = _rho.contiguous()
        potential = torch.zeros_like(density)

        for i in range(len(_rho)):
            rho = _rho[i].reshape(*self._gpts)
            for F_x, F_y, F_z, w in zip(F_xs_list, F_ys_list, F_zs_list, w_t_list):
                ## ijk, ia, jb, kc -> abc
                tmp = torch.tensordot(rho, F_x, ([0], [0]))
                tmp = torch.tensordot(tmp, F_y, ([0], [0]))
                tmp = torch.tensordot(tmp, F_z, ([0], [0]))
                potential[i] += w * tmp.reshape(-1)
        potential += density / self.__t_f**2
        if ndim == 1:
            potential = potential.squeeze()
        return potential

    @timer
    def batch_compute_potential(self, density, kq=[0, 0, 0]):
        """Batch version2 of 'child_compute_potential()'

        :dtype  density: torch.Tensor
        :param  density:
            shape=(nbands, ngpts)
        """
        assert isinstance(density, torch.Tensor)
        assert density.ndim == 2  # (nbands, ngpts)

        rho = density.reshape(-1, *self._gpts)
        nbands = len(density)
        if not density.is_contiguous():
            density = density.contiguous()
        potential = torch.zeros_like(density)  # shape=(nbands, ngpts)

        i_kq_x = np.argwhere(self.__kq_x == kq[0]).item()
        i_kq_y = np.argwhere(self.__kq_y == kq[1]).item()
        i_kq_z = np.argwhere(self.__kq_z == kq[2]).item()

        for F_x, F_y, F_z, w in zip(
            self.__F_xs_kq[i_kq_x],
            self.__F_ys_kq[i_kq_y],
            self.__F_zs_kq[i_kq_z],
            self.__w_t,
        ):
            ## ijkl, ja, kb, lc -> iabc
            tmp = torch.tensordot(rho, F_x, ([1], [0]))
            tmp = torch.tensordot(tmp, F_y, ([1], [0]))
            tmp = torch.tensordot(tmp, F_z, ([1], [0]))
            potential += w * tmp.reshape(nbands, -1)
        potential += density / self.__t_f**2
        return potential

    def plot_F_t(self, density, t_list, kq=[0, 0, 0]):
        ## construct F matrices
        F_xs = self._construct_F_matrix(t_list, [kq[0]], axis=0)[0]
        F_ys = self._construct_F_matrix(t_list, [kq[1]], axis=1)[0]
        F_zs = self._construct_F_matrix(t_list, [kq[2]], axis=2)[0]

        ## Choice datatype
        dtype_single = [torch.float32, torch.complex64]
        dtype_double = [torch.float64, torch.complex128]
        if density.dtype in dtype_single:
            F_xs = to_sp(F_xs)
            F_ys = to_sp(F_ys)
            F_zs = to_sp(F_zs)
            _complex = torch.complex64
        elif density.dtype in dtype_double:
            _complex = torch.complex128
        else:
            raise TypeError

        ## calculate F(t)
        rho = density.reshape(*self._gpts)
        F_t = torch.zeros(len(t_list), dtype=torch.complex128)
        for t, (F_x, F_y, F_z) in enumerate(zip(F_xs, F_ys, F_zs)):
            if _complex in [rho.dtype, F_x.dtype, F_y.dtype, F_z.dtype]:
                rho = rho.to(_complex)
                F_x = F_x.to(_complex)
                F_y = F_y.to(_complex)
                F_z = F_z.to(_complex)
            V_t = torch.tensordot(
                torch.tensordot(torch.tensordot(rho, F_x, ([0], [0])), F_y, ([0], [0])),
                F_z,
                ([0], [0]),
            )
            F_t[t] = self._grid.integrate(density.conj() * V_t.reshape(-1))
        return F_t

    def set_device(self, device=None):
        """Changes the device of F matrices and the corresponding weights."""
        for i in range(len(self.__F_xs_kq)):
            self.__F_xs_kq[i] = self.__F_xs_kq[i].to(device)
        for i in range(len(self.__F_ys_kq)):
            self.__F_ys_kq[i] = self.__F_ys_kq[i].to(device)
        for i in range(len(self.__F_zs_kq)):
            self.__F_zs_kq[i] = self.__F_zs_kq[i].to(device)
        self.__w_t = self.__w_t.to(device)
        self.device = device
        return

    def set_single_precision(self):
        self.__w_t_sp = self.__w_t.to(torch.float32)
        self.__F_xs_kq_sp = copy.deepcopy(self.__F_xs_kq)
        self.__F_ys_kq_sp = copy.deepcopy(self.__F_ys_kq)
        self.__F_zs_kq_sp = copy.deepcopy(self.__F_zs_kq)

        for i in range(len(self.__F_xs_kq_sp)):
            self.__F_xs_kq_sp[i] = to_sp(self.__F_xs_kq_sp[i])
        for i in range(len(self.__F_ys_kq)):
            self.__F_ys_kq_sp[i] = to_sp(self.__F_ys_kq_sp[i])
        for i in range(len(self.__F_zs_kq)):
            self.__F_zs_kq_sp[i] = to_sp(self.__F_zs_kq_sp[i])
        self.__use_sp = True
        return

    @property
    def w_t(self):
        return self.__w_t

    @property
    def F_xs_kq(self):
        return self.__F_xs_kq

    @property
    def F_ys_kq(self):
        return self.__F_ys_kq

    @property
    def F_zs_kq(self):
        return self.__F_zs_kq

    @property
    def use_sp(self):
        return self.__use_sp

    @use_sp.setter
    def use_sp(self, inp):
        assert isinstance(inp, bool)
        self.__use_sp = inp
        return


def to_sp(x):
    # double precision to single precision
    if x.dtype == torch.float64:
        return x.to(torch.float32)
    elif x.dtype == torch.complex128:
        return x.to(torch.complex64)
    else:
        raise TypeError


def legendre_t_sampling(a, b, n):
    """Legendre sampling.

    :type  a: float
    :param a: lower limit of integration
    :type  b: float
    :param b: upper limit of integration
    :type  n: int
    :param n: Number of points to sample
    :rtype: list[float], list[float]
    :return:
        list of t-values and list of corresponding weights
    """
    assert n >= 0
    if n == 0:
        return torch.zeros(0), torch.zeros(0)
    t_val, w_t = roots_legendre(n)
    t_val *= (b - a) / 2
    t_val += (b + a) / 2
    w_t *= (b - a) / 2
    return torch.from_numpy(t_val), torch.from_numpy(w_t)


if __name__ == "__main__":
    from gospel.Poisson.Poisson_ISF_python import Poisson_ISF
    from gospel.Grid import Grid
    from ase.build import bulk
    from ase.units import Bohr

    gpts = (100, 100, 100)
    fp = "double"

    atoms = bulk("C", "diamond", a=3.57 / Bohr, cubic=True)
    atoms.set_pbc([True, False, False])
    grid = Grid(atoms, gpts)
    print(grid)

    rho = np.exp(-np.linalg.norm(grid.points - grid.center_point, axis=-1) ** 2)
    rho /= np.sum(rho)
    rho -= np.ones_like(rho) / len(rho)  # Make neutral to converge
    rho = torch.from_numpy(rho)

    kq = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kq = kq @ atoms.cell.reciprocal() * 2 * np.pi  # k/(2*pi*L)
    print(f"kq = \n{kq}")

    solver = ISF_solver(grid, kq, fp=fp)
    print(solver)

    ## Plot F(t)
    import matplotlib.pyplot as plt

    for i in range(len(kq)):
        logt_list = np.arange(-3, 4, 0.1)
        t_list = 10**logt_list
        F_t = solver.plot_F_t(rho, t_list, kq[i])
        F_t_sp = solver.plot_F_t(rho.to(torch.float32), t_list, kq[i])
        print(f"logt_list = {logt_list}")
        print(f"F_t = {F_t}")

        plt.figure()
        plt.title(f"F(t) for k-q={kq[i]}")
        plt.plot(t_list, F_t, label="F(t)")
        plt.plot(t_list, F_t_sp, label="F(t)_sp")
        plt.xlabel("t")
        plt.ylabel("F(t)")
        plt.loglog()
        plt.legend()
        # plt.loglog(basex=10, basey=10)
    plt.show()
