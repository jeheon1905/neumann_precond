from typing import List, Optional
import math
import numpy as np
from scipy.special import erfcx, roots_legendre
import torch

from gospel import Grid
from gospel.Poisson.Poisson_solver import Poisson_solver
from gospel.precision import to_DP, to_SP, to_HP, to_BF16, convert_dtype
from gospel.util import Timer


# TODO: Does not support complex dtype yet.
class ISF_solver(Poisson_solver):
    """ISF Poisson solver.

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  t_sample: list or None
    :param t_sample:
        list of t values and weights and t_delta
    :type  k_vec: list or np.ndarray
    :param k_vec:
        k vector of the operator: (\nabla^2 + k^2), defaults to [0, 0, 0]
    :type  fp: str
    :param fp:
        floating-point precision level, options=["SP", "DP"], defaults to "DP"
    :type  init_F: bool
    :param init_F:
        whether initializing F matrices or not, defaults to True
    :type  device: torch.device
    :param device:
        device to run the computation on, defaults to torch.device("cpu")
    :type  use_MP: bool
    :param use_MP:
        whether using mixed precision or not, defaults to False
        F matrix construction is done in double precision.
    """

    def __init__(
        self,
        grid: Grid,
        t_sample: Optional[List] = None,
        k_vec: List[int] = [0, 0, 0],
        fp: str = "DP",
        init_F: bool = True,
        device: torch.device = torch.device("cpu"),
        use_MP: bool = False,
    ):
        assert grid.is_cartesian, "ISF_solver is only supported for cartesian grid."
        super().__init__(grid)
        self.__type = "ISF"
        self.__k_vec = k_vec
        self.__fp = fp
        self.__device = device
        self.__use_MP = use_MP

        if fp == "DP":
            self.__dtype_real, self.__dtype_complex = torch.float64, torch.complex128
        elif fp == "SP":
            self.__dtype_real, self.__dtype_complex = torch.float32, torch.complex64
        elif fp == "HP":
            self.__dtype_real, self.__dtype_complex = torch.float16, torch.complex32
        elif fp == "BF16":
            self.__dtype_real, self.__dtype_complex = torch.float16, None
        else:
            raise NotImplementedError("available 'fp' is one of ['DP', 'SP', 'HP', 'BF16].")

        # Sample t values, weights, and t_delta
        if t_sample is None:
            self.__t_val, self.__wts, self.__t_delta = t_sampling(grid.get_pbc())
        else:
            self.__t_val = torch.tensor(t_sample[0]).to(device, self.__dtype_real)
            self.__wts = torch.tensor(t_sample[1]).to(device, self.__dtype_real)
            self.__t_delta = t_sample[2]
            assert len(self.__t_val) == len(self.__wts)

        # Construct F matrices
        self.initialized = False
        if init_F:
            self.initialize(use_MP=use_MP)
        else:
            self.__Fx, self.__Fy, self.__Fz = None, None, None

        self.__Fx = self.__Fx.to(device)
        self.__Fy = self.__Fy.to(device)
        self.__Fz = self.__Fz.to(device)
        self.__wts = self.__wts.to(device)
        return

    def __str__(self):
        s = str()
        s += "\n=========================== [ ISF_solver ] =========================="
        s += f"\n* type          : {self.__type}"
        s += f"\n* k_vec         : {self.__k_vec}"
        s += f"\n* device        : {self.__device}"
        s += f"\n* use_MP        : {self.__use_MP}"
        s += f"\n* fp            : {self.__fp}"
        s += f"\n* pbc           : {self._pbc}"
        if self.initialized:
            s += f"\n\n[Print sampled t values]"
            s += f"\n* t_val  : {self.__t_val}"
            s += f"\n* wts    : {self.__wts}"
            s += f"\n* t_delta: {self.__t_delta}"
        s += "\n=====================================================================\n"
        return s

    def child_compute_potential(self, density):
        """Compute potentials

        :dtype  density: torch.Tensor
        :param  density:
            shape=(nbands, ngpts) or (ngpts,)

        :rtype: torch.Tensor
        :return:
            poisson potential, shape==density.shape
        """
        assert isinstance(density, torch.Tensor)

        ndim = density.ndim
        assert ndim in [1, 2]
        if ndim == 1:
            density = density.unsqueeze(dim=0)
        assert self.initialized, "F matrices are not initialized."
        _rho = density
        Fxs, Fys, Fzs, wts = self.__Fx, self.__Fy, self.__Fz, self.__wts

        if not _rho.is_contiguous():
            _rho = _rho.contiguous()
        # potential = torch.zeros_like(density)
        potential = density / (self.__t_delta**2 / math.pi)
        for i in range(len(_rho)):
            rho = _rho[i].reshape(*self._gpts)
            for Fx, Fy, Fz, wt in zip(Fxs, Fys, Fzs, wts):
                ## ijk, ia, jb, kc -> abc
                tmp = torch.tensordot(rho, Fx, ([0], [0]))
                tmp = torch.tensordot(tmp, Fy, ([0], [0]))
                tmp = torch.tensordot(tmp, Fz, ([0], [0]))
                potential[i] += wt * tmp.reshape(-1)
        # potential += density / (self.__t_delta**2 / math.pi)
        if ndim == 1:
            potential = potential.squeeze()
        return potential

    # @timer
    @Timer.timeit
    def batch_compute_potential(self, density, kq=[0, 0, 0]):
        """
        Batch version of 'child_compute_potential()'.
        It can be more faster than 'child_compute_potential()' in GPU and small grid size.

        :dtype  density: torch.Tensor
        :param  density:
            shape=(nbands, ngpts)
        """
        assert isinstance(density, torch.Tensor)
        assert density.ndim == 2  # (nbands, ngpts)
        assert self.initialized, "F matrices are not initialized."
        _rho = density
        Fxs, Fys, Fzs, wts = self.__Fx, self.__Fy, self.__Fz, self.__wts

        nbands = len(density)
        if not _rho.is_contiguous():
            _rho = density.contiguous()
        potential = torch.zeros_like(density)  # shape=(nbands, ngpts)

        rho = _rho.reshape(-1, *self._gpts)
        for Fx, Fy, Fz, wt in zip(Fxs, Fys, Fzs, wts):
            ## ijkl, ja, kb, lc -> iabc
            tmp = torch.tensordot(rho, Fx, ([1], [0]))
            tmp = torch.tensordot(tmp, Fy, ([1], [0]))
            tmp = torch.tensordot(tmp, Fz, ([1], [0]))
            potential += wt * tmp.reshape(nbands, -1)
        potential += density / (self.__t_delta**2 / math.pi)
        return potential

    # @timer
    @Timer.timeit
    def batch_compute_potential2(self, density, kq=[0, 0, 0]):
        """
        Batch version of 'child_compute_potential()'.
        It can be more faster than 'child_compute_potential()' in GPU and small grid size.

        :dtype  density: torch.Tensor
        :param  density:
            shape=(ngpts, nbands)

        :rtype: torch.Tensor
        :return:
            potential, shape=(ngpts, nbands)
        """
        assert isinstance(density, torch.Tensor)
        assert density.ndim == 2  # (nbands, ngpts)
        assert self.initialized, "F matrices are not initialized."
        _rho = density
        Fxs, Fys, Fzs, wts = self.__Fx, self.__Fy, self.__Fz, self.__wts

        nbands = density.shape[-1]
        ngpts, nbands = density.shape
        if not _rho.is_contiguous():
            _rho = density.contiguous()
        potential = torch.zeros_like(density)  # shape=(ngpts, nbands)

        rho = _rho.reshape(*self._gpts, nbands)
        for Fx, Fy, Fz, wt in zip(Fxs, Fys, Fzs, wts):
            ## ijkl, ia, jb, kc -> abcl
            tmp = torch.tensordot(rho, Fx, ([0], [0]))
            tmp = torch.tensordot(tmp, Fy, ([0], [0]))
            tmp = torch.tensordot(tmp, Fz, ([0], [0]))
            potential += wt * tmp.reshape(nbands, -1).T
        potential += density / (self.__t_delta**2 / math.pi)
        return potential

    def initialize(self, use_MP: bool = False) -> None:
        """Initialize F matrices"""
        if self.initialized:
            print("ISF_solver is already initialized.")
        else:
            # NOTE: F matrices are constructed in DP when use_MP is True.
            fp = "DP" if use_MP else self.__fp
            t_val, k_vec = self.__t_val, self.__k_vec
            self.__Fx = construct_F(
                self.__t_val, self._grid, 0, fp, self.__k_vec[0], self.__device
            )
            self.__Fy = construct_F(
                self.__t_val, self._grid, 1, fp, self.__k_vec[1], self.__device
            )
            self.__Fz = construct_F(
                self.__t_val, self._grid, 2, fp, self.__k_vec[2], self.__device
            )
            self.initialized = True

        # MP: dtype change after constructing F matrices
        if use_MP:
            to_FP = {"SP": to_SP, "HP": to_HP, "DP": to_DP, "BF16": to_BF16}[self.__fp]
            self.__wts = self.__wts.to(dtype=to_FP(self.__wts.dtype))
            self.__Fx = self.__Fx.to(dtype=to_FP(self.__Fx.dtype))
            self.__Fy = self.__Fy.to(dtype=to_FP(self.__Fy.dtype))
            self.__Fz = self.__Fz.to(dtype=to_FP(self.__Fz.dtype))
        return

    def set_F(self, Fx, Fy, Fz):
        """Set F matrices."""
        self.__Fx = Fx.clone()
        self.__Fy = Fy.clone()
        self.__Fz = Fz.clone()
        self.initialized = True
        return

    def plot_F_t(self, density, t_list, k_vec=[0, 0, 0]):
        raise NotImplementedError

    @property
    def wts(self):
        return self.__wts

    @property
    def Fx(self):
        return self.__Fx

    @property
    def Fy(self):
        return self.__Fy

    @property
    def Fz(self):
        return self.__Fz


def t_sampling(pbc=[False, False, False]):
    """Sample t values according to S. A. Losilla and D. Sundholm's JCP paper. (doi: 10.1063/1.3291027)"""
    if np.sum(pbc) == 3:
        t_i = 0.05
    else:
        t_i = 0.0
    if np.sum(pbc) == 0:
        num_t_1, num_t_2, num_t_3, num_t_4 = 0, 12, 12, 12
        t_1, t_2, t_3, t_f = 0.0, 0.4, 2.0, 500.0
    else:
        num_t_1, num_t_2, num_t_3, num_t_4 = 12, 12, 12, 12
        t_1, t_2, t_3, t_f = 0.2, 0.4, 2.0, 500.0

    ## Linear sampling
    t_val_1, w_t_1 = legendre_t_sampling(t_i, t_1, num_t_1)
    t_val_2, w_t_2 = legendre_t_sampling(t_1, t_2, num_t_2)
    t_val_3, w_t_3 = legendre_t_sampling(t_2, t_3, num_t_3)
    ## Logarithmic sampling
    t_val_4, w_t_4 = legendre_t_sampling(np.log(t_3), np.log(t_f), num_t_4)
    t_val_4 = torch.exp(t_val_4)
    ## Concatenate t values and weights.
    t_val = torch.cat([t_val_1, t_val_2, t_val_3, t_val_4])
    w_t = torch.cat([w_t_1, w_t_2, w_t_3, w_t_4 * t_val_4]) * (2.0 / math.sqrt(math.pi))
    return t_val, w_t, t_f


def construct_F(
    t_val: torch.Tensor,
    grid: Grid,
    axis: int,
    precision: str,
    kx: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Construct F matrices for the given axis

    :type  t_val: torch.Tensor
    :param t_val:
        sampled t values
    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  axis: int
    :param axis:
        index of axis, available value is 0, 1, or 2
    :type  precision: str
    :param precision:
        floating-point precision level, options=["DP", "SP", "HP"]
    :type  kx: float
    :param kx:
        k vector component for the given axis.

    :rtype: torch.Tensor
    :return:
        F matrices, shape=(Nt, Nx, Nx)
    """
    Lx = grid.cell_lengths[axis]
    hx = grid.spacings[axis]
    Nx = grid.gpts[axis]
    pbc = grid.get_pbc()

    ## Set n_cutoff
    if np.sum(pbc) == 3:
        n_cutoff = 100
    else:
        n_cutoff = int(np.round(2000 / Lx))
        n_cutoff = max(n_cutoff, 100)  # Set minimum of n_cutoff as 100.
        n_cutoff *= pbc[axis]
    print(f"n_cutoff(axis={axis}) is set to {n_cutoff}.")

    _dtype = torch.float64 if kx == 0 else torch.complex128
    _dtype = convert_dtype(_dtype, precision)

    ## Construct F matrices
    points = torch.from_numpy(np.mgrid[0:Lx:hx][:Nx]).to(device, _dtype)
    F_t = torch.zeros(len(t_val), Nx, Nx, dtype=_dtype, device=device)
    for i_t in range(len(F_t)):
        F_tmp = compute_F(t_val[i_t], points, hx, Lx, n_cutoff, kx)
        F_tmp = torch.cat([F_tmp.flip(0).conj(), F_tmp[1:]])
        F_t[i_t] = torch.stack([F_tmp[Nx - 1 - i : Nx - 1 - i + Nx] for i in range(Nx)])
    return F_t


def compute_F(t, x_bar, spacing, cell_length, n_cutoff, kx):
    """Compute analytic solution of integral of product of Sinc and Gaussian.

    :type  t: int
    :param t:
        t value
    :type  x_bar: torch.Tensor
    :param x_bar:
        grid points on an axis, shape=(Nx,)
    :type  spacing: float
    :param spacing:
        grid spacing
    :type  cell_length: float
    :param cell_length:
        length of one side of the cell
    :type  n_cutoff: int
    :param n_cutoff:
        the number of neighboring cells to be summed when PBC
    :type  kx: float
    :param kx:
        k vector component for the given axis.

    :rtype: torch.Tensor
    :return:
        F matrix, shape=(Nx,) where Nx is the number of grid points of the given axis
    """
    # erfcx = torch.special.erfcx  # no support complex variable...
    # scipy.special.erfcx do not support cuda.

    device = x_bar.device
    x_diff = x_bar.repeat(2 * n_cutoff + 1, 1)
    x_diff += cell_length * torch.arange(
        -n_cutoff, n_cutoff + 1, device=device
    ).reshape(-1, 1)

    if kx == 0:
        z = torch.pi / 2 / spacing / t + t * x_diff * 1j
        tmp = -((t * x_diff) ** 2)
        ret_val = torch.exp(tmp)
        ret_val -= (torch.exp(tmp - z**2) * erfcx(z.cpu()).to(device)).real
        F = ret_val.sum(0) * spacing
    else:
        z1 = (torch.pi + spacing * kx) / 2 / spacing / t - t * x_diff * 1j
        z2 = (torch.pi - spacing * kx) / 2 / spacing / t + t * x_diff * 1j
        tmp = -(t**2) * x_diff**2 - kx * x_diff * 1j
        ret_val = 2 * torch.exp(tmp)
        ret_val -= torch.exp(tmp - z1**2) * erfcx(z1.cpu()).to(device) + torch.exp(
            tmp - z2**2
        ) * erfcx(z2.cpu()).to(device)
        F = ret_val.sum(0) * spacing * 0.5
    return F


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


def create_ISF_solvers(grid, t_sample=None, fp="DP", kpts=np.array([[0, 0, 0]])):
    """Create ISF solvers for given k vectors. (Not tested!!!!!)

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  t_sample: list or None
    :param t_sample:
        list of t values and weights and t_delta
    :type  fp: str
    :param fp:
        floating-point precision level, options=["SP", "DP"], defaults to "DP"
    :type  kpts: list or np.ndarray
    :param kpts:
        k vectors of the operator: (\nabla^2 + k^2), shape=(nkpts, 3)

    :rtype: list[gospel.Poisson.ISF_solver]
    :return:
        list of ISF solvers
    """
    raise NotImplementedError

    if t_sample is None:
        t_val, wts, t_delta = t_sampling(grid.get_pbc())
    else:
        t_val = torch.tensor(t_sample[0])  # , dtype=self.__float)
        wts = torch.tensor(t_sample[1])  # , dtype=self.__float)
        t_delta = t_sample[2]

    kxs, kys, kzs = np.unique(kpts[:, 0]), np.unique(kpts[:, 1]), np.unique(kpts[:, 2])
    Fxs = np.empty(len(kxs), dtype=object)
    Fys = np.empty(len(kys), dtype=object)
    Fzs = np.empty(len(kzs), dtype=object)
    for i, kx in enumerate(kxs):
        Fxs[i] = construct_F(t_val, grid, 0, fp, kx)
    for i, ky in enumerate(kys):
        Fys[i] = construct_F(t_val, grid, 1, fp, ky)
    for i, kz in enumerate(kzs):
        Fzs[i] = construct_F(t_val, grid, 2, fp, kz)

    solvers = []
    for kpt in kpts:
        solver = ISF_solver(grid, k_vec=kpt, init_F=False)
        i_kx = np.argwhere(kxs == kpt[0]).item()
        i_ky = np.argwhere(kys == kpt[1]).item()
        i_kz = np.argwhere(kzs == kpt[2]).item()
        solver.set_F(Fxs[i_kx], Fys[i_ky], Fzs[i_kz])
        solvers.append(solver)
    return solvers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ISF solver speed test for SP, MP, and DP."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the computation on: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="set number of threads (default: 1)"
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    import time
    import torch
    from gospel.Grid import Grid
    from ase.build import molecule, bulk

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    device = args.device

    atoms = bulk("Si", "diamond", cubic=True, a=5.43)
    # grid = Grid(atoms, spacing=0.1)
    grid = Grid(atoms, spacing=0.07)  # NOTE: CNT_3_4
    # grid = Grid(atoms, spacing=0.05)  # NOTE: CNT_3_4
    print(grid)

    DP = [torch.float64, torch.complex128][0]
    SP = [torch.float32, torch.complex64][0]
    HP = [torch.float16, torch.complex32][0]
    k_vec = [[0, 0, 0], [0, 0, 1]][0]

    t_sample = t_sampling(grid.get_pbc())

    # NOTE: Nan values occur when the ISF_solver is constructed with SP. (device=cuda)
    solver_SP = ISF_solver(
        grid, t_sample, k_vec=k_vec, fp="SP", init_F=True, device=device
    )
    solver_DP = ISF_solver(
        grid, t_sample, k_vec=k_vec, fp="DP", init_F=True, device=device
    )
    # NOTE: MP works well because F matrices are constructed with DP.
    solver_MP_SP = ISF_solver(
        grid, t_sample, k_vec=k_vec, fp="SP", init_F=True, device=device, use_MP=True
    )
    solver_MP_HP = ISF_solver(
        grid, t_sample, k_vec=k_vec, fp="HP", init_F=True, device=device, use_MP=True
    )

    nbands = 100
    nbands = 310  # NOTE: CNT_3_4
    rho_DP = torch.randn(nbands, grid.ngpts, dtype=DP)
    rho_DP -= rho_DP.mean(-1).reshape(-1, 1)
    print(f"rho_DP.shape = {rho_DP.shape}")
    # nelec = 1000.0
    nelec = 0.0
    rho_DP += nelec / grid.ngpts
    # print(f"Debug: rho_DP.sum(-1) = {rho_DP.sum(-1)}")
    rho_SP = rho_DP.to(SP)
    rho_HP = rho_DP.to(HP)

    if device == "cuda":
        rho_DP = rho_DP.to(device)
        rho_SP = rho_SP.to(device)
        rho_HP = rho_HP.to(device)

        for i in range(10):
            _ = rho_DP @ rho_DP.T
        print("Warm-up Finished.")

    from time import time

    def cpu_timer(func):
        def wrapper(*args, **kwargs):
            st = time()
            result = func(*args, **kwargs)
            print(f"CPU Time: {time() - st} sec")
            return result

        return wrapper

    def cuda_timer(func):
        def wrapper(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            result = func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            print(f"cuda Time: {start.elapsed_time(end) / 1000} sec")
            return result

        return wrapper

    timer = cuda_timer if device == "cuda" else cpu_timer

    V_SP = timer(solver_SP.child_compute_potential)(rho_SP)
    V_MP_SP = timer(solver_MP_SP.child_compute_potential)(rho_SP)
    V_MP_HP = timer(solver_MP_HP.child_compute_potential)(rho_HP)
    V_DP = timer(solver_DP.child_compute_potential)(rho_DP)
    V_SP_batch = timer(solver_SP.batch_compute_potential)(rho_SP)
    V_MP_SP_batch = timer(solver_MP_SP.batch_compute_potential)(rho_SP)
    V_MP_HP_batch = timer(solver_MP_HP.batch_compute_potential)(rho_HP)
    V_DP_batch = timer(solver_DP.batch_compute_potential)(rho_DP)
    V_MP_SP = V_MP_SP.double()
    V_MP_HP = V_MP_HP.double()
    V_SP = V_SP.double()

    print("=============================================")
    print(f"|V_DP - V_DP_batch| = {abs(V_DP - V_DP_batch).mean()}")
    print(f"|V_DP - V_MP_SP| = {abs(V_DP - V_MP_SP).mean()}")
    print(f"|V_DP - V_MP_HP| = {abs(V_DP - V_MP_HP).mean()}")
    print(f"|V_DP - V_SP| = {abs(V_DP - V_SP).mean()}")
    print("=============================================")
    E_DP = grid.integrate(rho_DP * V_DP)
    E_MP_SP = grid.integrate(rho_DP * V_MP_SP)
    E_MP_HP = grid.integrate(rho_DP * V_MP_HP)
    E_SP = grid.integrate(rho_DP * V_SP)
    print(f"|E_DP - E_MP_SP| = {abs(E_DP - E_MP_SP).mean()}")
    print(f"|E_DP - E_MP_HP| = {abs(E_DP - E_MP_HP).mean()}")
    print(f"|E_DP - E_SP| = {abs(E_DP - E_SP).mean()}")
