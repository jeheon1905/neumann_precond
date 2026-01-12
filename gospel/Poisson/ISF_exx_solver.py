import numpy as np
from scipy.special import erfcx
from gospel.util import timer
from gospel.Poisson.Poisson_solver import Poisson_solver
from itertools import product
import sys
np.set_printoptions(threshold=sys.maxsize)


def create_ISF_exx_solver(grid, kpts=np.array([[0,0,0]])):
    kq = [k - q for k, q in product(kpts, kpts)]
    return ISF_exx_solver(grid, kq)


class ISF_exx_solver(Poisson_solver):
    @timer
    def __init__(self, grid, kq=[[0,0,0]], ISF_cutoff=None, use_fine_grid=False, fp=64):
        assert isinstance(ISF_cutoff, type(None)) or isinstance(ISF_cutoff, int),\
               "Input variable error. ISF_cutoff is 'None' or 'int' type."
        assert grid.is_cartesian, "ISF method is only supported for cartensian grid."

        super().__init__(grid, use_fine_grid)
        self.__type = 'ISF'
        self.__kq   = np.array(kq)
        self.__kq_x = np.unique(self.__kq[:,0])
        self.__kq_y = np.unique(self.__kq[:,1])
        self.__kq_z = np.unique(self.__kq[:,2])

        self.__cell_lengths = self._grid.cell_lengths

        self.__fp = fp
        if   fp == 32:  self.__np_float, self.__np_complex = np.float32, np.complex64
        elif fp == 64:  self.__np_float, self.__np_complex = np.float64, np.complex128
        else: assert False, "available 'fp' values are 32 and 64."

        ## Set n_cutoff ##
        self.__n_cutoff = ISF_cutoff
        if self.__n_cutoff != None:
            self.__n_cutoff = np.full(3, self.__n_cutoff) * self._pbc
        else:
            if np.sum(self._pbc) == 3: # P3D
                self.__n_cutoff = np.round(20 / self.__cell_lengths).astype(int)
                self.__n_cutoff[self.__n_cutoff < 5] = 5  # Set the minimum value of n_cutoff as 5.
            else:
                self.__n_cutoff = np.round(2000 / self.__cell_lengths).astype(int)
                self.__n_cutoff[self.__n_cutoff < 100] = 100  # Set the minimum value of n_cutoff as 100.
                self.__n_cutoff *= self._pbc
        print('ISF_cutoff is set to {}.'.format(self.__n_cutoff))

        ## t sampling ##
        """
        Divide 't (0 ~ inf)' with 4 ranges.             : [t_i, t_1], [t_1, t_2], [t_2, t_3], [t_3, t_f]
        The number of sampled points for each range are :   num_t_1 ,   num_t_2 ,   num_t_3 ,   num_t_4
        """
        if np.sum(self._pbc) == 3: t_i = 0.05   # P3D
        else:                      t_i = 0.0    # P0D, P1D, P2D
        t_1, num_t_1 =   0.2, 12
        t_2, num_t_2 =   0.4, 12
        t_3, num_t_3 =   2.0, 12
        t_f, num_t_4 = 500.0, 12
        self.__t_f   = t_f
        if np.sum(self._pbc) == 0: t_1, num_t_1 = 0.0, 0  # P0D
        self.__total_num_t = num_t_1 + num_t_2 + num_t_3 + num_t_4

        ## Linear sampling
        t_val_1, w_t_1 = legendre_t_sampling(       t_i ,        t_1 ,     num_t_1)
        t_val_2, w_t_2 = legendre_t_sampling(       t_1 ,        t_2 ,     num_t_2)
        t_val_3, w_t_3 = legendre_t_sampling(       t_2 ,        t_3 ,     num_t_3)
        ## Logarithmic sampling
        t_val_4, w_t_4 = legendre_t_sampling(np.log(t_3), np.log(t_f),     num_t_4)
        t_val_4        = np.exp(t_val_4)
        ## Concatenate t values and weights.
        self.__t_val = np.concatenate([t_val_1, t_val_2, t_val_3,         t_val_4])
        self.__w_t   = np.concatenate([  w_t_1,   w_t_2,   w_t_3,   w_t_4*t_val_4]) * (2.0 / np.sqrt(np.pi))
        ## Construct F matrices ##
        self.__F_xs_kq = self._construct_F_matrix(self.__t_val, self.__kq_x, axis=0)
        self.__F_ys_kq = self._construct_F_matrix(self.__t_val, self.__kq_y, axis=1)
        self.__F_zs_kq = self._construct_F_matrix(self.__t_val, self.__kq_z, axis=2)

        print("ISF method is used for Poisson")
        print("* [{}, {}] : {} points".format(t_i, t_1, num_t_1))
        print("* [{}, {}] : {} points".format(t_1, t_2, num_t_2))
        print("* [{}, {}] : {} points".format(t_2, t_3, num_t_3))
        print("* [{}, {}] : {} points".format(t_3, t_f, num_t_4))

        print(f"t_val = {self.__t_val}")
        return


    def __str__(self):
        s = str()
        s += '\n========================= [ ISF_exx_solver ] =========================='
        s += f"\n* type          : {self.__type}"
        s += f"\n* use_fine_grid : {self._use_fine_grid}"
        s += f"\n* kq            : \n{self.__kq}"
        if self.__type == 'ISF':
            s += f"\n* ISF_cutoff    : {self.__n_cutoff}"
            s += f"\n* fp            : {self.__fp}"
        s += '\n=======================================================================\n'
        return s


    @timer
    def _construct_F_matrix(self, t_val, kq, axis):
        from scipy.sparse import diags
        t_size = len(t_val)
        points = np.mgrid[0:self.__cell_lengths[axis]:self._spacings[axis]][:self._gpts[axis]]

        F_kq = np.empty(len(kq), dtype=object)
        for i_k in range(len(kq)):
            dtype = self.__np_float if kq[i_k] == 0 else self.__np_complex
            F_kq[i_k] = np.zeros((t_size, self._gpts[axis], self._gpts[axis]), dtype=dtype)
            for i_t in range(t_size):
                F_tmp = self._compute_F(t_val[i_t], points, self._spacings[axis],
                                        self.__cell_lengths[axis], self.__n_cutoff[axis],
                                        kq[i_k], dtype)
                F_tmp = np.append(F_tmp[::-1].conj(), F_tmp[1:])
                F_kq[i_k][i_t] = diags(F_tmp, range(-self._gpts[axis]+1, self._gpts[axis]),
                                       shape=(self._gpts[axis], self._gpts[axis])).toarray()
        return F_kq


    ## for testing speed comparison
    def numpy_compute_potential(self, density):
    #def child_compute_potential(self, density, kq=[0,0,0]):
        from time import time
        potential = np.zeros(self._gpts, dtype=density.dtype)
        rho = density.reshape(self._gpts)

        i_kq_x = np.where(self.__kq_x == kq[0])
        i_kq_y = np.where(self.__kq_y == kq[1])
        i_kq_z = np.where(self.__kq_z == kq[2])

        st = time()
        for F_x, F_y, F_z, w in zip(self.__F_xs_kq[i_kq_x], self.__F_ys_kq[i_kq_y],\
                                    self.__F_zs_kq[i_kq_z], self.__w_t):
            tmp = np.tensordot(rho, F_x, ([0],[0]))
            tmp = np.tensordot(tmp, F_y, ([0],[0]))
            tmp = np.tensordot(tmp, F_z, ([0],[0]))
            potential += w * tmp
        potential = potential.reshape(-1)
        potential += density / self.__t_f**2
        print(f" numpy_compute_potential Time : {time()-st}")
        return potential

    #def new_child_compute_potential(self, density, kq=[0,0,0]):
    @timer
    def child_compute_potential(self, density, kq=[0,0,0]):
        from time import time
        ## density.shape = (nkpts * nbands,)---(ngpts,)
        potential = np.zeros(len(density), dtype=object) # shape=(nbands,)---(ngpts,)

        i_kq_x = np.argwhere(self.__kq_x == kq[0]).item()
        i_kq_y = np.argwhere(self.__kq_y == kq[1]).item()
        i_kq_z = np.argwhere(self.__kq_z == kq[2]).item()

        for i in range(len(density)):
            rho = density[i].reshape(self._gpts)
            for F_x, F_y, F_z, w in zip(self.__F_xs_kq[i_kq_x], self.__F_ys_kq[i_kq_y],\
                                        self.__F_zs_kq[i_kq_z], self.__w_t):
                tmp = np.tensordot(rho, F_x, ([0],[0]))
                tmp = np.tensordot(tmp, F_y, ([0],[0]))
                tmp = np.tensordot(tmp, F_z, ([0],[0]))
                potential[i] += w * tmp
            potential[i] = potential[i].reshape(-1)
            potential[i] += density[i] / self.__t_f**2
        return potential


    def _compute_F(self, t, x_bar, spacing, cell_length, n_cutoff, kq, dtype):
        ### np.vectorize 하기!!
        """
        Arguments:
            t (np.ndarray) : sampled t values
            x_bar (np.ndarray) : grid points on an axis
            spacing (float) : grid spacing
            cell_length (float) : length of cell on an axis
            n_cutoff (int) : # of neighboring cells to be summed.
            kq (float) : kpoint vector. (k - q)
            dtype (type) : datatype of F matrix
        """
        F = np.zeros_like(x_bar, dtype=dtype)
        if kq == 0:
            for n in range(-n_cutoff, n_cutoff + 1):
                x_diff = x_bar + n * cell_length
                z = np.pi / 2 / spacing / t + t * x_diff * 1j
                tmp = - (t * x_diff)**2
                ret_val = np.exp(tmp)
                ret_val -= (np.exp(tmp - z**2) * erfcx(z)).real
                F += ret_val
            F *= spacing
        else:
            for n in range(-n_cutoff, n_cutoff + 1):
                x_diff = x_bar + n * cell_length
                z1 = (np.pi + spacing * kq) / 2 / spacing / t - t * x_diff * 1j
                z2 = (np.pi - spacing * kq) / 2 / spacing / t + t * x_diff * 1j
                tmp = - t**2 * x_diff**2 - kq * x_diff * 1j
                ret_val = 2 * np.exp(tmp)
                ret_val -= np.exp(tmp - z1**2) * erfcx(z1) + np.exp(tmp - z2**2) * erfcx(z2)
                F += ret_val
            F *= 0.5 * spacing
        return F


    def plot_F_t(self, density, t_list=None, kq=[0,0,0], dtype=np.float64):
        ## t sampling ##
        if t_list is None: t_list = self.__t_val
        else:              t_list = t_list
        t_size = len(t_list)

        ## construct F matrices
        F_xs = self._construct_F_matrix(t_list, [kq[0]], axis=0)[0]
        F_ys = self._construct_F_matrix(t_list, [kq[1]], axis=1)[0]
        F_zs = self._construct_F_matrix(t_list, [kq[2]], axis=2)[0]

        ## calculate F(t)
        rho = density.reshape(self._gpts)
        F_t = np.zeros(t_size, dtype=dtype)
        for t, (F_x, F_y, F_z) in enumerate(zip(F_xs, F_ys, F_zs)):
            V_t = np.tensordot(np.tensordot(np.tensordot(rho, F_x, ([0],[0])),
                                                              F_y, ([0],[0])),
                                                              F_z, ([0],[0]))
            F_t[t] = self._grid.integrate(density.conj() * V_t.reshape(-1))
        return F_t


    @property
    def t_val(self): return self.__t_val

    @property
    def F_xs_kq(self): return self.__F_xs_kq

    @property
    def F_ys_kq(self): return self.__F_ys_kq

    @property
    def F_zs_kq(self): return self.__F_zs_kq


def legendre_t_sampling(a, b, n):
    """
    Legendre sampling.
    a : lower limit of integration
    b : upper limit of integration
    n : Number of points to sample
    """
    from scipy.special import roots_legendre
    assert n >= 0
    if n == 0: return [], []
    t_val, w_t = roots_legendre(n)
    t_val *= (b - a) / 2
    t_val += (b + a) / 2
    w_t   *= (b - a) / 2
    return t_val, w_t



if __name__=="__main__":
    from gospel.Poisson.Poisson_ISF_python import Poisson_ISF
    from gospel.Grid import Grid
    from ase.build import bulk
    from ase.units import Bohr

    atoms = bulk('C', 'diamond', a=3.57 / Bohr, cubic=True)
    atoms.set_pbc([True, False, False])
    gpts  = (100, 100, 100)
    #gpts  = (10, 10, 10)
    grid = Grid(atoms, gpts)
    print(grid)

    rho = np.exp(-np.linalg.norm(grid.points - grid.center_point, axis=-1)**2)
    rho /= np.sum(rho)
    rho -= np.ones_like(rho) / len(rho) # Make neutral to converge
    kq = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kq = kq @ atoms.cell.reciprocal() * 2 * np.pi # k/(2*pi*L)
    print(f"kq = \n{kq}")
    #exx_solver = ISF_exx_solver(grid, kq, fp=32)
    exx_solver = ISF_exx_solver(grid, kq, fp=64)
    print(exx_solver)


    ## Plot F(t)
    import matplotlib.pyplot as plt
    for i in range(len(kq)):
        logt_list = np.arange(-3, 4, 0.1)
        t_list = 10**logt_list
        F_t = exx_solver.plot_F_t(rho, t_list, kq[i])
        print(f'logt_list = {logt_list}')
        print(f'F_t = {F_t}')

        plt.figure()
        plt.title(f"F(t) for k-q={kq[i]}")
        plt.plot(t_list, F_t, label='')
        plt.xlabel('t')
        plt.ylabel('F(t)')
        plt.loglog()
        # plt.loglog(basex=10, basey=10)
    plt.show()
