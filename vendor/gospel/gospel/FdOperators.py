import torch
import numpy as np
from scipy import sparse
from gospel.util import tensordot, scipy_to_torch_sparse, toTuple, timer
from functools import lru_cache

## Central finite difference coefficients
derivatives = [
    [0],
    [0, 1 / 2],  # fd_order=1
    [0, 2 / 3, -1 / 12],  # fd_order=2
    [0, 3 / 4, -3 / 20, 1 / 60],  # fd_order=3
    [0, 4 / 5, -1 / 5, 4 / 105, -1 / 280],  # fd_order=4
]

laplace = [
    [0],
    [-2, 1],  # fd_order=1
    [-5 / 2, 4 / 3, -1 / 12],  # fd_order=2
    [-49 / 18, 3 / 2, -3 / 20, 1 / 90],  # fd_order=3
    [-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],  # fd_order=4
    [-5269 / 1800, 5 / 3, -5 / 21, 5 / 126, -5 / 1008, 1 / 3150],
    [-5369 / 1800, 12 / 7, -15 / 56, 10 / 189, -1 / 112, 2 / 1925, -1 / 16632],
]

## Forward finite difference coefficients
derivatives_forward = [
    [-3 / 2, 2, -1 / 2],
    [-11 / 6, 3, -3 / 2, 1 / 3],
    [-25 / 12, 4, -3, 4 / 3, -1 / 4],
    [-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5],
    [-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6],
]

laplace_forward = [
    [1, -2, 1],
    [2, -5, 4, -1],
    [35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12],
    [15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6],
    [203 / 45, -87 / 5, 117 / 4, -254 / 9, 33 / 2, -27 / 5, 137 / 180],
    [469 / 90, -223 / 10, 879 / 20, -949 / 18, 41, -201 / 10, 1019 / 180, -7 / 10],
]

@lru_cache(maxsize=6)
@toTuple
def _make_1st_derivative_op(grid, axis, as_sparse=False, pytorch=False, device=None):
    """Make 1st derivative finite-difference matrix of each axis

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  axis: int
    :param axis:
        index of axis, 0, 1, and 2 for x-, y-, and z-axis, respectively
    :type  as_sparse: bool, optional
    :param as_sparse:
        return as sparse, defaults to False

    :rtype: np.ndarray or scipy.sparse.csr_matrix
    :return:
        finite-difference 1st derivative operator
    """
    FD_order = grid.FD_order
    assert 0 < FD_order and FD_order <= 4
    pbc = grid.get_pbc()[axis]
    gpt = grid.gpts[axis]

    ret_matrix = np.zeros((gpt, gpt), dtype=float)
    for fd in range(1, FD_order + 1):
        ret_matrix += np.diag(derivatives[FD_order][fd] * np.ones(gpt - fd), k=fd)
        if pbc:
            ret_matrix -= np.diag(derivatives[FD_order][fd] * np.ones(fd), k=gpt - fd)
    ret_matrix -= ret_matrix.T

    if pytorch:
        device = device if isinstance(device, torch.device) else torch.device('cpu')
        ret_matrix = torch.from_numpy(ret_matrix).to(device)
        if as_sparse:
            ret_matrix = ret_matrix.to_sparse_csr()
    else:
        if as_sparse:
            ret_matrix = sparse.csr_matrix(ret_matrix)
    return ret_matrix

#@lru_cache(maxsize=3)
@toTuple
def _make_2nd_derivative_op(grid, axis, as_sparse=False, pytorch=False, device=None):
    """Make 2nd derivative finite-difference matrix of each axis

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  axis: int
    :param axis:
        index of axis, 0, 1, and 2 for x-, y-, and z-axis, respectively
    :type  as_sparse: bool, optional
    :param as_sparse:
        return as sparse, defaults to False

    :rtype: np.ndarray or scipy.sparse.csr_matrix
    :return:
        finite-difference 2nd derivative operator
    """
    FD_order = grid.FD_order
    pbc = grid.get_pbc()[axis]
    gpt = grid.gpts[axis]
    assert 0 < FD_order and FD_order <= 6
    assert gpt >= (2 * FD_order + 1)

    ret_matrix = np.zeros((gpt, gpt), dtype=float)
    for fd in range(1, FD_order + 1):
        ret_matrix += np.diag(laplace[FD_order][fd] * np.ones(gpt - fd), k=fd)
        if pbc:
            ret_matrix += np.diag(laplace[FD_order][fd] * np.ones(fd), k=gpt - fd)
    ret_matrix += ret_matrix.T
    ret_matrix += np.diag(laplace[FD_order][0] * np.ones(gpt))

    if pytorch:
        device = device if isinstance(device, torch.device) else torch.device('cpu')
        ret_matrix = torch.from_numpy(ret_matrix).to(device)
        if as_sparse:
            ret_matrix = ret_matrix.to_sparse_csr()
    else:
        if as_sparse:
            ret_matrix = sparse.csr_matrix(ret_matrix)
    return ret_matrix


@toTuple
#@lru_cache(maxsize=3)
def make_kinetic_op(grid, kpt=[0, 0, 0], combine=True, as_sparse=True, pytorch=True, device=None):
    """Make kinetic operator

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  kpt: list, np.ndarray, optional
    :param kpt:
        k-point, defaults to [0, 0, 0]
    :type  combine: bool, optional
    :param combine:
        whether combine operators or not, defaults to True
    :type  as_sparse: bool, optional
    :param as_sparse:
        return as sparse or dense, defaults to True

    :rtype: tuple[np.ndarray], scipy.sparse.csr_matrix
    :return:
        kinetic operator

    Eq)
    \vec{T}_{\vec{k}} = ( \vec{\nabla} + i \vec{k} )^2
                      = \nabla^2 + 2i \vec{nabla} \cdot \vec{k} - k^2
    """
    if combine:
        assert as_sparse
    is_gamma = kpt[0] == 0 and kpt[1] == 0 and kpt[2] == 0
    gpts = grid.gpts
    spacings = grid.spacings

    I_xx = sparse.identity(gpts[0])
    I_yy = sparse.identity(gpts[1])
    I_zz = sparse.identity(gpts[2])
    T_xx = _make_2nd_derivative_op(grid, 0, as_sparse) / spacings[0] ** 2
    T_yy = _make_2nd_derivative_op(grid, 1, as_sparse) / spacings[1] ** 2
    T_zz = _make_2nd_derivative_op(grid, 2, as_sparse) / spacings[2] ** 2
    if not is_gamma:
        dx = _make_1st_derivative_op(grid, 0, as_sparse) / spacings[0]
        dy = _make_1st_derivative_op(grid, 1, as_sparse) / spacings[1]
        dz = _make_1st_derivative_op(grid, 2, as_sparse) / spacings[2]
        T_xx = T_xx.astype(complex)
        T_yy = T_yy.astype(complex)
        T_zz = T_zz.astype(complex)
        T_xx += 2 * 1j * kpt[0] * dx - kpt[0] ** 2 * I_xx
        T_yy += 2 * 1j * kpt[1] * dy - kpt[1] ** 2 * I_yy
        T_zz += 2 * 1j * kpt[2] * dz - kpt[2] ** 2 * I_zz
    T_xx *= -0.5
    T_yy *= -0.5
    T_zz *= -0.5

    if combine:
        T_xyzxyz = (
            sparse.kron(T_xx, sparse.kron(I_yy, I_zz))
            + sparse.kron(I_xx, sparse.kron(T_yy, I_zz))
            + sparse.kron(sparse.kron(I_xx, I_yy), T_zz)
        )
        if pytorch:
            T_xyzxyz = scipy_to_torch_sparse(T_xyzxyz)
            if isinstance(device, torch.device):
                T_xyzxyz = T_xyzxyz.to(device)
        return T_xyzxyz
    else:
        if pytorch:
            T_xx = torch.from_numpy(T_xx)
            T_yy = torch.from_numpy(T_yy)
            T_zz = torch.from_numpy(T_zz)
            if isinstance(device, torch.device):
                T_xx = T_xx.to(device)
                T_yy = T_yy.to(device)
                T_zz = T_zz.to(device)

        return T_xx, T_yy, T_zz

def calc_kinetic(gpts:torch.Tensor,
                 T_xx:torch.Tensor,
                 T_yy:torch.Tensor, 
                 T_zz:torch.Tensor, 
                 x:torch.Tensor)->torch.Tensor:
    """Calculate kinetic operation.

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  T_xx: np.ndarray, scipy.sparse.csr_matrix, torch.Tensor
    :param T_xx:
        kinetic operator of x-axis
    :type  T_yy: np.ndarray, scipy.sparse.csr_matrix, torch.Tensor
    :param T_yy:
        kinetic operator of y-ayis
    :type  T_zz: np.ndarray, scipy.sparse.csr_matrix, torch.Tensor
    :param T_zz:
        kinetic operator of z-azis
    :type  x: np.ndarray, torch.Tensor
    :param x:
        orbital, shape=(ngpts, nbands)

    :rtype: np.ndarray
    """
    #_x   = x.reshape(gpts[0],gpts[1],gpts[2], -1) # shape=(Nx, Ny, Nz, nbands)
    _x   = x.view(gpts[0],gpts[1],gpts[2], -1) # shape=(Nx, Ny, Nz, nbands) potential memory issue
    #_x   = x.reshape(*grid.gpts, -1) # shape=(Nx, Ny, Nz, nbands)
    return_val = torch.tensordot(T_xx, _x, ([1], [0]))
    return_val = return_val + torch.tensordot(T_yy, _x, ([1], [1])).permute([1, 0, 2, 3])
    return_val = return_val + torch.tensordot(T_zz, _x, ([1], [2])).permute([1, 2, 0, 3])
    return return_val.reshape(gpts[0]*gpts[1]*gpts[2], -1) 
    #return return_val.reshape(grid.ngpts, -1) 
    #return (tmp1 + tmp2 + tmp3).reshape(grid.ngpts, -1) 

@timer
def divergence(grid, values, use_sparse=True):
    """calculate divergence

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  values: np.ndarray
    :param values:
        values, shape=(3, ngpts)
    :type  use_sparse: bool, optional
    :param use_sparse:
        whether use dense or sparse for gradient operator, defaults to True

    div F = \nabla \cdot F
          = (\partial_x, \partial_y, \partial_z) \cdot (F_x, F_y, F_z)
          = \partial_x \cdot F_x + \partial_y \cdot F_y + \partial_z \cdot F_z
          =         A            +         B            +         C
    """
    assert values.shape == (3, grid.ngpts)
    spacings = grid.spacings
    gpts = grid.gpts
    pytorch  = isinstance(values, torch.Tensor)

    if pytorch:
        Dx = _make_1st_derivative_op(grid, 0, use_sparse, pytorch, values.device) / spacings[0]
        Dy = _make_1st_derivative_op(grid, 1, use_sparse, pytorch, values.device) / spacings[1]
        Dz = _make_1st_derivative_op(grid, 2, use_sparse, pytorch, values.device) / spacings[2]
        retval = (
            torch.tensordot(Dx, values[0].reshape(*gpts), ([1], [0]))
            + torch.tensordot(Dy, values[1].reshape(*gpts), ([1], [1])).permute([1, 0, 2])
            + torch.tensordot(Dz, values[2].reshape(*gpts), ([1], [2])).permute([1, 2, 0])
        ).reshape(-1)
    else:        
        Dx = _make_1st_derivative_op(grid, 0, use_sparse, pytorch) / spacings[0]
        Dy = _make_1st_derivative_op(grid, 1, use_sparse, pytorch) / spacings[1]
        Dz = _make_1st_derivative_op(grid, 2, use_sparse, pytorch) / spacings[2]
        retval = (
            tensordot(Dx, values[0].reshape(*gpts), ([1], [0]))
            + tensordot(Dy, values[1].reshape(*gpts), ([1], [1])).transpose([1, 0, 2])
            + tensordot(Dz, values[2].reshape(*gpts), ([1], [2])).transpose([1, 2, 0])
        ).reshape(-1)
    return retval


@timer
def gradient(grid, values, use_sparse=False):
#def gradient(grid, values, use_sparse=False, pytorch=True):
    """Calculate the gradient of input values on grid.

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  values: np.ndarray
    :param values:
        shape=(ngpts,)
    :type  use_sparse: bool, optional
    :param use_sparse:
        whether use dense or sparse for gradient operator, defaults to True

    :rtype: np.ndarray
    :return:
        gradient of values, shape=(3, ngpts)

                   ______                        ______          ______          ______
                 /      /|                     /      /|       /      /|       /      /|
                /      / |                    /      / |      /      / |      /      / |
      ______   /______/  |   ______          /______/  |     /______/  |     /______/  |
     |      |  |      |  /  /      /         | dval |  /     | dval |  /     | dval |  /
     | d_xx |  |  val | /  / d_zz /     =    |  /dx | /      |  /dy | /      |  /dz | /
     |______|  |______|/  /______/           |______|/   ,   |______|/   ,   |______|/
                ______
               |      |
               | d_yy |
               |______|
    """
    assert len(values) == grid.ngpts
    gpts = grid.gpts
    spacings = grid.spacings
    pytorch  = isinstance(values, torch.Tensor)
    if pytorch:
        Dx = _make_1st_derivative_op(grid, 0, use_sparse, pytorch, values.device) / spacings[0]
        Dy = _make_1st_derivative_op(grid, 1, use_sparse, pytorch, values.device) / spacings[1]
        Dz = _make_1st_derivative_op(grid, 2, use_sparse, pytorch, values.device) / spacings[2]
        assert Dx.layout == torch.strided, "torch sparse tensordot isn't supported yet."
        retval = torch.zeros((3, *gpts), dtype=values.dtype, device = values.device)
    else:
        Dx = _make_1st_derivative_op(grid, 0, use_sparse, pytorch) / spacings[0]
        Dy = _make_1st_derivative_op(grid, 1, use_sparse, pytorch) / spacings[1]
        Dz = _make_1st_derivative_op(grid, 2, use_sparse, pytorch) / spacings[2]
        retval = np.zeros((3, *gpts), dtype=values.dtype)

    values_3d = values.reshape(*gpts)

    if pytorch:
        Dx = Dx.to(dtype=values_3d.dtype)
        Dy = Dy.to(dtype=values_3d.dtype)
        Dz = Dz.to(dtype=values_3d.dtype)
        retval[0] = torch.tensordot(Dx, values_3d, ([1], [0]))
        retval[1] = torch.tensordot(Dy, values_3d, ([1], [1])).permute([1, 0, 2])
        retval[2] = torch.tensordot(Dz, values_3d, ([1], [2])).permute([1, 2, 0])
    else:
        retval[0] = tensordot(Dx, values_3d, (1, 0))
        retval[1] = tensordot(Dy, values_3d, (1, 1)).transpose([1, 0, 2])
        retval[2] = tensordot(Dz, values_3d, (1, 2)).transpose([1, 2, 0])
    
    return retval.reshape(3, -1)


def gradient_roll(grid, values):
    """
    calculate gradient of values, implemented using np.roll or slicings.
    It is very slow compared to gradient() for FD_order > =2.
    """
    _order = grid.FD_order

    val = values.reshape(*grid.gpts)
    retval = np.zeros((3, *grid.gpts), dtype=values.dtype)
    pbc = grid.get_pbc()

    for n in range(1, _order + 1):
        coeff = derivatives[_order][n]
        if pbc[0]:
            retval[0] += coeff * (np.roll(val, -n, axis=0) - np.roll(val, n, axis=0))
        else:
            retval[0][n:, :, :] -= coeff * val[:-n, :, :]
            retval[0][:-n, :, :] += coeff * val[n:, :, :]
        if pbc[1]:
            retval[1] += coeff * (np.roll(val, -n, axis=1) - np.roll(val, n, axis=1))
        else:
            retval[1][:, n:, :] -= coeff * val[:, :-n, :]
            retval[1][:, :-n, :] += coeff * val[:, n:, :]
        if pbc[2]:
            retval[2] += coeff * (np.roll(val, -n, axis=2) - np.roll(val, n, axis=2))
        else:
            retval[2][:, :, n:] -= coeff * val[:, :, :-n]
            retval[2][:, :, :-n] += coeff * val[:, :, n:]
    return retval.reshape(3, -1) / grid.spacings.reshape(-1, 1)


def calc_deriv_1D(R, local, deriv_order=1, FD_order=1):
    """Calculate derivatives of local potential on radial grid.
       Only support equidistance grid now.

    Arguments)
    R (np.ndarray): radial grid
    local (np.ndarray): local potential values on radial grid
    deriv_order (int): derivative order
    FD_order (int): finite-difference order
    """
    assert deriv_order in [1, 2]

    ## Make convolution function (conv)
    deriv = derivatives if deriv_order == 1 else laplace
    conv = np.zeros(2 * FD_order + 1)
    accuracy = FD_order - 1 + deriv_order
    for fd in range(1, FD_order + 1):
        conv[FD_order + fd] = deriv[accuracy][fd]
        conv[FD_order - fd] = -deriv[accuracy][fd]
    ## Convolution
    dlocal = np.convolve(local, conv[::-1], "same")

    ## Treat start- and end-points.
    deriv_forward = derivatives_forward if deriv_order == 1 else laplace_forward
    conv = np.array(deriv_forward[FD_order - 1])
    dlocal[:FD_order] = np.convolve(local, conv[::-1])[
        len(conv) - 1 : len(conv) - 1 + FD_order
    ]
    dlocal[-FD_order:] = np.convolve(local, -conv)[
        -len(conv) + 1 - FD_order : -len(conv) + 1
    ]

    spacing = R[1] - R[0]
    if not np.all(abs(np.diff(R) - spacing) < 1e-9):
        assert 0, "calc_local_deriv() is supported only for equidistance radial grid."
    dlocal /= spacing
    return dlocal


if __name__ == "__main__":
    # from Grid import Grid
    from gospel.Grid import Grid
    import numpy as np
    from time import time
    from ase import Atoms

    ####################################################################################
    print("=========================================================================")
    print("TEST : _make_1st_derivative_op() & _make_2nd_derivative_op()")
    FD_order = 2
    gpts = np.full(3, 7)
    cell = gpts
    grid = Grid(Atoms(cell=cell), gpts)
    grid.set_pbc([True, True, True])
    print(" * FD_order  : ", FD_order)
    print(" * gpts : ", gpts)
    print(" * pbc       : ", grid.get_pbc())

    axis = 0
    print("1st derivative matrix = \n", _make_1st_derivative_op(grid, axis))
    print("2nd derivative matrix = \n", _make_2nd_derivative_op(grid, axis))
    print("=========================================================================")
    ####################################################################################

    ####################################################################################
    print("=========================================================================")
    print("TEST : make_kinetic_op()")
    FD_order = 1
    gpts = np.full(3, 7)
    cell = gpts
    grid = Grid(Atoms(cell=cell), gpts)
    grid.set_pbc([False, False, False])
    kpt = [1, 1, 1]
    print(" * FD_order  : ", FD_order)
    print(" * gpts : ", gpts)
    print(" * pbc       : ", grid.get_pbc())
    print(" * kpt       : ", kpt)
    st = time()
    T_xyzxyz = make_kinetic_op(grid, combine=True)
    print("T_xyzxyz = \n", T_xyzxyz.to_dense())
    T_xyzxyz_test = make_kinetic_op(grid, kpt=kpt, combine=True)
    print("T_xyzxyz_test = \n", T_xyzxyz_test.to_dense())
    end = time()
    print("time = {} sec".format(end - st))
    print("=========================================================================")
    ####################################################################################

    ####################################################################################
    print("=========================================================================")
    print("TEST : gradient()")
    FD_order = 2
    gpts = np.full(3, 100)
    cell = gpts
    grid = Grid(Atoms(cell=cell), gpts)
    print(" * FD_order  : ", FD_order)
    print(" * gpts : ", gpts)
    print(" * pbc       : ", grid.get_pbc())
    density = torch.arange(np.prod(gpts), dtype=torch.float64)
    st = time()
    dn_x, dn_y, dn_z = gradient(grid, density)
    end = time()
    print("Time <gradient()> :  {} sec".format(end - st))
    print("=========================================================================")
    ####################################################################################

    ####################################################################################
    print("=========================================================================")
    print("TEST : divergence()")
    FD_order = 2
    gpts = np.full(3, 100)
    cell = gpts
    grid = Grid(Atoms(cell=cell), gpts)
    print(" * FD_order  : ", FD_order)
    print(" * gpts : ", gpts)
    print(" * pbc       : ", grid.get_pbc())
    density = torch.arange(np.prod(gpts), dtype=torch.float64)
    F = np.asarray(gradient(grid, density))
    st = time()
    div_F = divergence(grid, F)
    ed = time()
    print("Time <divergence()> : {} sec".format(ed - st))
    print("=========================================================================")
    ####################################################################################
