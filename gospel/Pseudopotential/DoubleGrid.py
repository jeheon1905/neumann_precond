import numpy as np
from gospel.FdOperators import _make_1st_derivative_op
from gospel.Grid import Grid


class DoubleGrid:
    """
    Make doubly-finer grid (double grid).
    values on double grid are calculated from interpolation.

    :type  grid: gospel.Grid
    :param grid:
        Grid object

    **Example**

    >>> grid = Grid(...)
    >>> double_grid = DoubleGrid(grid)
    >>> rho = torch.randn(Nx, Ny, Nz)
    >>> rho = double_grid.doubling(rho)  # rho.shape=(2 * Nx, 2 * Ny, 2 * Nz)
    """

    def __init__(self, grid):
        self.__grid = grid
        self.__fine_grid = Grid(grid.atoms, gpts=grid.gpts * [2, 2, 2])
        self.__Dx = _make_1st_derivative_op(grid, axis=0)
        self.__Dy = _make_1st_derivative_op(grid, axis=1)
        self.__Dz = _make_1st_derivative_op(grid, axis=2)
        return

    def doubling(self, inp):
        """Tricubicspline of equidistance grid, (Note that 3D PBC is assumed.)

        :type  inp: np.ndarray
        :param inp:
            values on orginal grid, shape=(ngpts,)

        :rtype: np.ndarray
        :return:
            values on double grid, shape=(2**3 * ngpts,)
        """
        inp = inp.reshape(self.__grid.gpts)
        Nx, Ny, Nz = self.__grid.gpts

        ## First axis ##
        fine_val1 = np.zeros((2 * Nx, Ny, Nz), dtype=inp.dtype)
        tmp1 = 0.5 * (inp + np.roll(inp, -1, axis=0))
        # Now, roll considers only PBC boundary condition
        dv_dx = self.__Dx @ inp.reshape(Nx, -1)
        dv_dx = dv_dx.reshape((Nx, Ny, Nz))
        tmp2 = 0.125 * (dv_dx - np.roll(dv_dx, -1, axis=0))
        fine_val1[np.arange(Nx) * 2, ...] = inp
        fine_val1[np.arange(Nx) * 2 + 1, ...] = tmp1 + tmp2

        ## Second axis ##
        fine_val2 = np.zeros((2 * Nx, 2 * Ny, Nz), dtype=inp.dtype)
        tmp1 = 0.5 * (fine_val1 + np.roll(fine_val1, -1, axis=1))
        dv_dy = self.__Dy @ np.transpose(fine_val1, (1, 2, 0)).reshape(Ny, -1)
        dv_dy = dv_dy.reshape((Ny, Nz, 2 * Nx))
        dv_dy = np.transpose(dv_dy, (2, 0, 1))
        tmp2 = 0.125 * (dv_dy - np.roll(dv_dy, -1, axis=1))
        fine_val2[:, np.arange(Ny) * 2, ...] = fine_val1
        fine_val2[:, np.arange(Ny) * 2 + 1, ...] = tmp1 + tmp2

        ## Third axis ##
        fine_val3 = np.zeros((2 * Nx, 2 * Ny, 2 * Nz), dtype=inp.dtype)
        tmp1 = 0.5 * (fine_val2 + np.roll(fine_val2, -1, axis=2))
        dv_dz = self.__Dz @ np.moveaxis(fine_val2, -1, 0).reshape(Nz, -1)
        dv_dz = dv_dz.reshape((Nz, 2 * Nx, 2 * Ny))
        dv_dz = np.transpose(dv_dz, (1, 2, 0))
        tmp2 = 0.125 * (dv_dz - np.roll(dv_dz, -1, axis=2))
        fine_val3[..., np.arange(Nz) * 2] = fine_val2
        fine_val3[..., np.arange(Nz) * 2 + 1] = tmp1 + tmp2

        return fine_val3.reshape(-1)

    def get_fine_grid(self):
        return self.__fine_grid


if __name__ == "__main__":
    from ase import Atoms
    from time import time
    from scipy.interpolate import RegularGridInterpolator as rgi

    atoms = Atoms("H", cell=[10, 10, 10])
    grid = Grid(atoms, gpts=(20, 20, 20))
    # grid = Grid(atoms, gpts=(30, 30, 30))
    print(grid)

    # def f(x, r=[5.0, 5.0, 5.0]):
    def f(x, r=[3, 4, 5]):
        return np.exp(-np.linalg.norm(x - r, axis=-1) ** 2)

    y = f(grid.points)

    st = time()
    doublegrid = DoubleGrid(grid)
    fine_grid = doublegrid.get_fine_grid()
    print(fine_grid)
    print(f"DoubleGrid   Time : {time() - st} sec")
    ref_y = f(fine_grid.points)
    fine_y = doublegrid.doubling(y)
    st = time()
    linear_y = rgi(
        grid.grid_points_on_axis,
        y.reshape(grid.gpts),
        bounds_error=False,
        fill_value=0.0,
    )(fine_grid.points)
    print(f"Scipy linear Time : {time() - st} sec")

    print(f"Integral(|ref_y|)            = {fine_grid.integrate(abs(ref_y))}")
    print(f"Integral(|ref_y - fine_y|)   = {fine_grid.integrate(abs(ref_y - fine_y))}")
    print(
        f"Integral(|ref_y - linear_y|) = {fine_grid.integrate(abs(ref_y - linear_y))}"
    )

    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("x-axis")
    gpts = fine_grid.gpts
    _ref_y = ref_y.reshape(gpts)[:, gpts[1] // 2, gpts[2] // 2]
    _fine_y = fine_y.reshape(gpts)[:, gpts[1] // 2, gpts[2] // 2]
    _linear_y = linear_y.reshape(gpts)[:, gpts[1] // 2, gpts[2] // 2]
    plt.plot(np.arange(gpts[0]), _linear_y, label="linear_y")
    plt.plot(np.arange(gpts[0]), _ref_y, label="ref_y", marker="o")
    plt.plot(np.arange(gpts[0]), _fine_y, label="fine_y")
    plt.legend()

    plt.figure()
    plt.title("y-axis")
    gpts = fine_grid.gpts
    _ref_y = ref_y.reshape(gpts)[gpts[0] // 2, :, gpts[2] // 2]
    _fine_y = fine_y.reshape(gpts)[gpts[0] // 2, :, gpts[2] // 2]
    _linear_y = linear_y.reshape(gpts)[gpts[0] // 2, :, gpts[2] // 2]
    plt.plot(np.arange(gpts[1]), _linear_y, label="linear_y")
    plt.plot(np.arange(gpts[1]), _ref_y, label="ref_y", marker="o")
    plt.plot(np.arange(gpts[1]), _fine_y, label="fine_y")
    plt.legend()

    plt.figure()
    plt.title("z-axis")
    gpts = fine_grid.gpts
    _ref_y = ref_y.reshape(gpts)[:, gpts[1] // 2, gpts[2] // 2]
    _fine_y = fine_y.reshape(gpts)[:, gpts[1] // 2, gpts[2] // 2]
    _linear_y = linear_y.reshape(gpts)[:, gpts[1] // 2, gpts[2] // 2]
    plt.plot(np.arange(gpts[2]), _linear_y, label="linear_y")
    plt.plot(np.arange(gpts[2]), _ref_y, label="ref_y", marker="o")
    plt.plot(np.arange(gpts[2]), _fine_y, label="fine_y")
    plt.legend()

    plt.show()
