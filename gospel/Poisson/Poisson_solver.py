"""
This class solves the Poisson equation.
You can get the potential from 'compute_potential()' method.

[Input]
    input_grid : coordinates of grid points for each axis. Type is 'numpy ndarray of (3,)'
                 ex) input_grid = [[0, 0.4, ...]. [0, 0.5, ...], [0, 0.6, ...]] when spacings for each axis are 0.4, 0.5, 0.6.
    pbc        : periodic boundart condition. Type is list or numpy ndarray of (3,). ex) np.array([ True, False, False ])

(Usage example)
    poisson_solver = Poisson_solver(grid_points, pbc)
    potential = poisson_solver.compute_potential(density)
    energy    = poisson_solver.compute_energy(density, potential)
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from gospel.Grid import Grid


class Poisson_solver:
    """Solving Poisson equation. This class is parent class.

    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  use_fine_grid: bool
    :param use_fine_grid:
        whether using fine grid or not
    """

    def __init__(self, grid, use_fine_grid=False):
        self._grid = grid
        self._pbc = self._grid.get_pbc()
        self._use_fine_grid = use_fine_grid
        if self._use_fine_grid:
            self._fine_grid = Grid(grid.atoms.copy(), grid.gpts * 2)  # 2x finer grid
            self._gpts = self._fine_grid.gpts
            self._spacings = self._fine_grid.spacings
        else:
            self._gpts = self._grid.gpts
            self._spacings = self._grid.spacings
        return

    def compute_potential(self, density, remove_avg=False):
        """Calculate Poisson potential from density.

        :type  density: torch.Tensor
        :param density:
            charge density part of Poisson equation, shape=(N,) or (nbands, N)
        :type  remove_avg: bool
        :param remove_avg:
            whether to remove average value of density or not, defaults to False
        """
        if remove_avg:
            density = density - self.compute_avg_charge(density)

        if self._use_fine_grid:
            ## Very slowe... -> Should be replaced to DoubleGrid later.
            interpolator = rgi(
                self.grid.grid_points_on_axis,
                density.reshape(self.grid.gpts),
                bounds_error=False,
                fill_value=None,
            )
            fine_density = interpolator(self._fine_grid.points.numpy()).astype(
                float
            )  # from coarse grid to fine grid

            integral_of_density = self.grid.integrate(density)
            integral_of_fine_density = self._fine_grid.integrate(fine_density)
            diff_of_integral = integral_of_fine_density - integral_of_density
            if abs(diff_of_integral) > 1e-6:
                print(
                    "WARNING: The integral difference between density and interpolated density is large. (={})".format(
                        diff_of_integral
                    )
                )
            fine_density -= diff_of_integral / (
                self._fine_grid.ngpts * self._fine_grid.microvolume
            )

            potential = self.child_compute_potential(fine_density)
            interpolator = rgi(
                self._fine_grid.grid_points_on_axis,
                potential.reshape(self._fine_grid.gpts),
                bounds_error=False,
                fill_value=None,
            )
            potential = interpolator(self.grid.points.numpy()).astype(
                float
            )  # from fine grid to coarse grid

        else:
            potential = self.child_compute_potential(density)
        return potential

    def compute_energy(self, density, potential):
        energy = 0.5 * self._grid.integrate(density * potential)
        return energy

    def compute_avg_charge(self, density):
        if np.sum(self._grid.get_pbc()) == 3:
            if density.dtype == np.object:
                avg_charge = np.array(list(map(np.mean, density)))
            else:
                avg_charge = density.mean(axis=-1).reshape(-1, 1)
        elif np.sum(self._grid.get_pbc()) == 0:
            avg_charge = 0.0
        else:
            raise NotImplementedError
        return avg_charge
