import torch
import numpy as np
from ase.units import Bohr
from scipy.interpolate import interp1d
from itertools import product

class Grid:
    """Make cartesian uniform-spaced grid and provide useful methods for calculation using grid.

    :type  atoms: ase.Atoms
    :param atoms:
        Atoms object containing pbc and cell information
    :type  gpts: list[int], optional
    :param gpts:
        the number of grid points of each axis, defaults to None
    :type  spacing: list[float], optional
    :param spacing:
        spacings of grid of each axis, defaults to None
    :type  FD_order: int
    :param FD_order:
        finite-differene order, defaults to 3
    """

    def __init__(self, atoms, gpts=None, spacing=None, FD_order=3):
        assert (
            spacing is not None or gpts is not None
        ), "Either 'spacing' or 'gpts' must be entered."
        assert not (spacing is None and gpts is None)
        if gpts is not None:
            assert len(gpts) == 3

        self.atoms = atoms.copy()
        self.__cell = self.atoms.get_cell()
        self.__cell_lengths = self.atoms.cell.cellpar()[:3]
        self.__cell_angles = self.atoms.cell.cellpar()[3:]
        self.__is_cartesian = np.all(abs(self.__cell_angles - [90, 90, 90]) < 1e-7)
        if gpts is None:
            self.__gpts = np.ceil(self.__cell_lengths / np.asarray(spacing)).astype(int)
        else:
            self.__gpts = np.array(gpts).astype(int)
        self.__FD_order = FD_order

        self.__ngpts = np.prod(self.__gpts)
        self.__spacings = self.__cell_lengths / self.__gpts
        self.__indices = self._make_indices()
        self.__points = self._make_grid_points(self.__indices)
        self.__center_point = [0.5, 0.5, 0.5] @ self.__cell
        if self.__is_cartesian:
            self.__microvolume = np.prod(self.__spacings)
        else:
            self.__microvolume = abs(np.linalg.det(self.__cell)) / self.__ngpts
        return

    def __hash__(self):
        return hash( ( tuple(self.__cell_lengths), tuple(self.__cell_angles), bool(self.__is_cartesian), tuple(self.__gpts), self.__FD_order, int(self.__ngpts), tuple(self.__spacings), tuple(self.__center_point), float(self.__microvolume)) )
    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        if ( self.atoms == other.atoms and
            all(self.gpts==other.gpts ) and 
            all(self.spacings==other.spacings) and 
            self.FD_order == other.FD_order ):
            return True
            
        return False

    def __str__(self):
        s = str()
        s += "\n============================= [ Grid ] ============================="
        s += f"\n* cell     (Bohr) : \n{np.round(np.asarray(self.__cell), 3)}"
        s += f"\n* cell     (ang ) : \n{np.round(np.asarray(self.__cell) * Bohr, 3)}"
        s += f"\n* spacings (Bohr) : {np.round(self.__spacings, 3)}"
        s += f"\n* spacings (ang ) : {np.round(self.__spacings * Bohr, 3)}"
        s += f"\n* gpts            : {self.__gpts}"
        s += f"\n* ngpts           : {self.__ngpts}"
        s += f"\n* pbc             : {self.atoms.get_pbc()}"
        s += f"\n* FD_order        : {self.__FD_order}"
        s += f"\n* center_point    : {self.__center_point} (Bohr)"
        s += "\n====================================================================\n"
        return s

    def __repr__(self):
        return f"Grid(gpts={self.__gpts}, spacings={self.__spacings}, pbc={self.get_pbc()}, cell={self.__cell}, FD_order={self.__FD_order})"

    def translation_to(self, center):
        """Translation grid points

        :type  center: np.ndarray
        :param center:
            center point's coordinate to move grid, shape=(3,)
        """
        diff_center = center - self.__center_point
        self.__center_point += diff_center
        self.__points += diff_center
        return

    def get_vertex_points(self):
        """Return 8 vertex points of cartesian cell (np.ndarray)"""
        from itertools import product

        vertex = np.asarray(list(product([0, 1], [0, 1], [0, 1]))) @ self.__cell
        return vertex

    def get_radius(self):
        """return the radius of sphere containing the cell"""
        radius = max(
            np.linalg.norm(self.get_vertex_points() - self.__center_point, axis=1)
        )
        return radius

    def integrate(self, values):
        """Integrate a function represented on grid.

        :type  values: np.ndarray or torch.Tensor
        :param values:
            values of the function on grid,
            shape='(ngpts,)' or '(n, ngpts)' or '(n,) with dtype=object'

        :rtype: float or np.ndarray
        :return:
            integrated value
        """
        assert isinstance(values, np.ndarray) or isinstance(values, torch.Tensor)
        assert (
            len(values.shape) == 2 or len(values.shape) == 1
        ), "The shape of 'values' should be (nspins, n) or (n,)."


        if len(values.shape) == 2:
            assert values.shape[-1] == self.__ngpts
            return self.__microvolume * values.sum(-1).reshape(-1, 1)

        elif len(values.shape) == 1:
            if values.shape[-1] == self.__ngpts:
                return self.__microvolume * values.sum(-1)
            else:
                # if values is torch.Tensor raise Error
                assert isinstance(values, np.ndarray), f" The shape of given torch.Tensor is incorrect {values.shape}"

                # case when numpy array whose dtype is object 
                ret_val = np.zeros(len(values))
                for i in range(len(values)):
                    assert values[i].shape[-1] == self.__ngpts
                    ret_val[i] = values[i].sum(-1).item() if isinstance(values[i],torch.tensor)  else values[i].sum(-1)
                return self.__microvolume * ret_val.reshape(-1, 1)
                

    def intplt(self, values, out_grid_pts, pad_size=1):
        """Interpolate to other grid points.

        :type  values: np.ndarray
        :param values:
            values on grid, shape=(ngpts,)
        :type  out_grid_pts: np.ndarray
        :param out_grid_pts:
            output grid points coordinates, shape=(gpts, 3)
        :type  pad_size: int, optional
        :param pad_size:
            padding size on periodic axis, defaults to 1

        :rtype: np.ndarray
        :return:
            interpolated values on out_grid_pts
        """
        from scipy.interpolate import RegularGridInterpolator as rgi

        assert (
            len(values) == self.__ngpts
        ), f"Wrong shape error. values.shape={values.shape}"
        assert (
            self.__is_cartesian
        ), "Grid.intplt method is only supported for cartesian grid."

        ## padding
        pbc = self.atoms.get_pbc()
        pad_size = pad_size * pbc
        tuple_of_grid_points_on_axis = tuple(
            np.arange(-pad_size[axis], self.__gpts[axis] + pad_size[axis])
            * self.__spacings[axis]
            for axis in range(3)
        )
        values = values.reshape(self.__gpts)
        values = np.pad(
            values,
            tuple((pad_size[axis], pad_size[axis]) for axis in range(3)),
            mode="wrap",
        )

        ## Make interpolator
        interpolator = rgi(
            tuple_of_grid_points_on_axis,
            values,
            bounds_error=False,
            fill_value=0.0,
        )

        ## move external points to corresponding internal points with consideration of periodicity
        coef = out_grid_pts @ np.linalg.inv(self.__cell)  # shape=(ngpts, 3)
        coef[:, pbc] %= 1.0

        out_val = interpolator(coef @ self.__cell)
        return out_val

    def spherical_to_grid(
        self,
        r,
        fn,
        center_position,
        num_near_cell=3,
        extplt=None,
        cutoff_radius=None,
        kind="cubic",
        device=torch.device("cpu"),
    ):
        """
        This function interpolates shperical function to cartessian grid.
        (If pbc is turned on, the effects from nearing cell are also calculated.)

        :type  r: np.ndarray
        :param r:
            radial grid
        :type  fn: np.ndarray
        :param fn:
            radial function values at r points
        :type  center_position: np.ndarray
        :param center_position:
            center position of spherical function. shape=(3,)
        :type  num_near_cell: int, optional
        :param num_near_cell:
            when PBC, the number of nearing cells which affect the interpolated function values,
            defaults to 3
        :type  extplt: str or None, optional
        :param extplt:
            whether extrapolate from outside the cutoff_radius, defaults to None
            e.g., extplt='-Z/r' or '-Z/r**2'
        :type  cutoff_radius: float or None, optional
        :param cutoff_radius:
            interpolate only within the cutoff_radius, and treat outside as zero.
        :type  kind: str, optional
        :param kind:
            interpolation method. default='cubic',
            options: 'cubic' or 'linear'

        :rtype: np.ndarray
        :return:
            interpolated values on grid.

        **Example**

        >>> ## If you want to represent a numerical radial function on grid.
        >>> radial_grid = np.arange(0, 10, 0.01)
        >>> radial_function = np.exp(-radial_grid**2)  # f(r) = e^{-r^2}
        >>> center_position = np.array([0, 0, 0])
        >>> grid = Grid(...)
        >>> val = grid.spherical_to_grid(radial_grid, radial_function, center_position, extplt="e**(-r**2)")
        """

        assert isinstance(r, np.ndarray)
        assert isinstance(fn, np.ndarray)
        # assert isinstance(center_position, np.ndarray) or isinstance(center_position, torch.Tensor)
        assert isinstance(center_position, torch.Tensor)

        num_near = [num_near_cell if self.atoms.get_pbc()[i] else 0 for i in range(3)]
        cubic_interp = interp1d(r, fn, kind=kind, bounds_error=False, fill_value=0.0)

        if cutoff_radius and cutoff_radius > self.__cell_lengths.min():
            ## TODO: Now, it does not support GPU calculation.
            ## Check whether 'num_near_cell' is enough or not.
            points = torch.from_numpy(self.__points)
            if cutoff_radius:
                n1, n2, n3 = (num_near + np.ones(3, dtype=int)) * self.atoms.get_pbc()
                for i_xyz in [
                    (n1, 0, 0),
                    (-n1, 0, 0),
                    (0, n2, 0),
                    (0, -n2, 0),
                    (0, 0, n3),
                    (0, 0, -n3),
                ]:
                    if i_xyz == (0, 0, 0):
                        continue
                    i_xyz = torch.tensor(i_xyz)
                    R_c = center_position + i_xyz @ self.__cell  # atom-center's position
                    d_r_R_c = np.linalg.norm(points - R_c, axis=1)  # |r - R_c|
                    if True in (d_r_R_c < cutoff_radius):
                        print(
                            f"WARNING!! method<Grid.spherical_to_grid()>: 'num_near_cell'(={num_near_cell}) is not sufficent to interpolate the effects of nearing cells."
                        )

        is_batch_i_xyz = True  # no memory efficient
        # is_batch_i_xyz = False # no memory efficient
        if is_batch_i_xyz:
            cell = self.get_cell().to(device)
            points = self.points.to(device)
            i_xyzs = torch.tensor(
                list(
                    product(
                        range(-num_near[0], num_near[0] + 1),
                        range(-num_near[1], num_near[1] + 1),
                        range(-num_near[2], num_near[2] + 1),
                    )
                ),
                device=device,
                dtype=cell.dtype,
            )
            shift_batch = (i_xyzs @ cell).unsqueeze(dim=1)
            nshifts = len(shift_batch)
            del i_xyzs
            R_c = center_position.to(device)
            R_c_batch = R_c.repeat(nshifts, 1, 1) + shift_batch
            del shift_batch

            ## masking outside image cell indices
            # re-setting shift_batch by removing no-overlaped cells
            center_point = torch.from_numpy(self.__center_point).to(device)
            d_c = (R_c_batch - center_point).norm(dim=-1)  # shape=(nshipfts, 1)
            # d_c = (R_c_batch - center_point).pow(2).sum(-1).sqrt()  # slower
            is_overlap_cell = d_c <= self.get_radius() + cutoff_radius  # 에러, cutoff_radius가 None일때 처리.
            R_c_batch = R_c_batch[is_overlap_cell.squeeze(dim=1)]
            nshifts = len(R_c_batch)

            points_batch = points.repeat(nshifts, 1, 1)  # shape=(nshifts, ngpts, 3)
            d_r_R_c = (points_batch - R_c_batch).norm(dim=-1)  # |r - R_c|
            # d_r_R_c = (points_batch - R_c_batch).pow(2).sum(-1).sqrt()  # slower
            del points_batch

            ret_val = torch.zeros_like(d_r_R_c)

            ## masking outside cutoff radius indices
            mask = d_r_R_c < cutoff_radius if cutoff_radius else Ellipsis

            ## interpolation
            ret_val[mask] += torch.from_numpy(cubic_interp(d_r_R_c[mask].cpu())).to(device)

            if extplt:
                assert cutoff_radius is not None
                ## e.g., '-Z/r' -> '-Z/d_r_R_c[~mask]'
                extplt = extplt.replace("r", "d_r_R_c[~mask]")
                ret_val[~mask] = eval(extplt)

            if (
                cutoff_radius is None
                and extplt is None
                and True in (d_r_R_c > r[-1])
                and (abs(fn[-1]) > 1e-10)
            ):
                print(
                    "WARNING!! method<Grid.spherical_to_grid()>: The range of the interpolated function exceeds the defined radius."
                )
            ret_val = ret_val.sum(dim=0).squeeze()
            return ret_val

        return_val = np.zeros(self.__ngpts)
        for i_xyz in product(
            range(-num_near[0], num_near[0] + 1),
            range(-num_near[1], num_near[1] + 1),
            range(-num_near[2], num_near[2] + 1),
        ):
            R_c = center_position + i_xyz @ self.__cell

            if cutoff_radius:
                d = np.linalg.norm(R_c - self.__center_point)
                if self.get_radius() + cutoff_radius < d:
                    continue

            d_r_R_c = np.linalg.norm(self.__points - R_c, axis=1)  # |r - R_c|
            if cutoff_radius:
                idx = np.where(d_r_R_c < cutoff_radius)[0]
                return_val[idx] += cubic_interp(d_r_R_c[idx])
            else:
                return_val += cubic_interp(d_r_R_c)

            if extplt:
                assert cutoff_radius is not None
                extra_idx = np.where(d_r_R_c >= cutoff_radius)[0]
                extplt = extplt.replace(
                    "r", "d_r_R_c[extra_idx]"
                )  ## e.g., '-Z/r' -> '-Z/d_r_R_c[extra_idx]'
                return_val[extra_idx] = eval(extplt)

            if (
                cutoff_radius is None
                and extplt is None
                and True in (d_r_R_c > r[-1])
                and (abs(fn[-1]) > 1e-10)
            ):
                print(
                    "WARNING!! method<Grid.spherical_to_grid()>: The range of the interpolated function exceeds the defined radius."
                )

        return return_val

    def compute_grid_points_on_axis(self, endpoint=False):
        """Return grid points on each cartesian axis.

        :type  endpoint: bool, optional
        :param endpoint:
            whether end point is contained or not. default=False

        :rtype: tuple
        :return:
            tuple of three arrays
        """
        gpts = self.__gpts + 1 if endpoint else self.__gpts
        return tuple(np.arange(gpts[i]) * self.__spacings[i] for i in range(3))

    def _make_grid_points(self, indices):
        """Calculate grid points' coordinates (np.ndarray), shape=(ngpts, 3)"""
        points = indices.astype(float)
        if self.__is_cartesian:
            points *= self.__spacings
        else:
            points = points @ (self.__cell / self.__gpts.reshape(-1, 1))
        return points

    def _make_indices(self, order="C"):
        """Return indices for each point of three-dimensional grid"""
        indices = np.indices(self.__gpts, dtype=int).reshape((3, -1), order=order).T
        return indices

    def get_points(self, order="C"):
        """Return grid points coordinates (np.ndarray), shape=(ngpts, 3)"""
        if order == "C":
            return self.__points
        elif order == "F":
            points = self._make_grid_points(self._make_indices(order="F"))
            return points

    @property
    def indices(self):
        # return self.__indices
        return torch.from_numpy(self.__indices)

    @property
    def FD_order(self):
        """Finite-difference order (int)"""
        return self.__FD_order

    @property
    def points(self):
        """Grid points' coordinates (torch.tensor), shape=(ngpts, 3)"""
        # return self.__points
        return torch.from_numpy(self.__points)

    @property
    def spacings(self):
        """grid spacing of each axis (np.ndarray)"""
        return self.__spacings

    @property
    def gpts(self):
        """the number of grid points of each axis (torch.tensor)"""
        # return self.__gpts
        return torch.from_numpy(self.__gpts)

    @property
    def microvolume(self):
        """microvolume of grid (float), :math:`V = h_x h_y h_z`"""
        return self.__microvolume

    @property
    def grid_points_on_axis(self):
        return self.compute_grid_points_on_axis()

    @property
    def center_point(self):
        """coordinates of the center of the cell (np.ndarray)"""
        return self.__center_point

    @property
    def ngpts(self):
        """The total number of grid points (int)"""
        return self.__ngpts

    @property
    def cell_lengths(self):
        return self.__cell_lengths

    @property
    def is_cartesian(self):
        """Cartesian grid or not (bool)"""
        return self.__is_cartesian

    def get_cell(self):
        """Return cell matrix (torch.tensor), shape=(3, 3)"""
        # return self.__cell
        return torch.from_numpy(np.array(self.__cell))

    def get_pbc(self):
        """Return pbc (np.ndarray[bool])"""
        return self.atoms.get_pbc()

    def set_pbc(self, pbc):
        """Set pbc (np.ndarray[bool])"""
        assert len(pbc) == 3
        self.atoms.set_pbc(pbc)
        return


if __name__ == "__main__":
    import sys
    from ase import Atoms

    np.set_printoptions(threshold=sys.maxsize)

    atoms = Atoms("H", cell=[1, 1, 1])

    grid = Grid(atoms, gpts=(4, 4, 4))
    print(grid.center_point)
    print("grid.get_points = \n", grid.get_points())
