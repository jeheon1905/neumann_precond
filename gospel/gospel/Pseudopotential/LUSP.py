import torch
import numpy as np
from gospel.Pseudopotential.Pseudopotential import Pseudopotential
from gospel.util import timer
from gospel.LinearOperator import LinearOperator


class LUSP(Pseudopotential):
    """Local Ultra-Soft Pseudopotential"""

    def __init__(
        self,
        grid,
        upf,
        NLCC=True,
        use_comp=True,
        filtering=False,
        num_near_cell=1,
        kpts=[[0, 0, 0]],
    ):
        super().__init__(
            grid, upf, NLCC, use_comp, filtering, num_near_cell, kpts
        )
        assert upf.pseudo_type == "LUSP"
        self.__augment = torch.from_numpy(
            self.calc_augment_function(self._grid, self._atoms, upf).reshape(-1, 1)
        )
        return

    def calc_augment_function(self, grid, atoms, upf):
        """Read augment function from upf and represent it on grid."""
        augment = np.ones(grid.ngpts)
        for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
            R_i = atoms.get_positions()[i_atom]
            augment += grid.spherical_to_grid(
                upf[i_sym].R,
                upf[i_sym].aug,
                R_i,
                cutoff_radius=upf[i_sym].aug_cutoff,
                num_near_cell=self._num_near_cell,
            )
        return augment

    def calc_forces(
        self,
        density,
        potential=None,
        deriv_density=True,
        deriv_comp=True,
    ):
        raise NotImplementedError

    def get_S(self, device=None):
        """Return overlap operator S

        :type  device: torch.device, optional
        :param device:
            device info, defaults to None
        :rtype: function
        :return:
            overlap operator S
        """
        aug = self.__augment.to(device)
        S = lambda x: aug * x
        return S

    def __str__(self):
        s = str()
        s += "\n========================= [ Pseudopotential ] ========================="
        s += "\n* type      : LUSP"
        s += "\n* upf       : "
        for filename in self._upf.filenames:
            s += "\n    " + filename
        s += f"\n* NLCC      : {self._NLCC}"
        s += f"\n* use_comp  : {self._use_comp}"
        s += f"\n* filtering : {self._filtering}"
        s += "\n=======================================================================\n"
        return s


if __name__ == "__main__":
    from ase import Atoms
    from gospel.Grid import Grid
    from gospel.Pseudopotential.UPF import UPF

    atoms = Atoms("Si")
    atoms.center(vacuum=3.0)
    grid = Grid(atoms, gpts=(10, 10, 10))
    upf = UPF(["/home/share/DATA/NNLP/LUSP_UPF/Si_3s_2.1403_10.0_QE.UPF"])
    lusp = LUSP(grid, upf)
    print(lusp)

    V_ext = lusp.V_ext
    print(V_ext.shape)
