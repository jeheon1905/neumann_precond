import torch
from itertools import product
import numpy as np
from gospel.util import timer
from gospel.ParallelHelper import ParallelHelper as PH


class Density:
    """
    This class manages density-related tasks.
    e.g., density's calculation, normalization, initialization, etc.

    :type  grid: gospel.Grid
    :param grid:
        Grid object
    :type  pp: gospel.Pseudopotential
    :param pp:
        Pseudopotential object
    :type  nelec: float
    :param nelec:
        the number of electrons
    :type  nspins: int
    :param nspins:
        the number of spins. (1 for 'unpolarized' and 2 for 'polarized')
    :type  magmom: float, optional
    :param magmom:
        magnetic momentum, defaults to 0.0
    :type  fix_magmom: bool, optional
    :param fix_magmom:
        whether to fix the magmom or not, defaults to True
    :type  init_type: dict or None
    :param init_type:
        initial guess type of densities or eigenvectors, defaults to None,
        e.g., init_type = {"density": "rhoatom", "orbital": None, "occ": None}
    :type device: torch.device, optional
    :param device:
        device to use for computation, defaults to None
    """

    def __init__(
        self, grid, pp, nelec, nspins, magmom=0.0, fix_magmom=True, init_type=None, device=None,
    ):
        self.__grid = grid
        self.__pp = pp
        self.__nelec = nelec
        self.__nspins = nspins
        self.__magmom = magmom
        self.__fix_magmom = fix_magmom
        self.__init_type = init_type
        self.__device = device

        self.__occ = None
        self.__eigvec = None
        if self.__init_type["orbital"] is not None:
            self._init_orbital()
            self._init_occ()
        self.__val = None  # should be set by self.set_density()

        self.__comp_charge = pp.get_comp_charge() 
        self.__NLCC_core_density = pp.calc_NLCC_core_density() / self.__nspins if pp.NLCC else 0
        return

    def __str__(self):
        s = str()
        s += "\n============================ [ Density ] ==========================="
        s += f"\n* nelec         : {self.__nelec}"
        s += f"\n* nspins        : {self.__nspins}"
        s += f"\n* magmom        : {self.__magmom}"
        s += f"\n* fix_magmom    : {self.__fix_magmom}"
        s += f'\n* density guess : {self.__init_type["density"]}'
        s += f'\n* orbital guess : {self.__init_type["orbital"]}'
        s += f'\n* occ guess     : {self.__init_type["occ"]}'
        s += "\n=====================================================================\n"
        return s

    def calc_density(self, eigvec, occ, S=None):
        """
        Calculate charge density from eigvec and occ, and save it as class object.
        Additionaly, save eigvec and occ as class object.

        :type  eigvec: np.ndarray[torch.Tensor]
        :param eigvec:
            eigenvectors, shape=(nspins, nibzkpts)---(nbands, ngpts)
        :type  occ: torch.Tensor
        :param occ:
            occupation numbers, shape=(nspins, nibzkpts, nbands)
        :type  S: gospel.LinearOperator or None, optional
        :param S:
            overlap matrix, defaults to None

        :rtype: None
        :return:
            No return, save density as class variable
        """
        self.__eigvec = eigvec
        self.__occ = occ
        nspins = occ.shape[0]
        nibzkpts = occ.shape[1]

        self.__val = torch.zeros_like(self.__val)
        p_occ = PH.split(occ).to(eigvec[0,0].device)

        assert p_occ.shape[-1] == eigvec[0,0].shape[0], f"cal_density {occ.shape} {p_occ.shape}, {eigvec[0,0].shape}"
        for i_s, i_k in product(range(nspins), range(nibzkpts)):
            if S is None:
                self.__val[i_s] += (
                    p_occ[i_s, i_k].reshape(-1, 1)
                    * (eigvec[i_s, i_k].conj() * eigvec[i_s, i_k]).real
                ).sum(0)
            else:
                self.__val[i_s] += (
                    p_occ[i_s, i_k].reshape(-1, 1)
                    * (eigvec[i_s, i_k].conj() * (S @ eigvec[i_s, i_k].T).T).real
                ).sum(0)

                ########################################################################
                natom = len(self.__grid.atoms)
                if natom == 1:
                    print("")
                    for i in range(len(eigvec[i_s, i_k])):
                        if p_occ[i_s, i_k][i] < 1e-5:
                            continue
                        tmp = (
                            eigvec[i_s, i_k][i].conj() * eigvec[i_s, i_k][i]
                        ).real.sum(0)
                        print(f"Debug: no augmented orbital density sum (i={i})= {tmp}")
                else:
                    print("Debug: Check Augmentation!!")
                    print(f"Debug: i_s, i_k = {i_s}, {i_k}")
                    tmp = (
                        p_occ[i_s, i_k].reshape(-1, 1)
                        * (eigvec[i_s, i_k].conj() * eigvec[i_s, i_k]).real
                    ).sum(0)
                    tmp = tmp.sum()
                    print(f"Debug: no augmented density sum = {tmp}")
                ########################################################################
        self.__val /= self.__grid.microvolume
        self.__val = PH.all_reduce(self.__val) # eigenvectors are distributed therefore they should be summed up

        # ##################################################################
        # ## Test symmetrize
        # from ase.build import bulk
        # from _gpaw import symmetrize, symmetrize_ft
        # import spglib
        # _atoms = bulk("C", "diamond", cubic=True, a=3.57)
        # symmetry = spglib.get_symmetry(_atoms)
        # idx = np.where(np.all(symmetry["translations"] == [0, 0, 0], axis=1))
        # op_scc = symmetry["rotations"][idx]
        # op_scc = op_scc.astype(np.int64)

        # for i in range(len(self.__val)):
        #     A_g = self.__val[i].numpy().reshape(*self.__grid.gpts)
        #     B_g = np.zeros_like(A_g)
        #     for op_cc in op_scc:
        #         b_g = np.zeros_like(A_g)
        #         print(f"Debug: op_cc=\n{op_cc}")
        #         # symmetrize(A_g, B_g, op_cc, 1 - _atoms.get_pbc())
        #         symmetrize(A_g, b_g, op_cc, 1 - _atoms.get_pbc())
        #         print(f"Debug: integral(|A_g - b_g|): {self.__grid.integrate(abs(A_g - b_g).reshape(-1))}")
        #         B_g += b_g
        #     B_g /= len(op_scc)
        #     self.__val[i] = torch.from_numpy(B_g.reshape(-1))
        # ##################################################################

        self.__val = self.normalize(self.__val)
        return

    def _init_orbital(self):
        init_type = self.__init_type["orbital"]
        if ".npy" in init_type:
            self.__eigvec = np.load(init_type, allow_pickle=True)
        elif ".pt" in init_type:
            self.__eigvec = torch.load(init_type, map_location=torch.device("cpu"))
        else:
            raise NotImplementedError
        return

    def _init_occ(self):
        init_type = self.__init_type["occ"]
        if ".npy" in init_type:
            self.__occ = np.load(init_type, allow_pickle=True)
        elif ".pt" in init_type:
            self.__occ = torch.load(init_type, map_location=torch.device("cpu"))
        else:
            raise NotImplementedError
        return

    @timer
    def init_density(self):
        """Initialize guess density"""
        if self.__val is not None:
            print("Density: Returns the already initialized density values.")
            return self.__val

        nspins = self.__nspins
        ngpts = self.__grid.ngpts
        device = self.__device

        init_density = torch.zeros(
            nspins, ngpts, dtype=torch.float64, device=device,
        )

        init_type = self.__init_type["density"]
        if isinstance(init_type, str):
            if init_type == "rhoatom":
                init_density += self.__pp.init_density(device) / nspins
                print(
                    "integral of init_density : ",
                    self.__grid.integrate(init_density).squeeze(),
                )
            elif init_type == "random":
                init_density = torch.rand(*init_density.shape, dtype=torch.float64, device=device)
            elif ".npy" in init_type:
                init_density = np.load(init_type, allow_pickle=True)
                init_density = torch.from_numpy(init_density).to(device)
            elif ".pt" in init_type:
                init_density = torch.load(init_type, map_location=torch.device("cpu"))
                init_density = init_density.to(device)
            else:
                raise ValueError(f"'{init_type}' is not valid.")
        elif isinstance(init_type, torch.Tensor):
            init_density = init_type.to(device)
        elif isinstance(init_type, np.ndarray):
            init_density = torch.from_numpy(init_type).to(device)
        else:
            raise ValueError(f"'{init_type}' is not valid.")
        init_density = self.normalize(init_density)

        assert isinstance(init_density, torch.Tensor)
        return init_density

    def normalize(self, density):
        ## Normalize density to reduce numerical errors
        ret_density = density
        integral_density = self.__grid.integrate(ret_density)
        if self.__fix_magmom and self.__magmom:
            assert self.__nspins == 2
            ret_density[0] *= (
                (self.__nelec + self.__magmom) / self.__nspins / integral_density[0]
            )
            ret_density[1] *= (
                (self.__nelec - self.__magmom) / self.__nspins / integral_density[1]
            )
        else:
            ret_density *= self.__nelec / self.__nspins / integral_density
        print(
            f"Integral of density (={integral_density.squeeze()}) is normalized to {self.__grid.integrate(ret_density).squeeze()}"
        )
        return ret_density

    @timer
    def get_density(self, nlcc=False, compensated=False):
        """Return Density

        :type  nlcc: bool, optional
        :param nlcc:
            whether to add non-linear core correction of pseudopotential, defaults to False
        :type  compensated: bool, optional
        :param compensated:
            whether to add the compensation charge, defaults to False

        :rtype: torch.Tensor
        :return:
            charge density, shape=(nspins, ngpts)
        """
        if nlcc:
            return self.__val + self.__NLCC_core_density  # shape=(nspins, ngpts)
        elif compensated:
            ret_val = self.__val.sum(0) - self.__comp_charge
            integral = self.__grid.integrate(ret_val)
            if abs(integral) > 1e-6 and self.__pp.use_comp:
                print(
                    f"Warning: Integral of compensated charge density (={integral}) should be zero."
                )
            return ret_val  # shape=(ngpts,)
        else:
            return self.__val  # shape=(nspins, ngpts)

    def set_density(self, inp):
        self.__val = inp

    def get_magmom(self):
        raise NotImplementedError

    def calc_dipole_moment(self):
        raise NotImplementedError

    @property
    def nelec(self):
        return self.__nelec

    @property
    def occ(self):
        return self.__occ

    @property
    def eigvec(self):
        return self.__eigvec

    @property
    def use_cuda(self):
        return self.__use_cuda

    @property
    def device(self):
        return self.__device