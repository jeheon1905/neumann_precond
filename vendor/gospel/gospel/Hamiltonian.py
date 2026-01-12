from typing import List, Union, Optional
import torch
import numpy as np

from gospel.LinearOperator import LinearOperator, MultiTypeLinearOperator
from gospel.FdOperators import make_kinetic_op, calc_kinetic
from gospel.util import scipy_to_torch_sparse, to_cuda
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.precision import to_DP, to_SP, to_HP, to_BF16
from gospel.util import Timer


class Hamiltonian:
    """Make, update, and diagonalize Hamiltonian.

    Args:
        nspins (int): The number of spins, 1 for 'unpolarized', 2 for 'polarized'.
        nbands (int): The number of bands.
        grid (gospel.Grid): Grid object.
        kpoint (gospel.Kpoint): Kpoint object.
        pp (gospel.Pseudopotential): Pseudopotential object.
        poisson_solver (gospel.Poisson.Poisson_solver): Poisson_solver object.
        xc_functional (gospel.XC.XC_Functional): XC_Functional object.
        eigensolver (gospel.Eigensolver): Eigensolver object.
        use_dense_kinetic (bool, optional):
            Whether to use dense kinetic operator. Defaults to False.
        device (torch.device, optional): Device to use. Defaults to None.
        multi_dtype (List[str], optional):
            List of precision types to use ('SP', 'HP', 'BF16', 'DP'). Defaults to None.
    """
    @Timer.timeit
    def __init__(
        self,
        nspins: int,
        nbands: int,
        grid,
        kpoint,
        pp,
        poisson_solver,
        xc_functional,
        eigensolver,
        use_dense_kinetic: bool = False,
        multi_dtype: List[str] = None,
        device: torch.device = None,
    ):
        self.__nspins = nspins
        self.__nbands = nbands
        self.__nkpts = kpoint.nkpts
        self.__nibzkpts = kpoint.nibzkpts
        self.__kpts = kpoint.kpts
        self.__gpts = grid.gpts
        self.__ngpts = grid.ngpts
        self.__shape = (self.__ngpts, self.__ngpts)

        self.grid = grid
        self.kpoint = kpoint
        self.pp = pp
        self.poisson_solver = poisson_solver
        self.xc_functional = xc_functional
        self.eigensolver = eigensolver

        self.__use_dense_kinetic = use_dense_kinetic
        self.__device = device
        self.__multi_dtype = multi_dtype

        # Initialization of Hamiltonian components
        #   Hamiltonian: H = T_s + V_H + V_xc + V_ext + V_NL
        #     T_s: kinetic operator for each k-point
        #     V_H: Hartree potential
        #     V_xc: XC potential
        #     V_ext: external potential
        #     V_NL: non-local potential
        #   Local potential: V_loc = V_H + V_xc + V_ext,
        #   where V_H and V_xc are changed during SCF cycles.
        self.__T_s = self.make_kinetic_operators(grid, self.__kpts, use_dense_kinetic)
        self.__V_H = torch.zeros(self.__ngpts, dtype=torch.float64)
        self.__V_xc = torch.zeros(self.__nspins, self.__ngpts, dtype=torch.float64)
        self.__V_ext = self.pp.V_ext.to(self.__device)
        self.__V_loc = torch.zeros(self.__nspins, self.__ngpts, dtype=torch.float64)

    @Timer.timeit
    def calc_forces(self, density, deriv_density=True, deriv_comp=True):
        """Calculate atomic Hellmann-Feynman forces.

        :type  density: gospel.Density
        :param density:
            Density class object
        :type  deriv_density: bool
        :param deriv_density:
            options for calculation of forces. derivatives of density or potential.
        :rtype: torch.Tensor
        :return:
            Hellmann-Feynman forces, shape=(natoms, 3)
        """
        F = self.pp.calc_forces(
            density,
            self.__V_H,
            # self.__V_H.cpu().numpy(),
            deriv_density=deriv_density,
            deriv_comp=deriv_comp,
        )
        return F

    def construct_hamiltonian_matrix(self, i_s, i_k):
        """Construct Hamiltonian matrix. (scipy.sparse) """
        from gospel.util import torch_diag_sparse
        #raise DeprecationWarning('construct_hamiltonian_matrix function is deprecated')
        assert not self.__use_dense_kinetic
        T = self.__T_s[i_k]
        V_loc = torch_diag_sparse(self.__V_loc[i_s])

        if self.pp.has_nonlocal:
            V_NL = self.pp.get_nonlocal_matrix(i_k)
            H = T + V_loc + V_NL.to_sparse_csr()
        else:
            H = T + V_loc
        non_zero_ratio = len(H.values()) / self.__ngpts**2 * 100
        print(f"Hamiltonian non-zero ratio (i_s={i_s}, i_k={i_k}) : {non_zero_ratio} %")
        return H

    def make_kinetic_operators(self, grid, kpts, use_dense_kinetic=False):
        """Make kinetic operators

        :rtype: list[tuple[torch.Tensor] or torch.Tensor]
        :return:
            list of kinetic operators for each k-point
        """
        T_s = []
        combine = False if use_dense_kinetic else True
        as_sparse = False if use_dense_kinetic else True
        for kpt in kpts:
            T_s.append(
                make_kinetic_op(grid, kpt, combine, as_sparse, device=self.__device)
            )
        return T_s

    def get_kinetic_operator(
        self,
        i_k: int,
        dtype: Optional[torch.dtype] = None,
    ):
        """Return kinetic operator corresponding to i_k-th k-point.

        :type  i_k: int
        :param i_k:
            index of k-point
        :type  dtype: torch.dtype or None, optional
        :param dtype:
            data type of kinetic operator

        :rtype: gospel.LinearOperator
        :return:
            kinetic operator
        """
        if self.__use_dense_kinetic:
            T_xx = self.__T_s[i_k][0].to(dtype=dtype)
            T_yy = self.__T_s[i_k][1].to(dtype=dtype)
            T_zz = self.__T_s[i_k][2].to(dtype=dtype)
            func = lambda x: calc_kinetic(self.grid.gpts, T_xx, T_yy, T_zz, x)
            dtype = T_xx.dtype
        else:
            T_s = to_cuda(self.__T_s[i_k], self.__device, dtype)
            func = lambda x: T_s @ x
            dtype = T_s.dtype

        op = LinearOperator(
            self.__shape,
            func,
            dtype=dtype,
            name=f"kinetic op (dtype={dtype})",
        )
        return op

    @Timer.timeit
    def get_linear_operator(
        self,
        i_s: int,
        i_k: int,
    ) -> Union[LinearOperator, MultiTypeLinearOperator]:
        """Return Hamiltonian operator as LinearOperator or MultiTypeLinearOperator"""
        if self.__multi_dtype is None or self.__multi_dtype == ["DP"]:
            return self._build_hamiltonian_operator(i_s, i_k)
        else:
            is_gamma = np.all(np.array(self.__kpts[i_k]) == 0)
            dtype = torch.float64 if is_gamma else torch.complex128

            dtype_converter = {"DP": to_DP, "SP": to_SP, "HP": to_HP, "BF16": to_BF16}
            dtype_list = [dtype_converter[_fp](dtype) for _fp in self.__multi_dtype]

            ops = [
                self._build_hamiltonian_operator(i_s, i_k, dtype=_dtype)
                for _dtype in dtype_list
            ]
            return MultiTypeLinearOperator(ops, "Hamiltonian (MP)")

    def _build_hamiltonian_operator(
        self,
        i_s: int,
        i_k: int,
        dtype: Optional[torch.dtype] = None,
        timing: Optional[bool] = True,  # TODO: make self.timing
        verbosity: Optional[bool] = False,  # TODO: make self.verbosity
    ) -> LinearOperator:
        """Return Hamiltonian operator.

        Args:
            i_s (int): Index of spin.
            i_k (int): Index of k-point.
            dtype (torch.dtype, optional): Data type of Hamiltonian operator. Defaults to None.
            timing (bool, optional): Whether to measure elapsed time of each component. Defaults to True.
            verbosity (bool, optional): Whether to print elapsed time of each component. Defaults to False.

        Returns:
            LinearOperator: Hamiltonian operator.
        """
        # Base operaters
        T = self.get_kinetic_operator(i_k, dtype)
        V_loc = self.__V_loc[i_s].reshape(-1, 1).to(self.__device, dtype)

        # Hamiltonian action
        def matvec(x):
            with Timer.track("Kinetic", timing, verbosity):
                retval = T @ x
            with Timer.track("V_loc", timing, verbosity):
                retval += V_loc * x
            if self.pp.has_nonlocal:
                with Timer.track("V_NL", timing, verbosity):
                    retval += self.pp.get_nonlocal_op(i_k, dtype)(x)
            return retval

        return LinearOperator(self.__shape, matvec, dtype, f"Hamiltonian ({dtype})")

    def diagonalize(self, convg_tol=1e-4, i_scf=None, bands=None, retHistory=False):
        """Diagonalize Hamiltonian

        :type convg_tol: float, optional
        :param convg_tol:
            tolerance of convergence of residual norm, defaults to 1e-4.
        :type  i_scf: int or None, optional
        :param i_scf:
            i-th iteration of SCF
        :type  bands: int or None, optional
        :param bands:
            number of lowest bands to check convergence, defaults to None (all bands)

        :rtype: tuple[np.ndarray, torch.Tensor]
        :return:
            eigenvalues and eigenvectors
        """
        return self.eigensolver.diagonalize(self, convg_tol, i_scf, bands, retHistory=retHistory)

    @Timer.timeit
    # def calc_energies(..., print=True)  # TODO: rename
    def calc_and_print_energies(self, density, eigval, eigvec, occ, echo=True):
        """Calculate and print total and each energy term

        :type  density: gospel.Density
        :param density:
            Density object
        :type  eigval: torch.Tensor
        :param eigval:
            eigenvalues, shape=(nspins, nibzkpts, nbands)
        :type  eigvec: np.ndarray[torch.Tensor]
        :param eigvec:
            eigenvectors, shape=(nspins, nibzkpts)---(nbands, ngpts)
        :type  occ: torch.Tensor
        :param occ:
            occupation number, shape=(nspins, nibzkpts, nbands)

        :rtype: float
        :return:
            Total energy with a.u.
        """

        ## Calculate E_{ext} and E_{NL}
        E_ext = self.pp.calc_external_energy(density)
        if self.pp.has_nonlocal:
            E_NL = self.pp.calc_nonlocal_energy(occ, eigvec)
        else:
            E_NL = 0.0

        ## Calculate E_{NN}
        E_NN = self.pp.ion_ion_repulsion_energy

        ## Calculate E_{kinetic}
        device = eigvec[0,0].device
        dtype = eigvec[0,0].dtype
        E_kin = torch.zeros(1, device=device, dtype=dtype )
        #list_i_bands = PH.split(torch.arange(self.__nbands, device=device))
        p_occ = PH.split(occ).to(device=device, dtype=dtype)

        for i_k in range(self.__nibzkpts):
            kinetic_op = self.get_kinetic_operator(i_k)
            for i_s in range(self.__nspins):
                E_kin += torch.sum( p_occ[i_s, i_k].unsqueeze(0) * ( eigvec[i_s, i_k].conj().T * (kinetic_op @ eigvec[i_s, i_k].T ) ).real )
        E_kin = PH.all_reduce(E_kin) 

#                for idx, i_band in enumerate(list_i_bands) :
#                #for i in range(self.__nbands):
#                    if abs(occ[i_s, i_k, i_band]) < 1e-7:
#                        continue
#                    val = eigvec[i_s, i_k][idx].conj() @ (kinetic_op @ eigvec[i_s, i_k][idx])
#    
#                    if val.dtype in [torch.complex64, torch.complex128]:
#                        assert abs(val.imag) < 1e-7
#                    E_kin += PH.all_reduce(occ[i_s, i_k, i_band] * val.real)

        ## Calculate E_{Hartree}
        E_H = (
            self.poisson_solver.compute_energy(
                density.get_density(compensated=True), self.__V_H
            )
            if self.poisson_solver != None
            else 0.0
        )

        ## Calculate E_{xc}
        Exc = self.xc_functional.get_Exc()

        
        ## 임시 ##
        E_NN = E_NN.detach().cpu() if isinstance(E_NN, torch.Tensor) else torch.tensor(E_NN, dtype=dtype)
        E_NL = E_NL.detach().cpu() if isinstance(E_NL, torch.Tensor) else torch.tensor(E_NL, dtype=dtype)
        Exc  = Exc.detach().cpu()  if isinstance(Exc,  torch.Tensor) else torch.tensor(Exc , dtype=dtype)
        ################################################################################

        ## E_{tot} = E_{kinetic} + E_{ext} + E_{NL} + E_{H} + E_{NN} + E_{xc}
        E_total = E_kin + E_ext + E_NL + E_H + E_NN + Exc
        eigval_sum = (occ * eigval).sum((1, 2))

        if (echo):
            ## print energies
            _round = lambda x: torch.round(x, decimals=10).item()
            print(f"\n{'':=<{12}} {'Energy (Hartree)'} {'':=<{14}}")
            print(f"|   {'Total Energy': <{20}}: {_round(E_total):<{17}}|")
            print("-" * 44)
            print(f"| * {'Ion-ion Energy': <{20}}: {_round(E_NN):<{17}}|")
            for i_s in range(len(eigval_sum)):
                print(
                    f"| * {'Eigenvals sum for '+str(i_s): <{20}}: {_round(eigval_sum[i_s]):<{17}}|"
                )
            print(f"| * {'Hartree Energy': <{20}}: {_round(E_H):<{17}}|")
            print(f"| * {'XC Energy': <{20}}: {_round(Exc):<{17}}|")
            print(f"| * {'Kinetic Energy': <{20}}: {_round(E_kin):<{17}}|")
            print(f"| * {'External Energy': <{20}}: {_round(E_ext):<{17}}|")
            print(f"| * {'Non-local Energy': <{20}}: {_round(E_NL):<{17}}|")
            print("=" * 44)
        return E_total

    def __getitem__(self, indx):
        i_s, i_k = indx
        if i_s >= self.__nspins or i_k >= self.__nibzkpts:
            raise IndexError
        else:
            return self.get_linear_operator(i_s, i_k)
            ## construct Hamiltonian matrix
            # return self.construct_hamiltonian_matrix(i_s, i_k)
            # return scipy_to_torch_sparse(self.construct_hamiltonian_matrix(i_s, i_k))

    @Timer.timeit
    def update(self, density):
        """Calculate and update Hartree and XC potentials from the input density.

        :type  density: gospel.Density
        :param density:
            Density class object
        """
        self.__V_H = self.poisson_solver.compute_potential(
            density.get_density(compensated=True)
        )
        self.xc_functional.compute(density.get_density(nlcc=True) )  
        self.__V_xc = self.xc_functional.get_Vxc()
        # V_ext is already moved on the device in constructor 
        # V_H is computed on the device 
        self.__V_loc = self.__V_H + self.__V_xc + self.__V_ext 
        return

    def get_S(self):
        """Return overlap operator"""
        if self.pp.get_S() is None:
            return None
        else:
            S = LinearOperator(
                self.__shape, self.pp.get_S(self.__device), name="overlap matrix"
            )
            return S

    @property
    def V_loc(self):
        return self.__V_loc

    @V_loc.setter
    def V_loc(self, inp):
        assert inp.shape == self.__V_loc.shape
        self.__V_loc = inp
        return

    @property
    def T_s(self, i_k=None):
        if i_k is None:
            return self.__T_s
        else:
            return self.__T_s[i_k]

    @property
    def shape(self):
        return self.__shape

    @property
    def ngpts(self):
        return self.__ngpts

    @property
    def gpts(self):
        return self.__gpts

    @property
    def nspins(self):
        return self.__nspins

    @property
    def nbands(self):
        return self.__nbands

    @property
    def nibzkpts(self):
        return self.__nibzkpts

    @property
    def kpts(self):
        return self.__kpts

    @property
    def nkpts(self):
        return self.__nkpts

    @property
    def device(self):
        return self.__device

    @property
    def multi_dtype(self):
        return self.__multi_dtype
