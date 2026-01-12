import torch
import numpy as np
from itertools import product
from gospel.util import timer, torch_diag_sparse, to_cuda
from gospel.FdOperators import make_kinetic_op
from gospel.Eigensolver.Eigensolver import Eigensolver

try:
    from tucy import Tucy

    AllowTucy = True
except:
    AllowTucy = False


class Tucker(Eigensolver):
    """Tucker decomposition method.

    :type maxiter: int, optional
    :param maxiter:
        the maximum number of iterations, defaults to 20
    :type tucker_iter: int
    :param tucker_iter:
        stop iteration to fix U matrices
    :type ranks: array[int]
    :param ranks:
        tucker ranks, e.g., ranks=(10, 10, 10)
    :type diag_method: str
    :param diag_method:
        name diagonalization method
    :type check_residue: str
    :param check_residue:
        whether check residue of restored eigenvectors or not, defaults to False
    :type local_shift: float
    :param local_shift:
        constant value to shift local potential, defaults to -500.
    :type mode: str
    :param mode:
        mode of tucker method, defaults to 'hosvd'

    Reference: J. Woo, W. Y. Kim, S. Choi, J. Chem. Theory Comput. 2022, 18, 5, 2875-2884
    """

    def __init__(
        self,
        maxiter=20,
        tucker_iter=1,
        ignore_kbproj=True,
        ranks=None,
        diag_method="lobpcg",
        check_residue=False,
        local_shift=-500.0,
        mode="hosvd",
    ):
        super().__init__()
        self._type = "tucker"
        self.__check_residue = check_residue
        assert type(maxiter)==int, "maxiter type should be int"
        self._maxiter = maxiter
        self.__tucker_iter = tucker_iter
        self.__ignore_kbproj = ignore_kbproj
        self.__ranks = ranks
        self.__diag_method = diag_method
        self.__local_shift = local_shift
        self.__mode = mode
        assert self.__mode in ["hosvd", "PW"], f"{self.__mode} is not supported."

        self.__tucker = None
        self.__local_potential = None
        return

    def __str__(self):
        s = str()
        s += "\n====================== [ Eigensolver(Tucker) ] ======================"
        s += f"\n* type          : tucker"
        s += f"\n* maxiter       : {self._maxiter}"
        s += f"\n* tucker_iter   : {self.__tucker_iter}"
        s += f"\n* ranks         : {self.__ranks}"
        s += f"\n* ignore_kbproj : {self.__ignore_kbproj}"
        s += f"\n* diag_method   : {self.__diag_method}"
        s += f"\n* local_shift   : {self.__local_shift}"
        s += f"\n* mode          : {self.__mode}"
        s += "\n=====================================================================\n"
        return s

    @timer
    def diagonalize(self, hamiltonian, convg_tol=1e-4, i_scf=0):
        """
        :type  hamiltonian: gospel.Hamiltonian
        :param hamiltonian:
            Hamiltonian class object
        :type convg_tol: float, optional
        :param convg_tol:
            tolerance of convergence of residual norm, defaults to 1e-4.
        :type  i_scf: int or None, optional
        :param i_scf:
            index of SCF iteration

        :rtype: tuple[np.ndarray, torch.Tensor]
        :return:
            eigenvalues and eigenvectors
        """

        print("==== Tucker Diagonalization ====")
        assert AllowTucy, "Tucy is not installed. Please check 'import Tucy' works fine"
        nspins = hamiltonian.nspins
        nibzkpts = hamiltonian.nibzkpts
        nbands = hamiltonian.nbands
        ngpts = hamiltonian.ngpts

        do_decompose = True if i_scf < self.__tucker_iter else False
        if do_decompose:
            if self.__tucker is None:
                self.__tucker = np.empty((nspins, nibzkpts), dtype=object)

            self.__local_potential = hamiltonian.V_loc

            ## Decompose
            for i_s, i_k in product(range(nspins), range(nibzkpts)):
                self.__tucker[i_s, i_k] = Tucy(hamiltonian.gpts, self.__ranks)
                dtype = hamiltonian.T_s[i_k].dtype
                if self.__mode == "PW":
                    self.__tucker[i_s, i_k].set_custom_Us(
                        self.__tucker[i_s, i_k].calculate_PW(kpt=[0, 0, 0])
                    )
                else:
                    if self.__ignore_kbproj:
                        tmp_hamiltonian = hamiltonian.T_s[i_k] + torch_diag_sparse(
                            hamiltonian.V_loc[i_s] + self.__local_shift, dtype=dtype
                        )
                    else:
                        tmp_hamiltonian = hamiltonian.construct_hamiltonian_matrix(
                            i_s, i_k
                        ) + torch_diag_sparse(
                            torch.full((ngpts,), self.__local_shift, dtype=dtype)
                        )
                    if hamiltonian.use_cuda:
                        tmp_hamiltonian = to_cuda(tmp_hamiltonian, torch.device("cuda"))
                    self.__tucker[i_s, i_k].decompose(tmp_hamiltonian)

            ## Project kinetic operator and KB prjectors
            D_matrix = hamiltonian.pp.KB_form.get_D_matrix()
            kb_proj = hamiltonian.pp.KB_form.get_KB_proj()
            for i_k in range(nibzkpts):
                kpt = hamiltonian.kpts[i_k]
                T_xx, T_yy, T_zz = make_kinetic_op(
                    hamiltonian.grid,
                    kpt=kpt,
                    combine=False,
                    as_sparse=False,
                )
                for i_s in range(nspins):
                    self.__tucker[i_s, i_k].project_kinetic_to_core(T_xx, T_yy, T_zz)
                    self.__tucker[i_s, i_k].project_local_potential_to_core(
                        self.__local_potential[i_s]
                    )
                    self.__tucker[i_s, i_k].project_KB_projectors_to_core(
                        D_matrix, kb_proj[i_k]
                    )
        else:
            ## Project difference of local potential between current and previous steps.
            local_potential_diff = hamiltonian.V_loc - self.__local_potential
            self.__local_potential = hamiltonian.V_loc
            for i_s, i_k in product(range(nspins), range(nibzkpts)):
                self.__tucker[i_s, i_k].project_local_potential_to_core(
                    local_potential_diff[i_s]
                )

        ## Diagonalize
        eigval = torch.zeros(nspins, nibzkpts, nbands, dtype=torch.float64)
        eigvec = np.empty((nspins, nibzkpts), dtype=object)
        for i_s, i_k in product(range(nspins), range(nibzkpts)):
            val, vec = self.__tucker[i_s, i_k].core_diagonalize(
                nbands,
                maxiter=self._maxiter,
                tol=convg_tol,
            )
            eigval[i_s, i_k], eigvec[i_s, i_k] = val, vec.T

        ## Check residue
        if self.__check_residue:
            self.check_residue(hamiltonian, eigval, eigvec)

        return eigval, eigvec

    @property
    def tucker(self):
        """Return Tucy object"""
        return self.__tucker
