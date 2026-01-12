from itertools import product
import numpy as np
import torch
from math import sqrt
from gospel.LinearOperator import LinearOperator, aslinearoperator
from gospel.Eigensolver.Eigensolver import Eigensolver, gen_eigh
from gospel.util import timer
import time


def vprint(*args, verbosity=True):
    if verbosity:
        for arg in args:
            print(arg, end="")
        print()
    else:
        pass


class Davidson(Eigensolver):
    """Block Davidson method. (https://arxiv.org/abs/0906.2569)

    :type maxiter: int, optional
    :param maxiter:
        the maximum number of iterations, defaults to 20
    :type nblock: int, optional
    :param nblock:
        the maximum size of block, defaults to 3
    :type locking: bool
    :param locking:
        whether locking converged eigenvectors or not
    :type verbosity: int, optional
    :param verbosity:
        verbose level (0 or 1), defaults 1
    :type fill_block: bool, optional
    :param fill_block:
        defaults to True
    """

    def __init__(
        self,
        maxiter=5,
        nblock=2,
        locking=True,
        verbosity=0,
        fill_block=True,
        dynamic=False,
        dynamic_tol=None,
        dynamic_maxiter=None,
    ):
        super().__init__()
        self._type = "davidson"
        assert type(maxiter)==int or type(maxiter)==list, "maxiter type should be int or list of int"
        if type(maxiter)==list:
            assert len(maxiter)==2, "if list is given for maxiter, its length should be 2"
            self._maxiter = maxiter
        else:
            self._maxiter = [maxiter]
        #self._maxiter = maxiter
        self.__nblock = nblock
        self.__locking = locking
        self.__verbosity = verbosity
        self.__fill_block = fill_block

        self.__dynamic = dynamic
        self.__dynamic_tol = dynamic_tol
        self.__dynamic_maxiter = dynamic_maxiter
        return

    def __str__(self):
        s = str()
        s += f"\n===================== [ Eigensolver(Davidson) ] ===================="
        s += f"\n* maxiter         : {self._maxiter}"
        s += f"\n* nblock          : {self.__nblock}"
        s += f"\n* locking         : {self.__locking}"
        s += f"\n* verbosity       : {self.__verbosity}"
        s += f"\n* fill_block      : {self.__fill_block}"
        s += f"\n* dynamic         : {self.__dynamic}"
        s += f"\n* dynamic_tol     : {self.__dynamic_tol}"
        s += f"\n* dynamic_maxiter : {self.__dynamic_maxiter}"
        s += f"\n====================================================================\n"
        return str(s)

    @timer
    def diagonalize(self, hamiltonian, convg_tol=1e-4, i_scf=None):
        """Diagonalize hamiltonians corresponding each spin and k-points.

        :type  hamiltonian: gospel.Hamiltonian
        :param hamiltonian:
            Hamiltonian class object
        :type convg_tol: float, optional
        :param convg_tol:
            tolerance of convergence of residual norm, defaults to 1e-4.
        :type  i_scf: int or None, optional
        :param i_scf:
            index of SCF iteration

        :rtype: (np.ndarray, np.ndarray)
        :return:
            eigenvalues and eigenvectors,
            eigval.shape=(nspins, nibzkpts, nbands)
            eigvec.shape=(nspins, nibzkpts)---(nbands, ngpts)
        """
        self._initialize_guess(hamiltonian)

        ## Solve eigenvalue problem
        eigval = torch.zeros_like(self._starting_value)
        eigvec = np.zeros_like(self._starting_vector)  # eigvec.dtype = object
        for i_s, i_k in product(range(eigvec.shape[0]), range(eigvec.shape[1])):
            A = hamiltonian[i_s, i_k]
            solve_options = {
                "A": A,
                "X": self._starting_vector[i_s, i_k].to(eigval.device),
                "B": hamiltonian.get_S(device=eigval.device),
                "preconditioner": self.preconditioner, 
                "tol": convg_tol,
                #"maxiter": self._maxiter,
                "maxiter": self._maxiter[-1] if i_scf!=0 else self._maxiter[0],
                "nblock": self.__nblock,
                "locking": self.__locking,
                "fill_block": self.__fill_block,
                # "dynamic": self.__dynamic,
                # "dynamicTol": self.__dynamic_tol,
                # "dynamicMaxiter": self.__dynamic_maxiter,
                "verbosityLevel": self.__verbosity,
                "i_scf": i_scf,
            }
            val, vec = davidson(**solve_options)
            #val, vec = val.cpu(), vec.cpu()
            eigval[i_s, i_k] = val
            eigvec[i_s, i_k] = vec.T  ## memory copy.. (not eifficient code)

            ## update starting vectors and values
            self._starting_value[i_s, i_k] = val
            self._starting_vector[i_s, i_k] = vec
        return eigval, eigvec

    @property
    def dynamic(self):
        return self.__dynamic


def davidson(
    A,
    X,
    B=None,
    preconditioner=None,
    tol=1e-4,
    maxiter=20,
    nblock=2,
    locking=True,
    fill_block=True,
    verbosityLevel=0,
    ortho="cholesky",
    retHistory=False,
    i_scf=0,
):
    """Block Davidson eigenvalue problem solver.

    :type A: LinearOperator
    :param A:
        Hamiltonian operator
    :type X: torch.Tensor
    :param X:
        guess eigenvectors
    :type B: LinearOperator or None, optional
    :param B:
        overlap matrix for generalized eigenvalue problem
    :type M: LinearOperator or None, optional
    :param M:
        precondition operator
    :type tol: float, optional
    :param tol:
        convergence tolerance, defaults to 1e-4.
    :type maxiter: int, optional
    :param maxiter:
        the maximum number of iterations, defaults to 20
    :type nblock: int, optional
    :param nblock:
        the maximum size of block, defaults to 2
    :type locking: bool, optional
    :param locking:
        whether locking converged eigenvectors or not, defaults to True
    :type dynamic: bool, optional
    :param dynamic:
        whether using dynamic precision method or not, defaults to False
    :type dynamicTol: float, optional
    :param dynamicTol:
        convergence tolerance for dynamic precision
        ('dynamicTol' is set to '(dimension) * 10**(-15/4)' as a default value.)
    :type dynamicMaxiter: int, optional
    :param dynamicMaxiter:
        the maximum number of iterations for dynamic precision, defaults to 10
    :type verbosityLevel: int, optional
    :param verbosityLevel:
        verbosity level, defaults to 0
    :type ortho: str, optional
    :param ortho:
        orthogonalize method, options=["qr", "cholesky"], defaults to "cholesky"

    :rtype: tuple[torch.Tensor]
    :return:
        eigenvalues and eigenvectors
    """
    neig = X.shape[-1]
    device = X.device
    fp_type = X.dtype
    A = aslinearoperator(A)
    B = aslinearoperator(B)
    #M = aslinearoperator(M)

    ## initializatioin
    X = _orthonormalize(X, method=ortho)
    AX = A @ X
    A_sub = X.T.conj() @ AX
    if B is None:
        BX = X  # shared data
        eigval, C = torch.linalg.eigh(A_sub)
    else:
        BX = B @ X
        B_sub = X.T.conj() @ BX
        eigval, C = gen_eigh(A_sub, B_sub)
    eigval, C = eigval[:neig], C[:, :neig]
    vprint("-*" * 30, "i_iter=1", verbosity=verbosityLevel)
    vprint(f"eigenvalue: {eigval}", verbosity=verbosityLevel)
    X = X @ C
    AX = AX @ C
    BX = BX @ C if B is not None else X

    ## Save convergence history
    eigHistory = []
    resHistory = []

    print(f"{AX}", flush=True)
    ## Start iteration
    find = False
    sub_dim = 0
    unlock = torch.full((neig,), True, device=device)
    preconditioner.reset_num_called()


    for i_iter in range(2, maxiter + 1):
        vprint("\n" + "=*" * 50, f"i_iter={i_iter}", verbosity=verbosityLevel)
        st = time.time()

        U = X
        AU_list = [AX]
        A_sub_list = [X.conj().T @ AX]        
        if B is not None:
            BU_list = [BX]
            B_sub_list = [X.conj().T @ BX]
        
        ## Start block expansion
        sub_dim = X.shape[-1]  # dimension of the subspace
        i_b = 1  # the number of expansions
        et = time.time()
        while True:
            st = time.time()
            ## Check the number of expansions
            if fill_block:
                if sub_dim == nblock * neig:
                    vprint(f"Stop block expansion.", verbosity=verbosityLevel)
                    break
            if not fill_block:
                if i_b == nblock:
                    vprint(
                        f"Stop block expansion. ({i_b} by {i_b})",
                        verbosity=verbosityLevel,
                    )
                    break

            ## Calculate residual
            R = AX[:, unlock] - eigval[unlock] * BX[:, unlock]
            residue = torch.linalg.norm(R, dim=0)
            vprint(f"residual norms: {residue}", verbosity=verbosityLevel)
            if retHistory and i_b == 1:
                eigHistory.append(eigval.to("cpu"))
                resHistory.append(residue.to("cpu"))

            ## Check convergence and lock converged states
            is_convg = residue < tol
            if locking:
                unlock[unlock.clone()] = ~is_convg
                R = R[:, ~is_convg]
            if torch.all(is_convg):
                find = True
                i_iter = i_iter - 1
                break

            eigval_ = eigval[unlock].clone()
            ## Check block is saturated
            if fill_block:
                if sub_dim == nblock * neig:
                    vprint(f"Stop block expansion.", verbosity=verbosityLevel)
                    break
                else:
                    if sub_dim + R.shape[1] <= nblock * neig:
                        pass
                    else:
                        cutBlock = nblock * neig - sub_dim
                        ii = torch.argsort(residue[~is_convg], descending=False)
                        R = R[:, ii < cutBlock]
                        eigval_ = eigval_[ii<cutBlock]
            ## Preconditioning
            R = preconditioner(R, A, eigval_, i_scf)
            del eigval_
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n',R.T.conj()@R)
            ## Orthonormalize R
            R -= U @ (U.conj().T @ R)
            R = _orthonormalize(R)
            AU_list.append(A @ R)
            if B is not None:
                BU_list.append(B @ R)

            ## Make A_sub_list
            for i in range(i_b + 1):
                A_sub_list.append(R.conj().T @ AU_list[i])
                if B is not None:
                    B_sub_list.append(R.conj().T @ BU_list[i])
            sub_dim_list = [AU.shape[1] for AU in AU_list]
            sub_dim = sum(sub_dim_list)

            U = torch.hstack([U, R])

            vprint(f"block size: {U.shape}", verbosity=verbosityLevel)
            del R

            ## Construct subspace matrices
            enum = 0
            A_sub = torch.zeros((sub_dim, sub_dim), dtype=fp_type, device=device)
            B_sub = torch.zeros_like(A_sub) if B is not None else None
            for i in range(i_b + 1):
                for j in range(i + 1):  # only fill lower triangle
                    st_i = sum(sub_dim_list[:i])
                    ed_i = sum(sub_dim_list[: i + 1])
                    st_j = sum(sub_dim_list[:j])
                    ed_j = sum(sub_dim_list[: j + 1])
                    A_sub[st_i:ed_i, st_j:ed_j] = A_sub_list[enum]
                    if B is not None:
                        B_sub[st_i:ed_i, st_j:ed_j] = B_sub_list[enum]
                    enum += 1
            ## subspace diagonalization and update eigenpairs
            if B is None:
                eigval, C = torch.linalg.eigh(A_sub)
            else:
                eigval, C = gen_eigh(A_sub, B_sub)
            eigval, C = eigval[:neig], C[:, :neig]
            vprint(f"eigenvalue: {eigval}", verbosity=verbosityLevel)
            X = U @ C
            AX = torch.hstack(AU_list) @ C
            BX = torch.hstack(BU_list) @ C if B is not None else X
            i_b += 1

        if find:
            if retHistory:
                eigHistory.append(eigval.to("cpu"))
                resHistory.append(residue.to("cpu"))
            break

    if find:
        vprint(
            f"* Block Davidson converged at (iter={i_iter}/{maxiter}, b={sub_dim}/{neig * nblock}) with residual norms: {residue}",
            verbosity=verbosityLevel,
        )
    else:
        vprint(
            f"Block Davidson do not converge with residual norms: {residue}",
            verbosity=verbosityLevel,
        )

    if retHistory:
        return eigval, X, eigHistory, resHistory
    else:
        return eigval, X


def _orthonormalize(X, method="cholesky"):
    if method == "cholesky":
        return _b_orthonormalize(None, X)[0]
    elif method == "qr":
        return torch.linalg.qr(X)[0]
    else:
        raise NotImplementedError


def _b_orthonormalize(B, V, BV=None, retInvR=False, MP=False):
    normalization = V.abs().max(axis=0)[0] + torch.finfo(V.dtype).eps
    V = V / normalization
    if BV is None:
        if B is not None:
            BV = B(V)
        else:
            BV = V  # Shared data!!!
    else:
        BV = BV / normalization

    if V.is_complex():
        SP, DP = torch.complex64, torch.complex128
    else:
        SP, DP = torch.float32, torch.float64

    if MP:
        if B is not None:
            V32, BV32 = V.to(SP), BV32.to(SP)
            VBV = torch.matmul(V32.T.conj(), BV32)
        else:
            V32 = V.to(SP)
            VBV = torch.matmul(V32.T.conj(), V32)
        VBV = VBV.to(DP)

    else:
        VBV = torch.matmul(V.T.conj(), BV)

    VBV = (VBV + VBV.T.conj()) / 2
    try:
        # decompose V.T@B@V into L@L.T. U = L.T
        LT = torch.linalg.cholesky(VBV, upper=True)
        LTinv = torch.linalg.inv(LT)
        V = torch.matmul(V, LTinv)
        # V' = V@Uinv -> V'BV'=Uinv.T.conj()@VBV@Uinv=I
        if B is not None:
            BV = torch.matmul(BV, LTinv)
        else:
            BV = None

    except:
        V = None
        BV = None
        LTinv = None

    if retInvR:
        return V, BV, LTinv, normalization

    else:
        return V, BV
