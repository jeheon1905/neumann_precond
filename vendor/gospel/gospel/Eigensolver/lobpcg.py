import warnings
from itertools import product
import numpy as np
import torch
import math
from gospel.Eigensolver.Eigensolver import Eigensolver
from gospel.Eigensolver.Eigensolver import gen_eigh as eigh
from gospel.util import timer
from gospel.LinearOperator import LinearOperator, aslinearoperator
import time

torch.backends.cuda.matmul.allow_tf32 = False
print(
    f"torch.backends.cuda.matmul.allow_tf32 : {torch.backends.cuda.matmul.allow_tf32}"
)


class LOBPCG(Eigensolver):
    def __init__(
        self,
        maxiter=5,
        verbosity=0,
        scheme=0,
        dynamic=False,
        dynamic_tol=None,
        dynamic_maxiter=None,
        dynamic_init=False,
        static_mask=True,
        verbose_every=1,
    ):
        super().__init__()
        self._type = "lobpcg"
        #self._maxiter = maxiter
        assert type(maxiter)==int or type(maxiter)==list, "maxiter type should be int or list of int"
        if type(maxiter)==list: 
            assert len(maxiter)==2, "if list is given for maxiter, its length should be 2"
            self._maxiter = maxiter
        else:
            self._maxiter = [maxiter]       
        self.__verbosity = verbosity
        self.__verbose_every = verbose_every
        self.__scheme = scheme
        self.__dynamic = dynamic
        self.__dynamic_tol = dynamic_tol
        self.__dynamic_maxiter = dynamic_maxiter
        self.__static_mask = static_mask
        return

    def __str__(self):
        s = str()
        s += f"\n===================== [ Eigensolver(LOBPCG) ] ===================="
        s += f"\n* maxiter         : {self._maxiter}"
        s += f"\n* verbosity       : {self.__verbosity}"
        s += f"\n* verbose_every   : {self.__verbose_every}"
        s += f"\n* scheme          : {self.__scheme}"
        s += f"\n* dynamic         : {self.__dynamic}"
        s += f"\n* dynamic_tol     : {self.__dynamic_tol}"
        s += f"\n* dynamic_maxiter : {self.__dynamic_maxiter}"
        s += f"\n* staticMask : {self.__static_mask}"
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
            # if self._preconditioner.device != A.device:
            #    self._preconditioner.set_device(A.device)
            solve_options = {
                "A": A,
                "X": self._starting_vector[i_s, i_k].to(eigval.device),
                "B": hamiltonian.get_S(device=eigval.device),
                "preconditioner": self.preconditioner, 
                "tol": convg_tol,
                #"maxiter": self._maxiter,
                "maxiter": self._maxiter[-1] if i_scf!=0 else self._maxiter[0],
                "largest": False,
                "scheme": self.__scheme,
                "dynamic": self.__dynamic,
                "dynamicTol": self.__dynamic_tol,
                "dynamicMaxiter": self.__dynamic_maxiter,
                "staticMask": self.__static_mask,
                "verbosityLevel": self.__verbosity,
                "verboseEvery": self.__verbose_every,
            }
            val, vec = lobpcg(**solve_options)
            #val = val.cpu()
            #vec = vec.cpu()
            eigval[i_s, i_k] = val
            eigvec[i_s, i_k] = vec.T

            ## update starting vectors and values
            self._starting_value[i_s, i_k] = val
            self._starting_vector[i_s, i_k] = vec
        return eigval, eigvec

    @property
    def dynamic(self):
        return self.__dynamic


def bmat(x):
    """Build a 2d-tensor from a nested sequence of 2d-tensor. Patch block 2d-tensors to return one large 2d-tensor.

    :type x: list[list[torch.Tensor]]
    :param x:
        Input data. Nested list of tensors.
    :rtype: torch.Tensor
    :return:
        block

    **Example**
    >>> A = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> B = torch.Tensor([[1.1, 2.1], [3.1, 4.1]])
    >>> C = torch.Tensor([[1.2, 2.2], [3.2, 4.2]])
    >>> D = torch.Tensor([[1.3, 2.3], [3.3, 4.3]])
    >>> bmat([[A, B],[C, D]])
    torch.Tensor([[1.0, 2.0, 1.1, 2.1],
                  [3.0, 4.0, 3.1, 4.1],
                  [1.2, 2.2, 1.3, 2.3],
                  [3.2, 4.2, 3.3, 4.3]])
    """
    return torch.cat([torch.cat(i, dim=1) for i in x], dim=0)


def _handle_gramA_gramB_verbosity(
    gramA, gramB, iterationNumber, verboseEvery, verbosityLevel
):
    if verbosityLevel > 1 and iterationNumber % verboseEvery == 0:
        _report_nonhermitian(gramA, "gramA")
        _report_nonhermitian(gramB, "gramB")
    if verbosityLevel > 10 and iterationNumber % verboseEvery == 0:
        # Note: not documented, but leave it in here for now
        torch.save(gramA, "gramA.pt")
        torch.save(gramB, "gramB.pt")


def _report_nonhermitian(M, name):
    """Report if `M` is not a Hermitian matrix given its type."""
    md = M - M.T.conj()
    nmd = torch.linalg.norm(md, 1)
    tol = 10 * torch.finfo(M.dtype).eps
    tol = max(tol, tol * torch.linalg.norm(M, 1))
    if nmd > tol:
        print(
            "matrix %s of the type %s is not sufficiently Hermitian:" % (name, M.dtype)
        )
        print("condition: %.e < %e" % (nmd, tol))
    return


def _as2d(ar):
    """If the input array is 2D return it, if it is 1D, append a dimension, making it a column vector."""
    if ar is None:
        return None
    if ar.ndim == 2:
        return ar
    elif ar.ndim == 1:
        aux = ar.view(-1, 1)
        return aux


def _applyConstraints(V, YBY, BY, Y):
    """Changes blockVectorV in place."""
    YBV = torch.matmul(BY.T.conj(), V)
    tmp = torch.linalg.cholesky_solve(YBV, YBY[0], upper=False)
    V -= torch.matmul(Y, tmp)


# NOTE: deprecated
def initialize(A, B, X, scheme, sizeX, largest):

    if X.is_complex():
        SP, DP = torch.complex64, torch.complex128
    else:
        SP, DP = torch.float32, torch.float64

    if scheme == 2:
        X = X.to(SP)

    # B-orthonormalize X.
    MP = scheme == 1
    X, BX = _b_orthonormalize(B, X, MP=MP)

    # Compute the initial Ritz vectors: solve the eigenproblem.

    if scheme == 0:
        AX = A(X)
    elif scheme == 1:
        AX = A.matvec32(X.to(SP)).to(DP)
    elif scheme == 2:
        AX = A.matvec32(X)

    if scheme == 1:
        gramXAX = torch.matmul(X.to(SP).T.conj(), AX.to(SP)).to(DP)
    else:
        gramXAX = torch.matmul(X.T.conj(), AX)

    eigval, eigsubvec = subspaceDiagonalize(gramXAX)

    ii = _get_indx(eigval, sizeX, largest)
    eigval = eigval[ii]

    eigsubvec = eigsubvec[:, ii]
    X = torch.matmul(X, eigsubvec)
    AX = torch.matmul(AX, eigsubvec)
    BX = torch.matmul(BX, eigsubvec) if B is not None else X
    R = AX - BX * eigval.unsqueeze(0)
    resnorm = torch.linalg.norm(R, dim=0)

    return X, BX, AX, eigval, resnorm, R


def subspaceDiagonalize(A, B=None):

    A = (A + A.T.conj()) / 2
    if B is not None:
        B = (B + B.T.conj()) / 2
    dev = A.device
    A = A.to("cpu")
    if B is not None:
        B = B.to("cpu")
    eigval, eigvec = eigh(A, B=B)
    eigval, eigvec = eigval.to(dev), eigvec.to(dev)
    return eigval, eigvec


def _b_orthonormalize(B, V, BV=None, retInvR=False, MP=False):
    """
    B-orthonormalize the given block vector using Cholesky.
    so that any :math:`v_i,v_j`  :math:`v_iBv_j=\delta_{ij}`

    :type B: torch.Tensor or gospel.LinearOperator
    :param B:
        A linear system of size (n, n) that is positive definite hermitian.
    :type blockVectorV: torch.Tensor
    :param blockVectorV:
        Input data of size (n, b), b < n.
        rank of this tensor should be exactly b,
        else it occurs cholesky decomposition error
    :type blockVectorBV: torch.Tensor, optional
    :param blockVectorBV:
        matmul results of B@V. Calculate matmul of B@V if it is None.
        Default is None.
    :type retInvR: bool, optional
    :param retInvR:
        return inverse of upper cholesky matrix of B and normalization term of V
        if it is True. Default False.
    :rtype: tuple
    :return:
        B-orthonormalized block vector, V, BV.
        if retInvR is True, return inverse of cholesky matrix and normalizer of V additionally.
    """
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


def _get_indx(_lambda, num, largest):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    ii = torch.argsort(_lambda, descending=largest)
    return ii[:num]


def check_verbosity(verbosityLevel, B, M, X, Y):
    if verbosityLevel:
        aux = "Solving "
        if B is None:
            aux += "standard"
        else:
            aux += "generalized"
        aux += " eigenvalue problem with"
        if M is None:
            aux += "out"
        aux += " preconditioning\n\n"
        aux += "matrix size %d\n" % X.shape[0]
        aux += "block size %d\n\n" % X.shape[1]
        if Y is None:
            aux += "No constraints\n\n"
        else:
            if Y.shape[1] > 1:
                aux += "%d constraints\n\n" % Y.shape[1]
            else:
                aux += "%d constraint\n\n" % Y.shape[1]
        warnings.filterwarnings(action="default")
        print(aux)


def lobpcg(
    A,
    X,
    B=None,
    preconditioner=None,
    Y=None,
    tol=None,
    maxiter=None,
    largest=True,
    verbosityLevel=0,
    verboseEvery=100,
    scheme=0,
    staticMask=True,
    numActiveLowerBound=0.0,
    dynamic=False,
    dynamicTol=None,
    dynamicMaxiter=None,
    retHistory=False,
    retHistoryDev=False,
    **kwargs
):
    if dynamic:
        if dynamicTol is None:
            tol = [
                    tol, 
                    max(1e-05, tol), 
                    max(1e-02, tol)
                    ]
        else:
            assert len(dynamicTol) == 3
            dynamicTol.sort()
            tol = [
                    dynamicTol[0], 
                    max(dynamicTol[1], 1e-05), 
                    max(dynamicTol[2], 1e-02)
                    ]
            
        niter = 0
        if retHistory:
            eigHistory = []
            resHistory = []
            eigHistoryDev = []
            resHistoryDev = []
        
        while scheme >= 0:
            if dynamicMaxiter is not None:
                maxiter = dynamicMaxiter[scheme]
            else:
                maxiter -= niter

            if maxiter == 0:
                if retHistory:
                    while len(eigHistory) != 3:
                        if retHistory:
                            if retHistoryDev:
                                eigHistory.append([])
                                eigHistoryDev.append([])
                                resHistory.append([])
                                resHistoryDev.append([])
                            else:
                                eigHistory.append([])
                                resHistory.append([])
                break

            res = lobpcg_(
                A,
                X,
                B=B,
                preconditioner=preconditioner,
                Y=Y,
                tol=tol[scheme],
                maxiter=maxiter,
                largest=largest,
                verbosityLevel=verbosityLevel,
                retHistory=retHistory,
                verboseEvery=verboseEvery,
                scheme=scheme,
                staticMask=staticMask,
                numActiveLowerBound=numActiveLowerBound,
            )
            
            scheme -= 1
            if retHistory:
                eig, X, eH, rH, niter = res
                eigHistory.append(eH)
                resHistory.append(rH)
            else:
                eig, X, niter = res
        
        if retHistory:
            res = eig, X, eigHistory, resHistory
        else:
            res = eig, X

    else:
        res = lobpcg_(
            A,
            X,
            B=B,
            preconditioner=preconditioner,
            Y=Y,
            tol=tol,
            maxiter=maxiter,
            largest=largest,
            verbosityLevel=verbosityLevel,
            retHistory=retHistory,
            verboseEvery=verboseEvery,
            scheme=scheme,
            staticMask=staticMask,
            numActiveLowerBound=numActiveLowerBound,
        )
    return res[:2]
    # return res[:-1]


def lobpcg_(
    A,
    X,
    B=None,
    preconditioner=None,
    Y=None,
    tol=1e-05,
    maxiter=None,
    largest=True,
    verbosityLevel=0,
    verboseEvery=100,
    scheme=0,
    staticMask=True,
    retHistory=False,
    numActiveLowerBound=0.0,
    SAVE_RESIDUAL=False,
):
    """Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

    LOBPCG is a preconditioned eigensolver for large symmetric positive
    definite (SPD) generalized eigenproblems.

    Parameters
    ----------
    :type A: torch.Tensor or LinearOperator
    :param A:
        The symmetric linear operator of the problem, usually a
        sparse matrix.  Often called the "stiffness matrix".
    :type X: torch.Tensor
    :param X:
        Initial approximation to the ``k`` eigenvectors (non-sparse). If `A`
        has ``shape=(n,n)`` then `X` should have shape ``shape=(n,k)``.
    :type B: torch.Tensor or LinearOperator, optional
    :param B:
        The right hand side operator in a generalized eigenproblem.
        By default, ``B = Identity``.  Often called the "mass matrix".
    :type M: torch.Tensor or LinearOperator, optional
    :param preconditioner:
        Preconditioner to `A`; by default ``M = Identity``.
        :math:`|M@A-I|_{F}` be sufficiently small
    :type Y: torch.Tensor or LinearOperator, optional
    :param Y:
        n-by-sizeY matrix of constraints (non-sparse), sizeY < n
        The iterations will be performed in the B-orthogonal complement
        of the column-space of Y. Y must be full rank.
    :type tol: scalar, optional
    :param tol:
        Solver tolerance (stopping criterion).
        The default is ``tol=n*sqrt(eps)``.
    :type maxiter: int, optional
    :param maxiter:
        Maximum number of iterations.  The default is ``maxiter = 20``.
    :type largest: bool, optional
    :param largest:
        If True, solve for the largest eigenvalues, otherwise the smallest.
    :type verbosityLevel: int, optional
    :param verbosityLevel:
        print intermediate solver output.  The default is ``verbosityLevel=0``.
    :type verboseEvery: int, optional
    :param verbosityEvery:
        print every this many iteration step. The default is ``verbosityEvery=300``.
    :type retLambdaHistory: bool, optional
    :param retLambdaHistory:
        Whether to return eigenvalue history.  Default is False.
    :type retResidualNormsHistory: bool, optional
    :param retResidualNormsHistory:
        Whether to return history of residual norms.  Default is False.
    :rtype: tuple
    :return:
        eigenvalue tensor, eigenvector tensor
    -------
    w : ndarray
        Array of ``k`` eigenvalues
    v : ndarray
        An array of ``k`` eigenvectors.  `v` has the same shape as `X`.
    lambdas : list of ndarray, optional
        The eigenvalue history, if `retLambdaHistory` is True.
    rnorms : list of ndarray, optional
        The history of residual norms, if `retResidualNormsHistory` is True.

    Notes
    -----
    If both ``retLambdaHistory`` and ``retResidualNormsHistory`` are True,
    the return tuple has the following format
    ``(lambda, V, lambda history, residual norms history)``.

    In the following ``n`` denotes the matrix size and ``m`` the number
    of required eigenvalues (smallest or largest).

    The LOBPCG code internally solves eigenproblems of the size ``3m`` on every
    iteration by calling the "standard" dense eigensolver, so if ``m`` is not
    small enough compared to ``n``, it does not make sense to call the LOBPCG
    code, but rather one should use the "standard" eigensolver, e.g. numpy or
    scipy function in this case.
    If one calls the LOBPCG algorithm for ``5m > n``, it will most likely break
    internally, so the code tries to call the standard function instead.

    It is not that ``n`` should be large for the LOBPCG to work, but rather the
    ratio ``n / m`` should be large. It you call LOBPCG with ``m=1``
    and ``n=10``, it works though ``n`` is small. The method is intended
    for extremely large ``n / m`` [4]_.

    The convergence speed depends basically on two factors:

    1. How well relatively separated the seeking eigenvalues are from the rest
       of the eigenvalues. One can try to vary ``m`` to make this better.

    2. How well conditioned the problem is. This can be changed by using proper
       preconditioning. For example, a rod vibration test problem (under tests
       directory) is ill-conditioned for large ``n``, so convergence will be
       slow, unless efficient preconditioning is used. For this specific
       problem, a good simple preconditioner function would be a linear solve
       for `A`, which is easy to code since A is tridiagonal.

    References
    ----------
    .. [1] A. V. Knyazev (2001),
           Toward the Optimal Preconditioned Eigensolver: Locally Optimal
           Block Preconditioned Conjugate Gradient Method.
           SIAM Journal on Scientific Computing 23, no. 2,
           pp. 517-541. :doi:`10.1137/S1064827500366124`

    .. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov
           (2007), Block Locally Optimal Preconditioned Eigenvalue Xolvers
           (BLOPEX) in hypre and PETSc. :arxiv:`0705.2626`

    .. [3] A. V. Knyazev's C and MATLAB implementations:
           https://bitbucket.org/joseroman/blopex

    .. [4] S. Yamada, T. Imamura, T. Kano, and M. Machida (2006),
           High-performance computing for exact numerical approaches to
           quantum many-body problems on the earth simulator. In Proceedings
           of the 2006 ACM/IEEE Conference on Supercomputing.
           :doi:`10.1145/1188455.1188504`
    """
    st = time.time()

    if X.ndim != 2:
        raise ValueError("expected rank-2 array for argument X")
    sizeY = Y.shape[1] if Y is not None else 0
    n, sizeX = X.shape
    assert (n - sizeY) >= (
        5 * sizeX
    ), "The problem size is small compared to the block size. Using dense eigensolver instead of LOBPCG."

    dev = X.device
    A = aslinearoperator(A)
    B = aslinearoperator(B)
    #M = aslinearoperator(M)
    #check_verbosity(verbosityLevel, B, M, X, Y)
    check_verbosity(verbosityLevel, B, preconditioner, X, Y)

    ## set defaults values
    scheme_ = ["DP", "MP", "SP"][scheme]
    MP = scheme == 1
    if X.is_complex():
        SP, DP = torch.complex64, torch.complex128
    else:
        SP, DP = torch.float32, torch.float64

    # Apply constraints to X.
    if Y is not None:
        if B is not None:
            BY = B(Y)
        else:
            BY = Y
        # gramYBY is a dense array.
        gramYBY = Y.T.conj() @ BY
        try:
            # gramYBY is a Cholesky factor from now on...
            gramYBY = torch.linalg.cholesky_ex(gramYBY, upper=False, check_errors=True)
        except RuntimeError as e:
            raise ValueError("cannot handle linearly dependent constraints") from e
        _applyConstraints(X, gramYBY, BY, Y)

    def get_mask(resnorm, mask, maskTolerance):
        ii = resnorm > maskTolerance
        if ii.float().mean() >= numActiveLowerBound:
            if not staticMask:
                mask = ii
            else:
                mask = mask & ii
        else:
            # The number of active should be at least numActiveLowerBound
            if not staticMask:
                N = min(math.ceil(sizeX * numActiveLowerBound), sizeX)
                _, idx = torch.topk(resnorm, N)
                mask = torch.zeros((sizeX,), dtype=bool, device=dev)
                mask[idx] = True
        return mask

    
    if scheme == 2:
        X = X.to(SP)
    # B-orthonormalize X.
    X, BX = _b_orthonormalize(B, X, MP=MP)

    # Compute the initial Ritz vectors: solve the eigenproblem.
    if scheme == 0:
        AX = A(X)
    elif scheme == 1:
        AX = A.matvec32(X.to(SP)).to(DP)
    elif scheme == 2:
        AX = A.matvec32(X)

    if scheme == 1:
        gramXAX = torch.matmul(X.to(SP).T.conj(), AX.to(SP)).to(DP)
    else:
        gramXAX = torch.matmul(X.T.conj(), AX)
    eigval, eigsubvec = subspaceDiagonalize(gramXAX)
    ii = _get_indx(eigval, sizeX, largest)
    eigval = eigval[ii]

    eigsubvec = eigsubvec[:, ii]
    X = torch.matmul(X, eigsubvec)
    AX = torch.matmul(AX, eigsubvec)
    BX = torch.matmul(BX, eigsubvec) if B is not None else X
    R = AX - BX * eigval.unsqueeze(0)
    resnorm = torch.linalg.norm(R, dim=0)
    
    mask = torch.ones((sizeX,), dtype=bool, device=dev)
    convergedResiduals = ~mask.clone()

    if retHistory:
        eigHistory = [eigval.to("cpu")]
        resHistory = [resnorm.to("cpu")]
        if SAVE_RESIDUAL:
            RHistory = [R.to("cpu")]

    ## Main iteration loop.
    previousBlockSize = sizeX
    ident = torch.eye(sizeX, dtype=X.dtype, device=dev)
    ident0 = torch.eye(sizeX, dtype=X.dtype, device=dev)
    P, AP, BP = None, None, None
    iterationNumber = 1
    restart = True
    explicitGramFlag = False
    currentBlockSize = mask.sum()
    cumBlockSize = 0

    # Check initial guess
    if (resnorm < tol).sum() == sizeX:
        print("Initial guess is already fine enough")
        print(f"Elapsed time[diagonalize-{scheme_}]: {time.time() - st} secs")
        print(f"Cumulative Blocks {scheme_}: {cumBlockSize}")
        print(f"final iteration {scheme_}: {iterationNumber}")
        print()
        if verbosityLevel > 0:
            print(f"final eigenvalue {scheme_}: {eigval}")
            print(f"final residual norms {scheme_}: {resnorm}")
            print(f"tol : {tol}")
        if X.dtype == SP:
            X, eigval = X.to(DP), eigval.double()
        if retHistory:
            # return eigval, X, eigHistory, resHistory, iterationNumber
            if SAVE_RESIDUAL:
                return eigval, X, eigHistory, resHistory, iterationNumber, RHistory
            else:
                return eigval, X, eigHistory, resHistory, iterationNumber
        else:
            return eigval, X, iterationNumber
    
    termination = None
    if verbosityLevel > 0 and verboseEvery != 1:
        print(
            f"iteration {iterationNumber}\n"
            f"current block size: {currentBlockSize}\n"
            f"eigenvalue: {eigval}\n"
            f"residual norms: {resnorm}\n"
            f"tol : {tol}\n"
        )

    while iterationNumber < maxiter:
        if currentBlockSize != previousBlockSize:
            previousBlockSize = currentBlockSize
            ident = torch.eye(currentBlockSize, dtype=X.dtype, device=dev)

        cumBlockSize += currentBlockSize
        if verbosityLevel > 0 and iterationNumber % verboseEvery == 0:
            print(
                f"iteration {iterationNumber}\n"
                f"current block size: {currentBlockSize}\n"
                f"eigenvalue: {eigval}\n"
                f"residual norms: {resnorm}\n"
                f"tol : {tol}\n"
            )

        activeR = _as2d(R[:, mask])
        if iterationNumber > 1:
            activeP = _as2d(P[:, mask])
            activeAP = _as2d(AP[:, mask])
            activeBP = _as2d(BP[:, mask]) if B is not None else None

        # TODO: now SP and MP are not available when preconditioning is not None.
        if preconditioner is not None:
            # Apply preconditioner T to the active residuals.
            if scheme == 0:
                # activeR = preconditioner(activeR)
                activeR = preconditioner(activeR, A, eigval[mask], None)
            elif scheme == 1:
                activeR = preconditioner(activeR, dtype=SP).to(DP)
            elif scheme == 2:
                activeR = preconditioner(activeR, dtype=SP)

        # Apply constraints to the preconditioned residuals.
        if Y is not None:
            _applyConstraints(activeR, gramYBY, BY, Y)

        # B-orthogonalize the preconditioned residuals to X.
        if B is not None:
            if scheme == 1:
                X32, BX32, activeR32 = X.to(SP), BX.to(SP), activeR.to(SP)
                XRX_ = torch.matmul(BX32.T.conj(), activeR32)
                XRX_ = torch.matmul(X32, XRX_)
                activeR = (activeR32 - XRX_).to(DP)
            else:
                XRX_ = torch.matmul(BX.T.conj(), activeR)
                XRX_ = torch.matmul(X, XRX_)
                activeR = activeR - XRX_
        else:
            if scheme == 1:
                X32, activeR32 = X.to(SP), activeR.to(SP)
                XRX_ = torch.matmul(X32.T.conj(), activeR32)
                XRX_ = torch.matmul(X32, XRX_)
                activeR = (activeR32 - XRX_).to(DP)
            else:
                XRX_ = torch.matmul(X.T.conj(), activeR)
                XRX_ = torch.matmul(X, XRX_)
                activeR = activeR - XRX_

        # B-orthonormalize the preconditioned residuals.
        aux = _b_orthonormalize(B, activeR, MP=MP)
        activeR, activeBR = aux

        if activeR is None:
            warnings.warn(
                f"Failed at iteration {iterationNumber} with accuracies {resnorm}\n"
                f"not reaching the requested tolerance {tol}.",
                UserWarning,
                stacklevel=2,
            )
            if scheme > 0:
                termination = "A"
                X, AX, BX = X.to(DP), AX.to(DP), BX.to(DP)
                eigval = eigval.double()
                iterationNumber += 1
            break

        if scheme == 0:
            activeAR = A(activeR)
        elif scheme == 1:
            activeAR = A.matvec32(activeR.to(SP)).to(DP)
        elif scheme == 2:
            activeAR = A.matvec32(activeR)
        if iterationNumber > 1:
            aux = _b_orthonormalize(B, activeP, BV=activeBP, retInvR=True, MP=MP)
            activeP, activeBP, invR, normal = aux

            if activeP is not None:
                activeAP = activeAP / normal
                activeAP = torch.matmul(activeAP, invR)
                restart = False

            else:
                restart = True
                warnings.warn(
                    f"Failed orthonormalization at iteration {iterationNumber}",
                    UserWarning,
                    stacklevel=1,
                )
                if scheme > 0:
                    termination = "A"
                    X, AX, BX = X.to(DP), AX.to(DP), BX.to(DP)
                    eigval = eigval.double()
                    iterationNumber += 1
                    break

        # Perform the Rayleigh Ritz Procedure:
        # Compute symmetric Gram matrices:
        myeps = 1e-8 if scheme == 0 and activeR.dtype == DP else 1e+1
        explicitGramFlag = True
        if not explicitGramFlag and resnorm.abs().max() < myeps:
            print("Changed To Explicit Gram Calculation")
        if resnorm.abs().max() > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else: # Once explicitGramFlag, forever explicitGramFlag.
            explicitGramFlag = True

        # Shared memory assingments to simplify the code
        if B is None:
            BX = X
            activeBR = activeR
            if iterationNumber > 1:
                activeBP = activeP

        # Common submatrices:
        if scheme == 1:
            X64, AX64, BX64 = X, AX, BX
            activeR64, activeAR64, activeBR64 = activeR, activeAR, activeBR
            X, AX, BX = X.to(SP), AX.to(SP), BX.to(SP)
            activeR = activeR.to(SP)
            activeAR = activeAR.to(SP)
            activeBR = activeBR.to(SP)

        gramXAX = torch.matmul(X.T.conj(), AX)
        gramXAR = torch.matmul(X.T.conj(), activeAR)
        gramRAR = torch.matmul(activeR.T.conj(), activeAR)

        if explicitGramFlag:
            gramXBX = torch.matmul(X.T.conj(), BX)
            gramRBR = torch.matmul(activeR.T.conj(), activeBR)
            gramXBR = torch.matmul(X.T.conj(), activeBR)
        else:
            gramXBX = ident0
            gramRBR = ident
            gramXBR = torch.zeros(
                    (sizeX, currentBlockSize), 
                    dtype=X.dtype, 
                    device=dev
                    )

        if not restart:
            if scheme == 1:
                activeP64, activeAP64 = activeP, activeAP
                activeP, activeAP = activeP.to(SP), activeAP.to(SP)
                if iterationNumber > 1:
                    activeBP64 = activeBP
                    activeBP = activeBP.to(SP)

            gramXAP = torch.matmul(X.T.conj(), activeAP)
            gramRAP = torch.matmul(activeR.T.conj(), activeAP)
            gramPAP = torch.matmul(activeP.T.conj(), activeAP)
            gramXBP = torch.matmul(X.T.conj(), activeBP)
            gramRBP = torch.matmul(activeR.T.conj(), activeBP)

            if explicitGramFlag:
                gramPBP = torch.matmul(activeP.T.conj(), activeBP)
            else:
                gramPBP = ident

            gramA = bmat(
                [
                    [gramXAX, gramXAR, gramXAP],
                    [gramXAR.T.conj(), gramRAR, gramRAP],
                    [gramXAP.T.conj(), gramRAP.T.conj(), gramPAP],
                ]
            )

            gramB = bmat(
                [
                    [gramXBX, gramXBR, gramXBP],
                    [gramXBR.T.conj(), gramRBR, gramRBP],
                    [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP],
                ]
            )

            if scheme == 1:
                gramA = gramA.to(DP)
                gramB = gramB.to(DP)
            _handle_gramA_gramB_verbosity(
                gramA, gramB, iterationNumber, verboseEvery, verbosityLevel
            )

            try:
                eigval, eigsubvec = subspaceDiagonalize(gramA, B=gramB)

            except Exception as e:
                warnings.warn(
                    f"Failed subspace diagonalization, at {iterationNumber} iter.\n"
                    f"error message : \n{e}",
                    UserWarning,
                    stacklevel=1,
                )
                # try again after dropping the direction vectors P from RR
                if scheme > 0:
                    explicitGramFlag = True
                restart = True

        if restart:
            gramA = bmat([[gramXAX, gramXAR], [gramXAR.T.conj(), gramRAR]])
            gramB = bmat([[gramXBX, gramXBR], [gramXBR.T.conj(), gramRBR]])
            _handle_gramA_gramB_verbosity(
                gramA, gramB, iterationNumber, verboseEvery, verbosityLevel
            )

            if scheme == 1:
                gramA = gramA.to(DP)
                gramB = gramB.to(DP)
            try:
                eigval, eigsubvec = subspaceDiagonalize(gramA, B=gramB)
            except Exception as e:
                if scheme > 0:
                    warnings.warn(
                        f"Failed subspace diagonalization, at {iterationNumber} iter.\n"
                        f"error message : \n{e}",
                        UserWarning,
                        stacklevel=1,
                    )
                    termination = "B"
                    X, AX, BX = X.to(DP), AX.to(DP), BX.to(DP)
                    eigval = eigval.double()
                    iterationNumber += 1
                    break
                raise ValueError("eigh has failed in lobpcg iterations") from e

        ii = _get_indx(eigval, sizeX, largest)
        eigval = eigval[ii]
        eigsubvec = eigsubvec[:, ii]

        if scheme == 1:
            X, AX, BX = X64, AX64, BX64
            activeR, activeAR, activeBR = activeR64, activeAR64, activeBR64
            if iterationNumber > 1:
                activeP, activeAP, activeBP = activeP64, activeAP64, activeBP64

        eigsubvecX = eigsubvec[:sizeX]
        eigsubvecR = eigsubvec[sizeX : sizeX + currentBlockSize]

        pp = torch.matmul(activeR, eigsubvecR)
        app = torch.matmul(activeAR, eigsubvecR)
        if not restart:
            eigsubvecP = eigsubvec[sizeX + currentBlockSize :]
            pp += torch.matmul(activeP, eigsubvecP)
            app += torch.matmul(activeAP, eigsubvecP)

        P, AP = pp, app
        X = torch.matmul(X, eigsubvecX) + pp
        AX = torch.matmul(AX, eigsubvecX) + app

        if B is not None:
            bpp = torch.matmul(activeBR, eigsubvecR)
            if not restart:
                bpp += torch.matmul(activeBP, eigsubvecP)
            BP = bpp
            BX = torch.matmul(BX, eigsubvecX) + bpp

        if B is not None:
            aux = BX * eigval.unsqueeze(0)
        else:
            aux = X * eigval.unsqueeze(0)
        R = AX - aux
        resnorm = torch.linalg.norm(R, dim=0)

        if retHistory:
            eigHistory.append(eigval.to("cpu"))
            resHistory.append(resnorm.to("cpu"))
            if SAVE_RESIDUAL:
                RHistory.append(R.to("cpu"))
        
        # Create Mask
        mask = get_mask(resnorm, mask, tol)
        currentBlockSize = mask.sum()
        if verbosityLevel > 1 and iterationNumber % verboseEvery == 0:
            print(f"activeMask : {mask}")

        # Convergence Check
        ii = resnorm < tol
        if staticMask:
            convergedResiduals = convergedResiduals | ii
        else:
            convergedResiduals = ii

        iterationNumber += 1
        if convergedResiduals.sum() == sizeX:
            if scheme > 0:
                termination = "C"
                X, AX, BX = X.to(DP), AX.to(DP), BX.to(DP)
                eigval = eigval.double()
            break

    if B is not None:
        aux = BX * eigval.unsqueeze(0)
    else:
        aux = X * eigval.unsqueeze(0)
    R = AX - aux
    resnorm = torch.linalg.norm(R, dim=0)
    
    termination = f"(termination type {termination})" if termination is not None else ""
    print(f"Elapsed time[diagonalize-{scheme_}]: {time.time() - st} secs {termination}")
    print(f"Cumulative Blocks {scheme_}: {cumBlockSize}")
    print(f"final iteration {scheme_}: {iterationNumber}")

    if verbosityLevel > 0:
        print(f"final eigenvalue {scheme_}: {eigval}")
        print(f"final residual norms {scheme_}: {resnorm}")
        print(f"tol : {tol}")

    if X.dtype == SP:
        X = X.to(DP)
        eigval = eigval.double()

    if retHistory:
        # return eigval, X, eigHistory, resHistory, iterationNumber
        if SAVE_RESIDUAL:
            return eigval, X, eigHistory, resHistory, iterationNumber, RHistory
        else:
            return eigval, X, eigHistory, resHistory, iterationNumber
    else:
        return eigval, X, iterationNumber
