"""
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).

References
----------
.. [1] A. V. Knyazev (2001),
       Toward the Optimal Preconditioned Eigensolver: Locally Optimal
       Block Preconditioned Conjugate Gradient Method.
       SIAM Journal on Scientific Computing 23, no. 2,
       pp. 517-541. :doi:`10.1137/S1064827500366124`

.. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov (2007),
       Block Locally Optimal Preconditioned Eigenvalue Xolvers (BLOPEX)
       in hypre and PETSc.  :arxiv:`0705.2626`

.. [3] A. V. Knyazev's C and MATLAB implementations:
       https://github.com/lobpcg/blopex
"""

import warnings
import math
import torch
import functools

from torch.linalg import inv
from torch.linalg import eigh as eigh_
from torch.linalg import norm
from scipy.linalg import LinAlgError, eigh as eigh_scipy

from LinearOperator import LinearOperator, aslinearoperator

import time
## accumulated time 
AR_time = 0.0
init_time = 0.0
AX_time = 0.0
orthonormalize_time = 0.0
norm_in_ortho = 0.0
VBV_in_ortho = 0.0
cho_inv_in_ortho = 0.0
restore_in_ortho = 0.0
sub_time = 0.0



### Documentation Format
"""Integrate a function represented on grid.

:type  values: np.ndarray
:param values:
    values of the function on grid,
    shape='(ngpts,)' or '(n, ngpts)' or '(n,) with dtype=object'

:rtype: float or np.ndarray
:return:
    integrated value
"""

def eigh_x(A, B=None):
    a = A.to('cpu').numpy()
    b = B.to('cpu').numpy() if B is not None else None

    eigval, eigvec = eigh_scipy(a, b=b, check_finite=False, )
    eigval = torch.from_numpy(eigval).to(A.device)
    eigvec = torch.from_numpy(eigvec).to(A.device)
    return eigval, eigvec


def subspaceDiagonalize(A, B=None):
    A = (A + A.T.conj())/2
    if B is not None:
        B = (B + B.T.conj())/2

    eigval, eigvec = eigh(A, B=B)
    return eigval, eigvec


def eigh(A, B=None):
    """
    Generalized eigenvalue problem solver for a complex Hermitian or real symmetric
    matrix, where B is positive-definite matrix. Solve :math:`Ax=\lambda Bx`, with
    cholesky decomposition :math:`L^{-1}AL^{*-1}\ L^*x=\lambda x\ s.t B=LL^*`
    When B is not defined, it is assumed identity matrix.
    :type  A: torch.Tensor
    :param A:
        tensor of shape (n,n), hermitian or real symmetric matrix
    :type B: torch.Tensor, optional
    :param B:
        tensor of shape (n,n), hermitian or real symmetric positive definite matrix.
        Assumed identity matrix of same shape of A if None. Default: None.
    :rtype: tuple
    :return:
        tuple of eigenvalue tensor and eigenvector tensor
    """
    
    if B is not None:
        L = torch.linalg.cholesky(B)
        Linv = inv(L)
        _A = (Linv @ A) @ Linv.conj().T
        eigval, eigvec = eigh_(_A)
        eigvec = Linv.conj().T @ eigvec
    else:
        eigval, eigvec = eigh_(A)

    return eigval, eigvec


def bmat(x):
    """
    Build a 2d-tensor from a nested sequence of 2d-tensor. Patch block 2d-tensors to
    return one large 2d-tensor.
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


def _report_nonhermitian(M, name):
    """
    Report if `M` is not a Hermitian matrix given its type.
    """

    md = M - M.T.conj()

    nmd = norm(md, 1)
    tol = 10 * torch.finfo(M.dtype).eps
    tol = max(tol, tol * norm(M, 1))
    if nmd > tol:
        print(
            "matrix %s of the type %s is not sufficiently Hermitian:" % (name, M.dtype)
        )
        print("condition: %.e < %e" % (nmd, tol))


def _as2d(ar):
    """
    If the input array is 2D return it, if it is 1D, append a dimension,
    making it a column vector.
    """
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


def _b_orthonormalize(B, V, BV=None, retInvR=False):
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
    global orthonormalize_time
    global norm_in_ortho
    global VBV_in_ortho
    global cho_inv_in_ortho
    global restore_in_ortho
    st = time.time()

    normalization = V.abs().max(axis=0)[0] + torch.finfo(V.dtype).eps
    V = V / normalization
    if BV is None:
        if B is not None:
            BV = B(V)
        else:
            BV = V  # Shared data!!!
    else:
        BV = BV / normalization
    
    VBV = torch.matmul(V.T.conj(), BV)
    VBV = (VBV + VBV.T.conj()) / 2
    try:
        # decompose V.T@B@V into L@L.T. U = L.T
        LT = torch.linalg.cholesky(VBV, upper=True)
        LTinv = inv(LT,)
        # V' = V@Uinv -> V'BV'=Uinv.T.conj()@VBV@Uinv=I
        # RBR = LTinv.conj().T@VBV@LTinv
        V = torch.matmul(V, LTinv)
        # blockVectorV = (cho_solve((VBV.T, True), blockVectorV.T)).T
        if B is not None:
            BV = torch.matmul(BV, LTinv)
            # blockVectorBV = (cho_solve((VBV.T, True), blockVectorBV.T)).T
        else:
            BV = None
    # except LinAlgError:
    except:
        # raise ValueError("Cholesky has failed")
        V = None
        BV = None
        LTinv = None


    if retInvR:
        return V, BV, LTinv, normalization
    else:
        return V, BV


def _get_indx(eigval, num, largest):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    ii = torch.argsort(eigval, descending=largest)
    return ii[:num]


def make_operator(A, B, M):
    A = aslinearoperator(A)
    B = aslinearoperator(B)
    M = aslinearoperator(M)
    return A, B, M


def check_system(A, X, B, M, Y):
    """check dimension and unify devices of inputs"""
    if X.ndim != 2:
        raise ValueError("expected rank-2 array for argument X")
    
    if A.device != X.device:
        raise ValueError(f"A is in {A.device}, while X in {X.device}")
    
    dev = A.device
    for m, name in zip([A, X, B, M, Y], ["A", "X", "B", "M", "Y"]):
        if m is not None:
            assert m.device == dev, f"{name}.device != {dev}"

    sizeY = Y.shape[1] if Y is not None else 0
    n, sizeX = X.shape
    assert (n - sizeY) >= (5 * sizeX)
    
    return dev


def check_args(maxiter, dynamicMaxiter, tol, dynamicTol, n, dynamic):

    maxiter = 20 if maxiter is None else maxiter
    dynamicMaxiter = maxiter if dynamicMaxiter is None else min(dynamicMaxiter, maxiter)
    tol = 10**(-19/2) * n if tol is None else tol
    dynamicTol = tol if dynamicTol is None else dynamicTol
    print(f"LOBPCG tolerance {tol, dynamicTol}")
    assert tol > 0 and dynamicTol > 0
    assert dynamic in [0,1,2]
    return maxiter, dynamicMaxiter, tol, dynamicTol
    

    #maxiter = 20 if maxiter is None else maxiter
    #dynamicMaxiter = maxiter if dynamicMaxiter is None else min(dynamicMaxiter, maxiter)
    #tol = (1e-15) ** (1 / 2) * n if tol is None else tol
    #dynamicTol = tol if dynamicTol is None else dynamicTol
    #assert tol > 0 and dynamicTol > 0
    #return maxiter, dynamicMaxiter, tol, dynamicTol

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
        warnings.filterwarnings(action='default')
        print(aux)

def initialize(A, B, X, dynamicLevel, sizeX, largest):
    # B-orthonormalize X.
    X, BX = _b_orthonormalize(B, X)
    # Compute the initial Ritz vectors: solve the eigenproblem.
    if X.is_complex():
        dtypes = [torch.complex64, torch.complex128]
    else:
        dtypes = [torch.float32, torch.float64]

    if dynamicLevel == 0:
        AX = A(X)
    elif dynamicLevel == 1:
        AX = A.matvec32(X.to(dtypes[0])).to(dtypes[1])
    elif dynamicLevel == 2:
        AX = A.matvec32(X)

    gramXAX = torch.matmul(X.T.conj(), AX)
    eigval, eigsubvec = subspaceDiagonalize(gramXAX)

    ii = _get_indx(eigval, sizeX, largest)
    eigval = eigval[ii]

    eigsubvec = eigsubvec[:, ii]
    X = torch.matmul(X, eigsubvec)
    AX = torch.matmul(AX, eigsubvec)
    if B is not None:
        BX = torch.matmul(BX, eigsubvec)
        aux = BX * eigval.unsqueeze(0)
    else:
        aux = X * eigval.unsqueeze(0)

    R = AX - aux
    aux = torch.sum(R.conj() * R, 0)
    if aux.is_complex(): 
        aux = aux.real
    residualNorms = torch.sqrt(aux)

    return X, BX, AX, eigval, residualNorms, R

def lobpcg(
    A,
    X,
    B=None,
    M=None,
    Y=None,
    tol=None,
    maxiter=None,
    dynamic=0,
    dynamicTol=None,
    dynamicMaxiter=None,
    dynamicInit=False,
    largest=True,
    verbosityLevel=0,
    retLambdaHistory=False,
    retResidualNormsHistory=False,
    verboseEvery=100,
    staticMask=True,
    numActiveLowerBound=0.0,
    dynamicTest=False,
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
    :param M:
        Preconditioner to `A`; by default ``M = Identity``.
        `M` should approximate the inverse of `A`.
        :math:`|M@A-I|_{F}` be sufficiently small
    :type Y: torch.Tensor or LinearOperator, optional
    :param Y:
        n-by-sizeY matrix of constraints (non-sparse), sizeY < n
        The iterations will be performed in the B-orthogonal complement
        of the column-space of Y. Y must be full rank.
    :type dynamic: bool, optional
    :param dynamic:
        If True, solve the eigenproblem with partially FP32. After sufficient
        convergence, convert to dtype=float64.
    :type dynamicTol: scalar, optional
    :param dynamicTol:
        Solver tolerance (conversion criterion to FP64).
        The default is ``tol=n*sqrt(eps)``.
    :type dynamicMaxiter: int, optional
    :param dynamicMaxiter:
        Maximum number of FP32 iterations.  The default is ``maxiter-1``.
    :type dynamicInit: bool, optional
    :param dynamicInit:
        If dynamicInit is ``True``, first calculate residual norm with float64.
        This is for cases initial guess is too accurate to float32 calculation.
        The default is ``True``.
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
    :type staticMask: bool, optional
    :param staticMask:
        This is masking policy. If True, once i_th vector's residual
        norms becomes smaller than tolerance, the vector masked permanantly.
        Else, once converged vectors could be un-masked after iteration. 
        Default is True.
    :type numActiveLowerBound: float, optional
    :param numActiveLowerBound:
        The lower bound of active vectors ratio over whole vectors. If set 1.0,
        no mask is applied which results in higher cost for each iteration. 
        Default is 0.0.
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

    Examples
    --------

    Solve ``A x = lambda x`` with constraints and preconditioning.

    >>> import numpy as np
    >>> from scipy.sparse import spdiags, issparse
    >>> from scipy.sparse.linalg import lobpcg, LinearOperator
    >>> n = 100
    >>> vals = np.arange(1, n + 1)
    >>> A = spdiags(vals, 0, n, n)
    >>> A.toarray()
    array([[  1.,   0.,   0., ...,   0.,   0.,   0.],
           [  0.,   2.,   0., ...,   0.,   0.,   0.],
           [  0.,   0.,   3., ...,   0.,   0.,   0.],
           ...,
           [  0.,   0.,   0., ...,  98.,   0.,   0.],
           [  0.,   0.,   0., ...,   0.,  99.,   0.],
           [  0.,   0.,   0., ...,   0.,   0., 100.]])

    Constraints:

    >>> Y = np.eye(n, 3)

    Initial guess for eigenvectors, should have linearly independent
    columns. Column dimension = number of requested eigenvalues.

    >>> X = np.random.rand(n, 3)

    Preconditioner in the inverse of A in this example:

    >>> invA = spdiags([1./vals], 0, n, n)

    The preconditiner must be defined by a function:

    >>> def precond( x ):
    ...     return invA @ x

    The argument x of the preconditioner function is a matrix inside `lobpcg`,
    thus the use of matrix-matrix product ``@``.

    The preconditioner function is passed to lobpcg as a `LinearOperator`:

    >>> M = LinearOperator(matvec=precond, matmat=precond,
    ...                    shape=(n, n), dtype=float)

    Let us now solve the eigenvalue problem for the matrix A:

    >>> eigenvalues, _ = lobpcg(A, X, Y=Y, M=M, largest=False)
    >>> eigenvalues
    array([4., 5., 6.])

    Note that the vectors passed in Y are the eigenvectors of the 3 smallest
    eigenvalues. The results returned are orthogonal to those.

    """
    global AR_time
    global init_time
    global AX_time
    global orthonormalize_time
    global norm_in_ortho
    global VBV_in_ortho
    global cho_inv_in_ortho
    global restore_in_ortho
    global sub_time
    AR_time = 0.0
    init_time = 0.0
    AX_time = 0.0
    orthonormalize_time = 0.0
    norm_in_ortho = 0.0
    VBV_in_ortho = 0.0
    cho_inv_in_ortho = 0.0
    restore_in_ortho = 0.0
    sub_time = 0.0
    
    st_total = time.time()
    ## check dimension
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
    M = aslinearoperator(M)
    check_verbosity(verbosityLevel, B, M, X, Y)
    
    ## set defaults values
    maxiter, dynamicMaxiter, tol, dynamicTol = check_args(
        maxiter, dynamicMaxiter, tol, dynamicTol, n, dynamic
    )
    dynamicLevel = dynamic
    
    residualTolerance = dynamicTol if dynamicLevel > 0 else tol
    if X.is_complex():
        dtypes = [torch.complex64, torch.complex128]
    else:
        dtypes = [torch.float32, torch.float64]
    
    # Apply constraints to X.
    if Y is not None:
        if B is not None:
            BY = B(Y)
        else:
            BY = Y
        # gramYBY is a dense array.
        gramYBY = torch.matmul(Y.T.conj(), BY)
        try:
            # gramYBY is a Cholesky factor from now on...
            gramYBY = torch.linalg.cholesky_ex(gramYBY, upper=False, check_errors=True)
        except RuntimeError as e:
            raise ValueError("cannot handle linearly dependent constraints") from e
        _applyConstraints(X, gramYBY, BY, Y)
    
    def get_mask(residualNorms, mask, maskTolerance, staticMask=True):
        ii = residualNorms > maskTolerance
        if ii.float().mean() >= numActiveLowerBound:
            if not staticMask:
                mask = ii
            else:
                mask = mask & ii
        else:
            # The number of active should be at least numActiveLowerBound
            if not staticMask:
                N = min(math.ceil(sizeX * numActiveLowerBound), sizeX)
                _, idx = torch.topk(residualNorms, N)
                mask = torch.zeros((sizeX,), dtype=bool, device=dev)
                mask[idx] = True
        return mask

    if not dynamicInit and dynamicLevel == 2:
        X = X.to(dtypes[0])
    initDynamicLevel = 0 if dynamicInit else dynamicLevel

    st = time.time()
    X, BX, AX, eigval, residualNorms, R = initialize(
        A, B, X, initDynamicLevel, sizeX, largest
    )
    init_time = time.time() - st

    mask = torch.ones((sizeX,), dtype=bool, device=dev)
    convergedResiduals = ~mask.clone()
    
    if dynamicTest and dynamicLevel > 0:
        _ = A(X.double()) - eigval.double()*X.double()
        _ = torch.sum(_.conj() * _, 0)
        test_residualNorms = torch.sqrt(_)
        residualNormsHistory_test = [test_residualNorms.to("cpu")]
        gramXAX_ = X.double().T@A(X.double())
        gramXBX_ = X.double().T@X.double() 
        test_eigval, test_eigvec = subspaceDiagonalize(gramXAX_, B=gramXBX_)
        eigvalHistory_test = [test_eigval.to("cpu")]
    residualNormsHistory = [residualNorms.to("cpu")]
    eigvalHistory = [eigval.to("cpu")]


    previousBlockSize = sizeX
    ident = torch.eye(sizeX, dtype=X.dtype, device=dev)
    ident0 = torch.eye(sizeX, dtype=X.dtype, device=dev)

    # Main iteration loop.
    
    # set during iteration, Initially None
    P = None
    AP = None
    BP = None

    iterationNumber = 0
    restart = True
    explicitGramFlag = False
    currentBlockSize = mask.sum()
    # Convergence Check
    if dynamicLevel == 0 or dynamicInit:
        if (residualNorms < tol).sum() == sizeX:  # 64tol converged - initial guess is already eigen vector
            print("Initial guess is already fine enough")
            print("eigenvalue:", eigval)
            print("residual norms:", residualNorms)
            print("tol:",tol)
            print()
            if retLambdaHistory:
                if retResidualNormsHistory:
                    if dynamicTest:
                        return eigval, X, eigvalHistory, eigvalHistory_test, residualNormsHistory, residualNormsHistory_test
                    else:
                        return eigval, X, eigvalHistory, residualNormsHistory
                else:
                    return eigval, X, eigvalHistory
            else:
                if retResidualNormsHistory:
                    if dynamicTest:
                        return eigval, X, residualNormsHistory, residualNormsHistory_test
                    else:
                        return eigval, X, residualNormsHistory
                else:
                    return eigval, X
    
    if dynamicLevel > 0:
        if (residualNorms < dynamicTol).sum() == sizeX:  # 32tol converged - skip fp32 iter
            if verbosityLevel > 0:
                print("Initial guess is too good to iter with float32")
                print("Skip float32 iteration")
                print("eigenvalue:", eigval)
                print("residual norms:", residualNorms)
                print(
                    "----------------------------------------------------------------"
                )
            total_float32_iter = 0
            if not dynamicInit and dynamicLevel == 2:
                X = X.to(dtypes[1])
                dynamicLevel = 0
                X, BX, AX, eigval, residualNorms, R = initialize(
                    A, B, X, dynamicLevel, sizeX, largest
                )
                
                residualNormsHistory.append(residualNorms.to("cpu"))
             
        else: # 32tol not converged - start from fp32 iter
            if dynamicInit and dynamicLevel == 2:
                X = X.to(dtypes[0])
                AX = AX.to(dtypes[0])
                eigval = eigval.to(dtypes[0])
                R = R.to(dtypes[0])
                if BX is not None:
                    BX = BX.to(dtypes[0]) 
                ident = torch.eye(sizeX, dtype=X.dtype, device=dev)
                ident0 = torch.eye(sizeX, dtype=X.dtype, device=dev)
                
    cumBlockSize = 0        
    while iterationNumber < maxiter:
        if currentBlockSize != previousBlockSize:
            previousBlockSize = currentBlockSize
            ident = torch.eye(currentBlockSize, dtype=A.dtype, device=dev)
        cumBlockSize += currentBlockSize
        if verbosityLevel > 0 and iterationNumber % verboseEvery == 0:
            print(
                f"iteration {iterationNumber}\n"
                f"current block size: {currentBlockSize}\n"
                f"eigenvalue: {eigval}\n"
                f"residual norms: {residualNorms}\n"
                f"tol : {residualTolerance}\n"
            )
            

        activeR = _as2d(R[:, mask])

        if iterationNumber > 0:# and not restart:
            activeP = _as2d(P[:, mask])
            activeAP = _as2d(AP[:, mask])
            activeBP = _as2d(BP[:, mask]) if B is not None else None

        if M is not None:
            # Apply preconditioner T to the active residuals.
            if dynamicLevel == 0:
                activeR = M(activeR)
            elif dynamicLevel == 1:
                activeR = M.matvec32(activeR.to(dtypes[0])).to(dtypes[1])
            elif dynamicLevel == 2:
                activeR = M.matvec32(activeR)


        # Apply constraints to the preconditioned residuals.
        if Y is not None:
            _applyConstraints(activeR, gramYBY, BY, Y)

        # B-orthogonalize the preconditioned residuals to X.
        if B is not None:
            activeR = activeR - torch.matmul(X, torch.matmul(BX.T.conj(), activeR))
        else:
            activeR = activeR - torch.matmul(X, torch.matmul(X.T.conj(), activeR))

        # B-orthonormalize the preconditioned residuals.
        aux = _b_orthonormalize(B, activeR)
        activeR, activeBR = aux

        if activeR is None:
            warnings.warn(
                f"Failed at iteration {iterationNumber} with accuracies "
                f"{residualNorms}\n not reaching the requested "
                f"tolerance {residualTolerance}.",
                UserWarning,
                stacklevel=2,
            )
            if dynamicLevel > 0:
                print(f"Elapsed time[diagonalize-DYN]: {time.time() - st} secs (termination type A)")
                print(f"Cumulative Blocks (DYN): {cumBlockSize}")
                staticMask = True
                cumBlockSize = 0
                total_float32_iter = iterationNumber + 1
                maxiter -= (iterationNumber + 1)
                iterationNumber = 0
                dynamicLevel = 0
                residualTolerance = tol
                X = X.to(dtypes[1])
                if maxiter == 0:
                    BX = BX.to(dtypes[1]) 
                    AX = AX.to(dtypes[1])
                    eigval = eigval.to(dtypes[1])
                    break
                #if dynamicTest: break
                
                if verbosityLevel > 0:
                    print(
                        f"\ntotal float32 iteration (type A): {total_float32_iter}\n"
                        f"eigenvalue: {eigval}\n"
                        f"residual norms: {residualNorms}\n"
                        f"------------------------------"
                    )

                X, BX, AX, eigval, residualNorms, R = initialize(
                    A, B, X, dynamicLevel, sizeX, largest
                )
                residualNormsHistory.append(residualNorms.to("cpu"))
                mask = torch.ones((sizeX,), dtype=bool, device=dev)
                convergedResiduals = ~mask.clone()
                P, AP, BP = None, None, None
                restart, explicitGramFlag = True, False
                currentBlockSize = mask.sum()
                continue
            break

        st = time.time()
        if dynamicLevel == 0:
            activeAR = A(activeR)
        elif dynamicLevel == 1:
            activeAR = A.matvec32(activeR.to(dtypes[0])).to(dtypes[1])
        elif dynamicLevel == 2:
            activeAR = A.matvec32(activeR)

        AR_time += time.time() - st

        if iterationNumber > 0:# and not restart:
            aux = _b_orthonormalize(
                B, activeP, BV=activeBP, retInvR=True
            )
            activeP, activeBP, invR, normal = aux
            # Function _b_orthonormalize returns None if Cholesky fails
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
                if dynamicLevel > 0:
                    print(f"Elapsed time[diagonalize-DYN]: {time.time() - st} secs (termination type A)")
                    print(f"Cumulative Blocks (DYN): {cumBlockSize}")
                    staticMask = True
                    cumBlockSize = 0
                    total_float32_iter = iterationNumber + 1
                    maxiter -= iterationNumber + 1
                    iterationNumber = 0
                    dynamicLevel = 0
                    residualTolerance = tol
                    X = X.to(dtypes[1])
                    if maxiter == 0:
                        BX = BX.to(dtypes[1]) 
                        AX = AX.to(dtypes[1])
                        eigval = eigval.to(dtypes[1])
                        break
                    #if dynamicTest: break
                    
                    if verbosityLevel > 0:
                        print(
                            f"\ntotal float32 iteration (type A): {total_float32_iter}\n"
                            f"eigenvalue: {eigval}\n"
                            f"residual norms: {residualNorms}\n"
                            f"------------------------------"
                        )

                    X, BX, AX, eigval, residualNorms, R = initialize(
                        A, B, X, dynamicLevel, sizeX, largest
                    )
                    residualNormsHistory.append(residualNorms.to("cpu"))
                    mask = torch.ones((sizeX,), dtype=bool, device=dev)
                    convergedResiduals = ~mask.clone()
                    P, AP, BP = None, None, None
                    restart, explicitGramFlag = True, False
                    currentBlockSize = mask.sum()
                    continue


        # Perform the Rayleigh Ritz Procedure:
        # Compute symmetric Gram matrices:

        if dynamicLevel > 0:
            if activeR.dtype == dtypes[0]: # float32
                myeps = 10
            else: # float64
                myeps = 1e-6
        else:
            if activeR.dtype == dtypes[0]: # float32
                myeps = 10
            else: # float64
                myeps = 1e-8

        if not explicitGramFlag and residualNorms.abs().max()<myeps:
            print("Changed To Explicit Gram Calculation")
        if residualNorms.abs().max() > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else:
            # Once explicitGramFlag, forever explicitGramFlag.
            explicitGramFlag = True

        # Shared memory assingments to simplify the code
        if B is None:
            BX = X
            activeBR = activeR
            if not restart:
                activeBP = activeP

        st_sub = time.time()
        # Common submatrices:
        if dynamicLevel == 1:
            X32 = X.to(dtypes[0])
            AX32 = AX.to(dtypes[0])
            activeAR32 = activeAR.to(dtypes[0])
            activeR32 = activeR.to(dtypes[0])

            gramXAX = torch.matmul(X32.T.conj(), AX32)
            gramXAR = torch.matmul(X32.T.conj(), activeAR32)
            gramRAR = torch.matmul(activeR32.T.conj(), activeAR32)

        else:
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
            gramXBR = torch.zeros((sizeX, currentBlockSize), dtype=X.dtype, device=dev)

        def _handle_gramA_gramB_verbosity(gramA, gramB):
            if verbosityLevel > 1 and iterationNumber % verboseEvery == 0:
                _report_nonhermitian(gramA, "gramA")
                _report_nonhermitian(gramB, "gramB")
            if verbosityLevel > 10 and iterationNumber % verboseEvery == 0:
                # Note: not documented, but leave it in here for now
                torch.save(gramA, "gramA.pt")
                torch.save(gramB, "gramB.pt")

        if not restart:
            if dynamicLevel == 1:
                activeAP32 = activeAP.to(dtypes[0])
                activeP32 = activeP.to(dtypes[0])
                
                gramXAP = torch.matmul(X32.T.conj(), activeAP32)
                gramRAP = torch.matmul(activeR32.T.conj(), activeAP32)
                gramPAP = torch.matmul(activeP32.T.conj(), activeAP32)
            else:
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

            if dynamicLevel == 1:
                gramA = gramA.to(dtypes[1])

            _handle_gramA_gramB_verbosity(gramA, gramB)

            try:
                eigval, eigsubvec = subspaceDiagonalize(gramA, B=gramB)
            except Exception as e:
                warnings.warn(
                    f"Failed subspace diagonalization, at {iterationNumber} iter."
                    f"\nerror message : \n{e}",
                    UserWarning,
                    stacklevel=1,
                )
                # try again after dropping the direction vectors P from RR
                if dynamicLevel > 0:
                    explicitGramFlag=True
                restart = True

        if restart:
            gramA = bmat([[gramXAX, gramXAR], [gramXAR.T.conj(), gramRAR]])
            gramB = bmat([[gramXBX, gramXBR], [gramXBR.T.conj(), gramRBR]])
            _handle_gramA_gramB_verbosity(gramA, gramB)

            if dynamicLevel == 1:
                gramA = gramA.to(dtypes[1])
            try:
                eigval, eigsubvec = subspaceDiagonalize(gramA, B=gramB)
            except Exception as e:
                if dynamicLevel > 0:
                    print(f"Elapsed time[diagonalize-DYN]: {time.time() - st} secs (termination type B)")
                    print(f"Cumulative Blocks (DYN): {cumBlockSize}")
                    staticMask = True
                    cumBlockSize = 0
                    st = time.time()
                    warnings.warn(
                        f"Failed subspace diagonalization, at {iterationNumber} iter.\n"
                        f"error message : \n{e}",
                        UserWarning,
                        stacklevel=1,
                    )
                    total_float32_iter = iterationNumber + 1
                    maxiter -= iterationNumber + 1
                    iterationNumber = 0
                    dynamicLevel = 0
                    residualTolerance = tol
                    X = X.to(dtypes[1])
                    if maxiter == 0:
                        X = X.to(dtypes[1])
                        BX = BX.to(dtypes[1]) 
                        AX = AX.to(dtypes[1])
                        eigval = eigval.to(dtypes[1])
                        break
                    #if dynamicTest: break
                    if verbosityLevel > 0:
                        print(
                            f"\ntotal float32 iteration (type B): {total_float32_iter}\n"
                            f"eigenvalue: {eigval}\n"
                            f"residual norms: {residualNorms}\n"
                            f"------------------------------"
                        )

                    X, BX, AX, eigval, residualNorms, R = initialize(
                        A, B, X, dynamicLevel, sizeX, largest
                    )
                    residualNormsHistory.append(residualNorms.to("cpu"))
                    mask = torch.ones((sizeX,), dtype=bool, device=dev)
                    convergedResiduals = ~mask.clone()
                    P, AP, BP = None, None, None
                    restart, explicitGramFlag = True, False
                    currentBlockSize = mask.sum()
                    continue
                raise ValueError("eigh has failed in lobpcg iterations") from e

        ii = _get_indx(eigval, sizeX, largest)
        eigval = eigval[ii]
        eigsubvec = eigsubvec[:, ii]
        eigvalHistory.append(eigval.to("cpu"))

        # Compute Ritz vectors.
        if B is not None:
            if not restart:
                eigsubvecX = eigsubvec[:sizeX]
                eigsubvecR = eigsubvec[sizeX : sizeX + currentBlockSize]
                eigsubvecP = eigsubvec[sizeX + currentBlockSize :]

                pp = torch.matmul(activeR, eigsubvecR)
                pp += torch.matmul(activeP, eigsubvecP)

                app = torch.matmul(activeAR, eigsubvecR)
                app += torch.matmul(activeAP, eigsubvecP)

                bpp = torch.matmul(activeBR, eigsubvecR)
                bpp += torch.matmul(activeBP, eigsubvecP)
            else:
                eigsubvecX = eigsubvec[:sizeX]
                eigsubvecR = eigsubvec[sizeX:]

                pp = torch.matmul(activeR, eigsubvecR)
                app = torch.matmul(activeAR, eigsubvecR)
                bpp = torch.matmul(activeBR, eigsubvecR)

            if verbosityLevel > 10 and iterationNumber % verboseEvery == 0:
                print(pp)
                print(app)
                print(bpp)

            X = torch.matmul(X, eigsubvecX) + pp
            AX = torch.matmul(AX, eigsubvecX) + app
            BX = torch.matmul(BX, eigsubvecX) + bpp

            P, AP, BP = pp, app, bpp

        else:
            if not restart:
                eigsubvecX = eigsubvec[:sizeX]
                eigsubvecR = eigsubvec[sizeX : sizeX + currentBlockSize]
                eigsubvecP = eigsubvec[sizeX + currentBlockSize :]

                pp = torch.matmul(activeR, eigsubvecR)
                pp += torch.matmul(activeP, eigsubvecP)

                app = torch.matmul(activeAR, eigsubvecR)
                app += torch.matmul(activeAP, eigsubvecP)
            else:
                eigsubvecX = eigsubvec[:sizeX]
                eigsubvecR = eigsubvec[sizeX:]

                pp = torch.matmul(activeR, eigsubvecR)
                app = torch.matmul(activeAR, eigsubvecR)

            if verbosityLevel > 10 and iterationNumber % verboseEvery == 0:
                print(pp)
                print(app)

            X = torch.matmul(X, eigsubvecX) + pp
            AX = torch.matmul(AX, eigsubvecX) + app

            P, AP = pp, app

        if B is not None:
            aux = BX * eigval.unsqueeze(0)
        else:
            aux = X * eigval.unsqueeze(0)
        sub_time += time.time() - st_sub

        # Calc residual R, AX - _lambda@X
        R = AX - aux
        aux = torch.sum(R.conj() * R, 0)
        if aux.is_complex(): 
            aux = aux.real
        residualNorms = torch.sqrt(aux)
        
        if dynamicTest and dynamicLevel > 0:
            _ = A(X.double()) - eigval.double()*X.double()
            _ = torch.sum(_.conj() * _, 0)
            test_residualNorms = torch.sqrt(_)
            residualNormsHistory_test.append(test_residualNorms.to("cpu"))
            gramXAX_ = X.double().T@A(X.double())
            gramXBX_ = X.double().T@X.double() 
            test_eigval, test_eigvec = subspaceDiagonalize(gramXAX_, B=gramXBX_)
            eigvalHistory_test.append(test_eigval.to("cpu"))
        residualNormsHistory.append(residualNorms.to("cpu"))


        # Create Mask
        mask = get_mask(residualNorms, mask, residualTolerance, staticMask=staticMask)
        currentBlockSize = mask.sum()
        if verbosityLevel > 1 and iterationNumber % verboseEvery == 0:
            print(f"activeMask : {mask}")

        # Convergence Check
        ii = residualNorms < residualTolerance
        if staticMask:
            convergedResiduals = convergedResiduals | ii
        else:
            convergedResiduals = ii

        if dynamicLevel > 0 and (
            iterationNumber >= min(dynamicMaxiter - 1, maxiter - 1)
            or convergedResiduals.sum() == sizeX
        ):
            print(f"Elapsed time[diagonalize-DYN]: {time.time() - st} secs (termination type C)")
            print(f"Cumulative Blocks (DYN): {cumBlockSize}")
            staticMask = True
            cumBlockSize = 0
            st = time.time()
            total_float32_iter = iterationNumber + 1
            maxiter -= iterationNumber + 1
            iterationNumber = 0
            dynamicLevel = 0
            residualTolerance = tol
            X = X.to(dtypes[1])
            if maxiter == 0:
                BX = BX.to(dtypes[1]) 
                AX = AX.to(dtypes[1])
                eigval = eigval.to(dtypes[1])
                break
            #if dynamicTest: break
            
            if verbosityLevel > 0:
                print(
                    f"\ntotal float32 iteration (type C): {total_float32_iter}\n"
                    f"eigenvalue: {eigval}\n"
                    f"residual norms: {residualNorms}\n"
                    f"------------------------------"
                )

            X, BX, AX, eigval, residualNorms, R = initialize(
                A, B, X, dynamicLevel, sizeX, largest
            )
            residualNormsHistory.append(residualNorms.to("cpu"))
            mask = torch.ones((sizeX,), dtype=bool, device=dev)
            convergedResiduals = ~mask.clone()
            P, AP, BP = None, None, None
            restart, explicitGramFlag = True, False
            currentBlockSize = mask.sum()
            continue 

        iterationNumber += 1
        if not dynamicLevel > 0 and convergedResiduals.sum() == sizeX: 
            break
        

    if B is not None:
        aux = BX * eigval.unsqueeze(0)
    else:
        aux = X * eigval.unsqueeze(0)

    R = AX - aux

    aux = torch.sum(R.conj() * R, 0)
    if aux.dtype in [torch.complex128, torch.complex64]:
        aux = aux.real
    residualNorms = torch.sqrt(aux)

    # Future work: Need to add Postprocessing here:
    # Making sure eigenvectors "exactly" satisfy the blockVectorY constrains?
    # Making sure eigenvecotrs are "exactly" othonormalized by final "exact" RR
    # Computing the actual true residuals

    print(f"================================================================")
    # print(f"* make_op_time: {make_op_time} sec")
    print(f"* init_time: {init_time} sec")
    print(f"    AX_time: {AX_time} sec")
    print(f"* AR_time: {AR_time} sec")
    print(f"* orthonormalize_time: {orthonormalize_time} sec")
    print(f"    norm in ortho: {norm_in_ortho} sec")
    print(f"    VBV in ortho: {VBV_in_ortho} sec")
    print(f"    cho_inv in ortho: {cho_inv_in_ortho} sec")
    print(f"    restore in ortho: {restore_in_ortho} sec")
    print(f"* sub_time: {sub_time} sec")
    print(f"* TOTAL TIME: {time.time() - st_total} sec")
    print(f"================================================================")

    if verbosityLevel > 0:
        print(f"Elapsed time[diagonalize-FP64]: {time.time() - st} secs")
        print(f"Cumulative Blocks (FP64): {cumBlockSize}")
        if dynamic > 0:
            print(f"\nfinal iteration : {total_float32_iter} + {iterationNumber}")
        else:
            print("\nfinal iteration :", iterationNumber)
        print("final eigenvalue:", eigval)
        print("final residual norms:", residualNorms)
    
    if retLambdaHistory:
        if retResidualNormsHistory:
            if dynamicTest:
                return eigval, X, eigvalHistory, eigvalHistory_test, residualNormsHistory, residualNormsHistory_test
            else:
                return eigval, X, eigvalHistory, residualNormsHistory
        else:
            return eigval, X, eigvalHistory
    else:
        if retResidualNormsHistory:
            if dynamicTest:
                return eigval, X, residualNormsHistory, residualNormsHistory_test
            else:
                return eigval, X, residualNormsHistory
        else:
            return eigval, X



if __name__ == "__main__":
    import numpy as np
    from scipy.sparse.linalg import lobpcg as scipy_lobpcg

    matrix_type = ["hermitian", "symmetric"][0]
    A = np.random.randn(100, 100)
    X = np.random.randn(100, 10)
    if matrix_type == "hermitian":
        A = (A + 1j * A) / 2
        X = (X + 1j * X) / 2
    A = (A + A.T.conj()) / 2
    options = {
        "A": A,
        "X": X,
        "tol": None,
        "maxiter": None,
        "largest": False,
    }
    val, vec = scipy_lobpcg(**options)
    print(f"eigval (scipy):\n{val}")
    print("=" * 80)

    A = torch.from_numpy(A)
    X = torch.from_numpy(X)
    options = {
        "A": A,
        "X": X,
        "tol": None,
        "maxiter": 20,
        "largest": False,
    }
    val, vec = lobpcg(**options)
    print(f"eigval (torch):\n{val}")
