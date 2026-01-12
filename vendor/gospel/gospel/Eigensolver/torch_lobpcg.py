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

from torch.linalg import inv
from torch.linalg import eigh as eigh_

from LinearOperator import LinearOperator, aslinearoperator

# from scipy.sparse.linalg import lobpcg

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
        _lambda, eighv = eigh_(_A)
        eighv = Linv.conj().T @ eighv
        return _lambda, eighv
    else:
        _lambda, eighv = eigh_(A)
        return _lambda, eighv


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
    from torch.linalg import norm

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


def _applyConstraints(blockVectorV, factYBY, blockVectorBY, blockVectorY):
    """Changes blockVectorV in place."""
    YBV = torch.matmul(blockVectorBY.T.conj(), blockVectorV)
    tmp = torch.linalg.cholesky_solve(YBV, factYBY[0], upper=False)
    blockVectorV -= torch.matmul(blockVectorY, tmp)


def _b_orthonormalize(B, blockVectorV, blockVectorBV=None, retInvR=False, tmp=False):
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
    normalization = blockVectorV.max(axis=0)[0] + torch.finfo(blockVectorV.dtype).eps
    blockVectorV = blockVectorV / normalization
    if blockVectorBV is None:
        if B is not None:
            blockVectorBV = B(blockVectorV)
        else:
            blockVectorBV = blockVectorV  # Shared data!!!
    else:
        blockVectorBV = blockVectorBV / normalization
    VBV = torch.matmul(blockVectorV.T.conj(), blockVectorBV)
    VBV = (VBV + VBV.T.conj()) / 2
    if tmp:
        w, v = eigh(VBV)
        print(f"Minium eigen values : {min(w).item():0.11f}")
    try:
        # decompose V.T@B@V into L@L.T. U = L.T
        LT = torch.linalg.cholesky(VBV, upper=True)
        LTinv = inv(LT,)
        # V' = V@Uinv -> V'BV'=Uinv.T.conj()@VBV@Uinv=I
        blockVectorV = torch.matmul(blockVectorV, LTinv)
        # blockVectorV = (cho_solve((VBV.T, True), blockVectorV.T)).T
        if B is not None:
            blockVectorBV = torch.matmul(blockVectorBV, LTinv)
            # blockVectorBV = (cho_solve((VBV.T, True), blockVectorBV.T)).T
        else:
            blockVectorBV = None
    # except LinAlgError:
    except:
        # raise ValueError("Cholesky has failed")
        blockVectorV = None
        blockVectorBV = None
        LTinv = None

    if retInvR:
        return blockVectorV, blockVectorBV, LTinv, normalization
    else:
        return blockVectorV, blockVectorBV


def _get_indx(_lambda, num, largest):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    ii = torch.argsort(_lambda, descending=largest)
    return ii[:num]


def make_operator(A, B, M, dynamic):
    if dynamic:
        A = aslinearoperator(A, A_32=A.float())
        B = aslinearoperator(B)
        M = aslinearoperator(M, A_32=M.float() if M is not None else None)
    else:
        A = aslinearoperator(A)
        B = aslinearoperator(B)
        M = aslinearoperator(M)
    return A, B, M


def check(A, X, B, M, Y):
    if X.ndim != 2:
        raise ValueError("expected rank-2 array for argument X")
    if A.device != X.device:
        raise ValueError(f"A is in {A.device}, while X in {X.device}")
    dev = A.device
    out = [m.to(dev) if m is not None else None for m in [A, X, B, M, Y]]

    sizeY = Y.shape[1] if Y is not None else 0
    n, sizeX = X.shape
    assert (n - sizeY) >= (5 * sizeX)

    return out


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
        print(aux)


def lobpcg(
    A,
    X,
    B=None,
    M=None,
    Y=None,
    tol=None,
    maxiter=None,
    dynamicTol=None,
    dynamicMaxiter=None,
    dynamic=False,
    largest=True,
    verbosityLevel=0,
    retLambdaHistory=False,
    retResidualNormsHistory=False,
    test=False,
    verboseEvery=100,
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
    A, X, B, M, Y = check(A, X, B, M, Y)
    dev_ = A.device
    A, B, M = make_operator(A, B, M, dynamic)
    check_verbosity(verbosityLevel, B, M, X, Y)

    # Block size.
    blockVectorX = X
    blockVectorY = Y
    sizeY = blockVectorY.shape[1] if blockVectorY is not None else 0
    n, sizeX = blockVectorX.shape

    # Criteria
    changed = not dynamic
    maxiter = 20 if maxiter is None else maxiter
    dynamicMaxiter = maxiter - 1 if dynamicMaxiter is None else dynamicMaxiter
    tol = (1e-15) ** (1 / 2) * n if tol is None else tol
    dynamicTol = (1e-15) ** (1 / 4) * n if dynamicTol is None else dynamicTol
    assert tol > 0 and dynamicTol > 0
    residualTolerance = dynamicTol if dynamic else tol

    # Apply constraints to X.
    if blockVectorY is not None:
        if B is not None:
            blockVectorBY = B(blockVectorY)
        else:
            blockVectorBY = blockVectorY
        # gramYBY is a dense array.
        gramYBY = torch.matmul(blockVectorY.T.conj(), blockVectorBY)
        try:
            # gramYBY is a Cholesky factor from now on...
            gramYBY = torch.linalg.cholesky_ex(gramYBY, upper=False, check_errors=True)
        except RuntimeError as e:
            raise ValueError("cannot handle linearly dependent constraints") from e
        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)

    def change_dtype(operators, dtype):
        for op in operators:
            op.dtype = dtype
        if verbosityLevel >= 0:
            print(f"Change dtype to {dtype}")

    def initialize(
        A, B, blockVectorX,
    ):
        dev_ = blockVectorX.device
        # B-orthonormalize X.
        blockVectorX, blockVectorBX = _b_orthonormalize(B, blockVectorX)

        # Compute the initial Ritz vectors: solve the eigenproblem.
        blockVectorAX = A(blockVectorX)
        gramXAX = torch.matmul(blockVectorX.T.conj(), blockVectorAX)

        _lambda, eigBlockVector = eigh(gramXAX)
        ii = _get_indx(_lambda, sizeX, largest)
        _lambda = _lambda[ii]

        eigBlockVector = eigBlockVector[:, ii]
        blockVectorX = torch.matmul(blockVectorX, eigBlockVector)
        blockVectorAX = torch.matmul(blockVectorAX, eigBlockVector)
        if B is not None:
            blockVectorBX = torch.matmul(blockVectorBX, eigBlockVector)

        # Active index set.
        activeMask = torch.ones((sizeX,), dtype=bool, device=dev_)

        return blockVectorX, blockVectorBX, blockVectorAX, _lambda, activeMask

    blockVectorX, blockVectorBX, blockVectorAX, _lambda, activeMask = initialize(
        A, B, blockVectorX
    )

    lambdaHistory = [_lambda]
    residualNormsHistory = []

    previousBlockSize = sizeX
    ident = torch.eye(sizeX, dtype=A.dtype, device=dev_)
    ident0 = torch.eye(sizeX, dtype=A.dtype, device=dev_)

    # Main iteration loop.
    # set during iteration, Initially None
    blockVectorP = None
    blockVectorAP = None
    blockVectorBP = None

    iterationNumber = -1
    restart = True
    explicitGramFlag = False

    while iterationNumber < maxiter:
        iterationNumber += 1
        if verbosityLevel >= 0 and iterationNumber % verboseEvery == 0:
            print("\niteration %d" % iterationNumber)
            # print(f"explicitGramFlag : {explicitGramFlag}")
            # print(f"restart : {restart}")
            if verbosityLevel > 2:
                print("XBX : \n", blockVectorX.T @ blockVectorX)
                # print("XBX : \n", blockVectorX.T @ blockVectorBX)
                print(
                    "PBP : \n",
                    blockVectorP.T @ blockVectorP if blockVectorP is not None else None,
                )

        if B is not None:
            aux = blockVectorBX * _lambda.unsqueeze(0)
        else:
            aux = blockVectorX * _lambda.unsqueeze(0)

        # Calc residual, AX - _lambda@X
        blockVectorR = blockVectorAX - aux

        aux = torch.sum(blockVectorR.conj() * blockVectorR, 0)
        residualNorms = torch.sqrt(aux)

        residualNormsHistory.append(residualNorms)

        ii = torch.where(residualNorms > residualTolerance, True, False)
        activeMask = activeMask & ii
        if verbosityLevel > 2:
            print(f"activeMask : {activeMask}")

        currentBlockSize = activeMask.sum()

        # if dynamic, initialization and first loop are done with FP64.
        # This is for cases where the initial prediction is sufficiently accurate.
        # if residual norm calculated with float64 are sufficiently small (dynamicTol),
        # then pass float32 iterations.
        dynamic32 = dynamic and not changed

        if currentBlockSize == 0:
            if dynamic32:
                # case1. first float64 iter with good initial guess
                # case2. dynamic float32 iteration converged
                # in both case, convert to FP64
                changed = True
                residualTolerance = tol
                _maxiter = maxiter
                operators = [A, M] if M is not None else [A]
                aux = A.dtype == torch.float64
                change_dtype(operators, torch.float64)
                (
                    blockVectorX,
                    blockVectorBX,
                    blockVectorAX,
                    _lambda,
                    activeMask,
                ) = initialize(A, B, blockVectorX)
                blockVectorP, blockVectorAP, blockVectorBP = None, None, None
                changed, restart, explicitGramFlag = True, True, False
                if verbosityLevel >= 0:
                    if aux:
                        # case 1
                        print(iterationNumber)
                        print("Initial guess is too good to iter with float32")
                        print("Skip float32 iteration")
                        print("eigenvalue:", _lambda)
                        print("residual norms:", residualNorms)
                        print(
                            "----------------------------------------------------------------"
                        )
                    else:
                        # case 2
                        print(f"\ntotal float32 iteration : {iterationNumber}")
                        print("eigenvalue:", _lambda)
                        print("residual norms:", residualNorms)
                        print(
                            "----------------------------------------------------------------"
                        )
                iterationNumber = -1
                if test:
                    break
                continue
            else:
                break

        if iterationNumber in {0, dynamicMaxiter} and dynamic32:
            # if iterationNumber == 0: dtype float64 -> float32
            # if iterationNumber == maxiter: dtype float32 -> float64
            changed = False
            operators = [A, M] if M is not None else [A]
            dtype = torch.float32 if iterationNumber == 0 else torch.float64
            change_dtype(operators, dtype)
            if verbosityLevel >= 0 and iterationNumber != 0:
                print(f"\ntotal float32 iteration : {iterationNumber}")
                print("eigenvalue:", _lambda)
                print("residual norms:", residualNorms)

        if dynamic32:
            # during dynamic float32 iter, masking nothing.
            activeMask = torch.ones((sizeX,), dtype=bool, device=dev_)
            currentBlockSize = sizeX

        if currentBlockSize != previousBlockSize:
            previousBlockSize = currentBlockSize
            ident = torch.eye(currentBlockSize, dtype=A.dtype, device=dev_)

        if verbosityLevel >= 0 and iterationNumber % verboseEvery == 0:
            print("current block size: ", currentBlockSize)
            print("eigenvalue: ", _lambda)
            print("residual norms: ", residualNorms)
            print(f"tol : {residualTolerance}")

        activeBlockVectorR = _as2d(blockVectorR[:, activeMask])

        if iterationNumber > 0:
            activeBlockVectorP = _as2d(blockVectorP[:, activeMask])
            activeBlockVectorAP = _as2d(blockVectorAP[:, activeMask])
            if B is not None:
                activeBlockVectorBP = _as2d(blockVectorBP[:, activeMask])

        if M is not None:
            # Apply preconditioner T to the active residuals.
            activeBlockVectorR = M(activeBlockVectorR)

        # Apply constraints to the preconditioned residuals.
        if blockVectorY is not None:
            _applyConstraints(activeBlockVectorR, gramYBY, blockVectorBY, blockVectorY)

        # B-orthogonalize the preconditioned residuals to X.
        if B is not None:
            activeBlockVectorR = activeBlockVectorR - torch.matmul(
                blockVectorX, torch.matmul(blockVectorBX.T.conj(), activeBlockVectorR)
            )
        else:
            activeBlockVectorR = activeBlockVectorR - torch.matmul(
                blockVectorX, torch.matmul(blockVectorX.T.conj(), activeBlockVectorR)
            )

        # B-orthonormalize the preconditioned residuals.
        aux = _b_orthonormalize(B, activeBlockVectorR)
        activeBlockVectorR, activeBlockVectorBR = aux

        if activeBlockVectorR is None:
            warnings.warn(
                f"Failed at iteration {iterationNumber} with accuracies "
                f"{residualNorms}\n not reaching the requested "
                f"tolerance {residualTolerance}.",
                UserWarning,
                stacklevel=2,
            )
            if dynamic32:
                changed = True
                residualTolerance = tol
                _maxiter = maxiter
                operators = [A, M] if M is not None else [A]
                aux = A.dtype == torch.float64
                change_dtype(operators, torch.float64)
                (
                    blockVectorX,
                    blockVectorBX,
                    blockVectorAX,
                    _lambda,
                    activeMask,
                ) = initialize(A, B, blockVectorX)
                blockVectorP, blockVectorAP, blockVectorBP = None, None, None
                changed, restart, explicitGramFlag = True, True, False
                iterationNumber = -1
                continue
            else:
                break

        activeBlockVectorAR = A(activeBlockVectorR)

        if iterationNumber > 0:
            if B is not None:
                aux = _b_orthonormalize(
                    B, activeBlockVectorP, activeBlockVectorBP, retInvR=True
                )
                activeBlockVectorP, activeBlockVectorBP, invR, normal = aux

            else:
                aux = _b_orthonormalize(B, activeBlockVectorP, retInvR=True, tmp=True)
                activeBlockVectorP, _, invR, normal = aux
            # Function _b_orthonormalize returns None if Cholesky fails
            if activeBlockVectorP is not None:
                activeBlockVectorAP = activeBlockVectorAP / normal
                activeBlockVectorAP = torch.matmul(activeBlockVectorAP, invR)
                restart = False
            else:
                if verbosityLevel >= 0:
                    warnings.warn(
                        f"Failed orthonormalization at iteration " f"{iterationNumber}",
                        UserWarning,
                        stacklevel=2,
                    )
                if dynamic32:
                    changed = True
                    residualTolerance = tol
                    _maxiter = maxiter
                    operators = [A, M] if M is not None else [A]
                    aux = A.dtype == torch.float64
                    change_dtype(operators, torch.float64)
                    (
                        blockVectorX,
                        blockVectorBX,
                        blockVectorAX,
                        _lambda,
                        activeMask,
                    ) = initialize(A, B, blockVectorX)
                    blockVectorP, blockVectorAP, blockVectorBP = None, None, None
                    changed, restart, explicitGramFlag = True, True, False
                    iterationNumber = -1
                    continue

                else:
                    restart = True

        # Perform the Rayleigh Ritz Procedure:
        # Compute symmetric Gram matrices:

        if activeBlockVectorAR.dtype == "float32":
            myeps = 1
        elif activeBlockVectorR.dtype == "float32":
            myeps = 1e-4
        else:
            myeps = 1e-8

        if residualNorms.max() > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else:
            # Once explicitGramFlag, forever explicitGramFlag.
            explicitGramFlag = True

        # Shared memory assingments to simplify the code
        if B is None:
            blockVectorBX = blockVectorX
            activeBlockVectorBR = activeBlockVectorR
            if not restart:
                activeBlockVectorBP = activeBlockVectorP

        # Common submatrices:
        gramXAR = torch.matmul(blockVectorX.T.conj(), activeBlockVectorAR)
        gramRAR = torch.matmul(activeBlockVectorR.T.conj(), activeBlockVectorAR)

        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T.conj()) / 2
            gramXAX = torch.matmul(blockVectorX.T.conj(), blockVectorAX)
            gramXAX = (gramXAX + gramXAX.T.conj()) / 2
            gramXBX = torch.matmul(blockVectorX.T.conj(), blockVectorBX)
            gramRBR = torch.matmul(activeBlockVectorR.T.conj(), activeBlockVectorBR)
            gramXBR = torch.matmul(blockVectorX.T.conj(), activeBlockVectorBR)
        else:
            gramXAX = torch.diag(_lambda)
            gramXBX = ident0
            gramRBR = ident
            gramXBR = torch.zeros((sizeX, currentBlockSize), dtype=A.dtype, device=dev_)

        def _handle_gramA_gramB_verbosity(gramA, gramB):
            if verbosityLevel >= 1 and iterationNumber % verboseEvery == 0:
                _report_nonhermitian(gramA, "gramA")
                _report_nonhermitian(gramB, "gramB")
            if verbosityLevel > 10 and iterationNumber % verboseEvery == 0:
                # Note: not documented, but leave it in here for now
                torch.save(gramA, "gramA.pt")
                torch.save(gramB, "gramB.pt")

        if not restart:
            gramXAP = torch.matmul(blockVectorX.T.conj(), activeBlockVectorAP)
            gramRAP = torch.matmul(activeBlockVectorR.T.conj(), activeBlockVectorAP)
            gramPAP = torch.matmul(activeBlockVectorP.T.conj(), activeBlockVectorAP)
            gramXBP = torch.matmul(blockVectorX.T.conj(), activeBlockVectorBP)
            gramRBP = torch.matmul(activeBlockVectorR.T.conj(), activeBlockVectorBP)
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T.conj()) / 2
                gramPBP = torch.matmul(activeBlockVectorP.T.conj(), activeBlockVectorBP)
            else:
                gramPBP = ident
            gramA = bmat(
                [
                    [gramXAX, gramXAR, gramXAP],
                    [gramXAR.T.conj(), gramRAR, gramRAP],
                    [gramXAP.T.conj(), gramRAP.T.conj(), gramPAP],
                ]
            )
            for i, GRAM in enumerate(
                [
                    [gramXBX, gramXBR, gramXBP],
                    [gramXBR.T.conj(), gramRBR, gramRBP],
                    [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP],
                ]
            ):
                for j, GRAM_ in enumerate(GRAM):
                    if GRAM_.device != dev_:
                        print(i, j)

            gramB = bmat(
                [
                    [gramXBX, gramXBR, gramXBP],
                    [gramXBR.T.conj(), gramRBR, gramRBP],
                    [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP],
                ]
            )
            gramA = (gramA + gramA.T.conj()) / 2

            _handle_gramA_gramB_verbosity(gramA, gramB)

            try:
                _lambda, eigBlockVector = eigh(gramA, B=gramB)
                if verbosityLevel >= 2 and iterationNumber % verboseEvery == 0:
                    print(f"gram XAX XAR XAP \n{gramXAX}\n{gramXAR}\n{gramXAP}")
                    print(f"gram RAR RAP PAP \n{gramRAR}\n{gramRAP}\n{gramPAP}")
                    print(f"gram XBX XBR XBP \n{gramXBX}\n{gramXBR}\n{gramXBP}")
                    print(f"gram RBR RBP PBP \n{gramRBR}\n{gramRBP}\n{gramPBP}")
                    print("")

            except RuntimeError:
                # try again after dropping the direction vectors P from RR
                restart = True

        if restart:
            gramA = bmat([[gramXAX, gramXAR], [gramXAR.T.conj(), gramRAR]])
            gramB = bmat([[gramXBX, gramXBR], [gramXBR.T.conj(), gramRBR]])
            gramA = (gramA + gramA.T.conj()) / 2

            _handle_gramA_gramB_verbosity(gramA, gramB)

            try:
                _lambda, eigBlockVector = eigh(gramA)
                if verbosityLevel >= 2 and iterationNumber % verboseEvery == 0:
                    print(f"gram XAX XAR RAR \n{gramXAX}\n{gramXAR}\n{gramRAR}")
                    print(f"gram XBX XBR RBR \n{gramXBX}\n{gramXBR}\n{gramRBR}")
                    print("")
            except RuntimeError as e:
                raise ValueError("eigh has failed in lobpcg iterations") from e

        ii = _get_indx(_lambda, sizeX, largest)

        if verbosityLevel >= 10 and iterationNumber % verboseEvery == 0:
            print("gram eigh result")
            print("lambda index : ", ii)
            print("_lambda : ", _lambda)
            print("eigvec : \n", eigBlockVector)
            print("")

        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]

        lambdaHistory.append(_lambda)

        # Compute Ritz vectors.
        if B is not None:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX : sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize :]

                pp = torch.matmul(activeBlockVectorR, eigBlockVectorR)
                pp += torch.matmul(activeBlockVectorP, eigBlockVectorP)

                app = torch.matmul(activeBlockVectorAR, eigBlockVectorR)
                app += torch.matmul(activeBlockVectorAP, eigBlockVectorP)

                bpp = torch.matmul(activeBlockVectorBR, eigBlockVectorR)
                bpp += torch.matmul(activeBlockVectorBP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = torch.matmul(activeBlockVectorR, eigBlockVectorR)
                app = torch.matmul(activeBlockVectorAR, eigBlockVectorR)
                bpp = torch.matmul(activeBlockVectorBR, eigBlockVectorR)

            if verbosityLevel > 10:
                print(pp)
                print(app)
                print(bpp)

            blockVectorX = torch.matmul(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = torch.matmul(blockVectorAX, eigBlockVectorX) + app
            blockVectorBX = torch.matmul(blockVectorBX, eigBlockVectorX) + bpp

            blockVectorP, blockVectorAP, blockVectorBP = pp, app, bpp

        else:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX : sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize :]

                pp = torch.matmul(activeBlockVectorR, eigBlockVectorR)
                pp += torch.matmul(activeBlockVectorP, eigBlockVectorP)

                app = torch.matmul(activeBlockVectorAR, eigBlockVectorR)
                app += torch.matmul(activeBlockVectorAP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = torch.matmul(activeBlockVectorR, eigBlockVectorR)
                app = torch.matmul(activeBlockVectorAR, eigBlockVectorR)

            if verbosityLevel > 10:
                print(pp)
                print(app)

            blockVectorX = torch.matmul(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = torch.matmul(blockVectorAX, eigBlockVectorX) + app

            blockVectorP, blockVectorAP = pp, app

    if B is not None:
        aux = blockVectorBX * _lambda.unsqueeze(0)
    else:
        aux = blockVectorX * _lambda.unsqueeze(0)

    blockVectorR = blockVectorAX - aux

    aux = torch.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = torch.sqrt(aux)

    # Future work: Need to add Postprocessing here:
    # Making sure eigenvectors "exactly" satisfy the blockVectorY constrains?
    # Making sure eigenvecotrs are "exactly" othonormalized by final "exact" RR
    # Computing the actual true residuals

    if verbosityLevel >= 0:
        print("\nfinal iteration :", iterationNumber)
        print("final eigenvalue:", _lambda)
        print("final residual norms:", residualNorms)
    if retLambdaHistory:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, lambdaHistory, residualNormsHistory
        else:
            return _lambda, blockVectorX, lambdaHistory
    else:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, residualNormsHistory
        else:
            return _lambda, blockVectorX
