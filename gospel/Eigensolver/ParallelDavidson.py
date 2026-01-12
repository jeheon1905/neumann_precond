import warnings
from itertools import product
import numpy as np
import torch
from typing import Tuple, Optional

from gospel.LinearOperator import LinearOperator, aslinearoperator
from gospel.Eigensolver.Eigensolver import (
    Eigensolver,
    gen_eigh,
    parallel_orthonormalize,
)
from gospel.Eigensolver.precondition import Preconditioner
from gospel.util import Timer, vprint
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.Hamiltonian import Hamiltonian
from gospel.precision import to_DP, to_SP, to_HP, to_BF16


def _is_normalized(X: torch.Tensor, thr: float = 1e-10) -> bool:
    """
    Check if the given matrix X is properly normalized.

    The function verifies whether all column vectors in X satisfy ||x|| = 1.
    If the deviation exceeds the given threshold, normalization is considered broken.
    """
    # Compute ||X|| - 1 to check if normalization is broken
    norms_diff = torch.abs((X.conj() * X).sum(dim=0).real.sqrt() - 1.0)
    is_broken_local = (norms_diff > thr).any()

    # Merge across all ranks to check if any rank has broken normalization
    is_broken_global = PH.merge(is_broken_local.unsqueeze(0)).sum().to(bool)
    return not is_broken_global


class Davidson(Eigensolver):
    """Block Davidson iterative eigenvalue solver. (https://arxiv.org/abs/0906.2569)

    :type maxiter: int, optional
    :param maxiter:
        the maximum number of iterations, defaults to 5
    :type nblock: int, optional
    :param nblock:
        the maximum size of block, defaults to 3
    :type locking: bool
    :param locking:
        whether to lock converged eigenvectors or not
    :type verbosity: int, optional
    :param verbosity:
        verbose level (0 or 1), defaults 1
    :type fill_block: bool, optional
    :param fill_block:
        defaults to True
    """

    def __init__(
        self,
        maxiter: int = 5,
        nblock: int = 2,
        locking: bool = True,
        verbosity: int = 0,
        fill_block: bool = True,
        use_MP: bool = False,
        MP_dtype: str = "SP",
        MP_scheme: int = 1,
    ):
        super().__init__()
        self._type = "davidson"
        self._maxiter = maxiter
        # TODO: the class variables will be relaced to parent class's variables.
        self.__nblock = nblock
        self.__locking = locking
        self.__verbosity = verbosity
        self.__fill_block = fill_block
        if fill_block and PH.global_size > 1:
            raise ValueError("Parallel case (global_size>1) needs fill_block=False.")
        self.__use_MP = use_MP
        self.__MP_dtype = MP_dtype
        self.__MP_scheme = MP_scheme
        return

    def __str__(self):
        info = (
            f"\n================ [ Eigensolver(ParallelDavidson) ] ================="
            f"\n* maxiter            : {self._maxiter}"
            f"\n* nblock             : {self.__nblock}"
            f"\n* locking            : {self.__locking}"
            f"\n* verbosity          : {self.__verbosity}"
            f"\n* fill_block         : {self.__fill_block}"
            f"\n* use_MP             : {self.__use_MP}"
            f"\n* MP_dtype           : {self.__MP_dtype}"
            f"\n* MP_scheme          : {self.__MP_scheme}"
            f"\n====================================================================\n"
        )
        return info

    @Timer.timeit
    def diagonalize(
        self,
        hamiltonian: Hamiltonian,
        convg_tol: float = 1e-4,
        i_scf: int = None,
        bands: int = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
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
        :type  bands: int or None, optional
        :param bands:
            number of lowest bands to check convergence, defaults to None (all bands)

        :rtype: (np.ndarray, np.ndarray)
        :return:
            eigenvalues and eigenvectors,
            eigval.shape=(nspins, nibzkpts, nbands)
            eigvec.shape=(nspins, nibzkpts)---(nbands, ngpts)
        """
        self._initialize_guess(hamiltonian)

        # TODO: remove eigval and eigvec for memory efficiency
        # eigval = torch.zeros_like(self._starting_value)
        eigvec = np.zeros_like(self._starting_vector)  # eigvec.dtype = object

        # Solve EVP for Hamiltonians
        for i_s, i_k in product(range(hamiltonian.nspins), range(hamiltonian.nibzkpts)):
            A = hamiltonian[i_s, i_k]
            solve_options = {
                "A": A,
                "X": self._starting_vector[i_s, i_k],
                "B": hamiltonian.get_S(),
                "preconditioner": self.preconditioner,
                "tol": convg_tol,
                "maxiter": self._maxiter,
                "nblock": self.__nblock,
                "locking": self.__locking,
                "fill_block": self.__fill_block,
                "verbosity": self.__verbosity,
                "i_scf": i_scf,
                "use_MP": self.__use_MP,
                "MP_dtype": self.__MP_dtype,
                "MP_scheme": self.__MP_scheme,
                "bands": bands,
            }
            val, vec = davidson(**solve_options)

            # eigval[i_s, i_k] = val
            eigvec[i_s, i_k] = vec.T
            # TODO: change the eigvec.shape to (ngpts, nbands)
            # to do fix Density.calc_density, Hamiltonian.calc_and_print_energies, Hamiltonian.calc_forces, etc.

            ## update starting vectors and values
            self._starting_value[i_s, i_k] = val
            self._starting_vector[i_s, i_k] = vec  # not efficient memory
        # return eigval, eigvec
        return self._starting_value, eigvec

    @property
    def use_MP(self):
        return self.__use_MP

    @property
    def MP_dtype(self):
        return self.__MP_dtype


@Timer.timeit
def davidson(
    A: LinearOperator,
    X: torch.Tensor,
    B: Optional[LinearOperator] = None,
    preconditioner: Optional[Preconditioner] = None,
    tol: float = 1e-4,
    maxiter: int = 20,
    nblock: int = 2,
    locking: bool = True,
    fill_block: bool = True,
    verbosity: int = 0,
    ortho: str = "cholesky",
    retHistory: bool = False,
    i_scf: int = 0,
    skip_init_ortho: bool = True,
    timing: bool = False,
    use_MP: bool = False,
    MP_dtype: str = "SP",
    MP_scheme: int = 1,
    debug_recalc_convg_history: bool = False,
    bands: int = None,
) -> Tuple:
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
    :type preconditioner: Preconditioner or None, optional
    :param preconditioner:
        preconditioner to accelerate the convergence rate
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
        whether to lock converged eigenvectors or not, defaults to True
    :type fill_block: bool, optional
    :param fill_block:
        whether to fill the given blocks fully, defaults to True
    :type verbosity: int, optional
    :param verbosity:
        verbosity level, defaults to 0
    :type ortho: str, optional
    :param ortho:
        orthogonalize method, options=["qr", "cholesky"], defaults to "cholesky"
    :type retHistory: bool, optional
    :param retHistory:
        whether to return eigenvalue and residual history, defaults to False
    :type i_scf: int, optional
    :param i_scf:
        index of SCF iteration, defaults to 0
    :type skip_init_ortho: bool, optional
    :param skip_init_ortho:
        whether to skip the initial orthonormalization, defaults to True
    :type timing: bool, optional
    :param timing:
        whether to measure the elapsed time or not, defaults to False
    :type use_MP: bool, optional
    :param use_MP:
        whether to use mixed precision or not, defaults to False
    :type MP_dtype: str
    :param MP_dtype:
        mixed precision data type, options=["SP", "HP"], defaults to "SP"
    :type debug_recalc_convg_history: bool, optional
    :param debug_recalc_convg_history:
        whether to recalculate the convergence history or not, defaults to False
    :type  bands: int or None, optional
    :param bands:
        number of lowest bands to check convergence, defaults to None (all bands)

    :rtype: tuple[torch.Tensor]
    :return:
        eigenvalues and eigenvectors

    - Description of MP schemes

    | MP_scheme | Rotation | Preconditioning | Projections | A, B | eigh | Cholesky | inv |
    |-----------|----------|-----------------|-------------|------|------|----------|-----|
    | 1         | SP       | SP              | SP          | SP   | DP   | DP       | DP  |
    | 2         | DP       | SP              | SP          | DP   | DP   | DP       | DP  |
    | 3         | DP       | SP              | SP          | DP   | DP   | DP       | DP  |
    | 4         | SP       | SP              | DP          | SP   | DP   | DP       | DP  |
    | 5         | SP       | SP              | DP          | DP   | DP   | DP       | DP  |
    """
    # Distribution info
    # all  : residue, A_sub_list, B_sub_list, eigval, C, is_convg, unlock
    # split: AX, BX, R, is_convg, U, X, my_convg
    # NOTE: This function updates X in-place for memory efficiency.

    assert ortho in ["cholesky"], f"Invalid ortho method: {ortho}"
    assert MP_dtype in ["SP", "HP", "BF16"], f"Invalid MP dtype: {MP_dtype}"

    # reset num_call (for SCF calculation, num_call need to be reinitialized)
    if preconditioner is not None:
        preconditioner.reset_num_called() 

    # Initialization
    vprint("-*" * 50, "i_iter=1", verbosity=verbosity)
    device = X.device
    Timer.start("Diag. Iter.", timing)
    A = aslinearoperator(A)
    B = aslinearoperator(B)
    FP = X.dtype
    SP = {"SP": to_SP, "HP": to_HP, "BF16": to_BF16}[MP_dtype](FP)

    if use_MP:
        """
        Scheme 1, 2, 4: DP -> SP for X
        Scheme 3, 5   : DP -> DP for X (no change)
        """
        FP = to_DP(FP)  # only supports FP=DP
        _dtype = SP if MP_scheme in [1, 2, 4] else None
        X = X.to(_dtype)
        assert MP_scheme in [1, 2, 3, 4, 5], f"Invalid MP scheme: {MP_scheme}"

    X_ = PH.redistribute(X)
    neig = X_.shape[-1]

    # Containers for convergence history
    if retHistory:
        eigHistory = []
        resHistory = []

    # Decide whether to orthonormalize after checking the norms of the vectors
    if not skip_init_ortho:
        if not _is_normalized(X, thr=1e-10):
            with Timer.track("Orthonormalize X", timing, verbosity):
                X_ = parallel_orthonormalize(
                    X_,
                    use_MP=use_MP,
                    proj_fp=SP if MP_scheme in [1, 2, 3] else FP,
                    chol_fp=FP,
                    rot_fp=SP if MP_scheme in [1, 4, 5] else FP,
                    out_fp=SP if MP_scheme in [1, 2, 4] else FP,
                    timing=timing,
                    verbosity=verbosity,
                )
            # in-place update for memory efficiency
            X[:, :] = PH.redistribute(X_, dim0=0, dim1=1)

    # Subspace projection
    with Timer.track("A @ X, B @ X & Redistribution", timing, verbosity):
        AX_ = PH.redistribute(A @ X)
        BX_ = PH.redistribute(B @ X) if B is not None else None

    with Timer.track("Projection (X.H @ AX)", timing, verbosity):
        _dtype = SP if use_MP and MP_scheme in [1, 2, 3] else FP
        XT_ = X_.T.conj().to(_dtype)
        A_sub = PH.all_reduce(XT_ @ AX_.to(_dtype)).to(FP)
        B_sub = PH.all_reduce(XT_ @ BX_.to(_dtype)).to(FP) if B is not None else None
        del XT_

    # Subspace diagonalization
    with Timer.track("Subspace Diagonalization", timing, verbosity):
        if B is None:
            eigval, C = torch.linalg.eigh(A_sub)
        else:
            eigval, C = gen_eigh(A_sub, B_sub)

    # Rotation
    with Timer.track("Rotation", timing, verbosity):
        _dtype = SP if use_MP and MP_scheme in [1, 4, 5] else FP
        C = C.to(_dtype)
        X_ = X_.to(_dtype) @ C
        AX_ = AX_.to(_dtype) @ C
        # If A is highly sparse, the following line can be more efficient.
        # TODO: make it optional?
        # AX_ = PH.redistribute(A @ PH.redistribute(X_, dim0=0, dim1=0))
        BX_ = BX_ @ C if B is not None else X_
        del C
    Timer.stop("Diag. Iter.", timing, verbosity)

    # Start iteration loop
    find = False
    sub_dim = 0
    unlock = torch.full((neig,), True, device=device)
    bands = neig if bands is None else bands

    _dtype = SP if use_MP and MP_scheme in [1, 2, 4] else FP
    AU_list = torch.empty(
        X_.shape[0],
        nblock * neig,
        device=device,
        dtype=_dtype,
    )
    BU_list = torch.empty_like(AU_list) if B is not None else None

    for i_iter in range(2, maxiter + 1):
        Timer.start("Diag. Iter.", timing)
        vprint("\n" + "=*" * 50, f"i_iter={i_iter}", verbosity=verbosity)

        U_ = X_
        AU_list[:, : AX_.shape[1]] = AX_  # memory inefficiency (causing a deep copy)
        # Solutions
        # - AX_를 AU_list로부터 slicing해서 사용?
        # - AU_list를 list로 만들기?
        if B is not None:
            BU_list[:, : BX_.shape[1]] = BX_

        # Subspace projection
        with Timer.track("Projection (X.H @ AX)", device, verbosity):
            _dtype = SP if use_MP and MP_scheme in [1, 2, 3] else FP
            XT_ = X_.T.conj().to(_dtype)
            A_sub_list = [PH.all_reduce(XT_ @ AX_.to(_dtype))]
            B_sub_list = (
                [PH.all_reduce(XT_ @ BX_.to(_dtype))] if B is not None else None
            )
            del XT_

        # Start block expansion
        sub_dim = A_sub_list[-1].shape[-1]  # dimension of the subspace
        sub_dim_list = [sub_dim]
        i_b = 1  # the number of expansions

        while True:
            # Check the number of expansions
            if fill_block:
                if sub_dim == nblock * neig:
                    vprint(f"Stop block expansion.", verbosity=verbosity)
                    break
            else:
                if i_b == nblock:
                    vprint(
                        f"Stop block expansion. ({i_b} by {i_b})",
                        verbosity=verbosity,
                    )
                    break

            # Calculate residual
            with Timer.track("AX - \lambda BX", timing, verbosity):
                _slicing = Ellipsis if unlock.all() else unlock
                _eigval = eigval.to(BX_.dtype)[_slicing].unsqueeze(0)
                R_ = AX_[:, _slicing] - _eigval * BX_[:, _slicing]
            residue = torch.sqrt(PH.all_reduce(torch.sum(R_.conj() * R_, dim=0).real))
            # vprint(f"residual norms: {residue}", verbosity=verbosity)

            if retHistory and i_b == 1:
                with Timer.track("Save History", timing, verbosity):
                    if debug_recalc_convg_history:
                        with Timer.track("Recalc. Residual", timing, verbosity):
                            _eigval, _residue = recalc_convg_history(
                                A, B, X_, dtype=to_DP(FP)
                            )
                        eigHistory.append(_eigval.to("cpu"))
                        resHistory.append(_residue.to("cpu"))
                    else:
                        eigHistory.append(eigval.to("cpu"))
                        resHistory.append(residue.to("cpu"))

            R = PH.redistribute(R_, dim0=0, dim1=1)
            del R_

            # Check convergence and lock converged states
            with Timer.track("Locking & Fill Block", timing, verbosity):
                is_convg = residue < tol
                my_unlock_eigval = PH.split(eigval[unlock]).clone()

                # Convergence check for bands parameter
                current_convg_status = ~unlock
                current_convg_status[unlock] = is_convg
                if torch.all(current_convg_status[:bands]):
                    find = True
                    break

                if locking:
                    unlock[unlock.clone()] = ~is_convg
                    R = R[:, ~PH.split(is_convg)]
                    my_unlock_eigval = my_unlock_eigval[~PH.split(is_convg)]

                # Check block is saturated
                if fill_block:
                    if sub_dim == nblock * neig:
                        vprint(f"Stop block expansion.", verbosity=verbosity)
                        break
                    else:
                        if sub_dim + len(residue) <= nblock * neig:
                            pass
                        else:
                            cutBlock = nblock * neig - sub_dim
                            ii = torch.argsort(residue[~is_convg], descending=False)
                            R = R[:, ii < cutBlock]
                            my_unlock_eigval = my_unlock_eigval[ii < cutBlock]

            # Preconditioning
            if preconditioner is not None:
                with Timer.track("Preconditioning", timing, verbosity):
                    if use_MP:
                        R = R.to(SP)
                        my_unlock_eigval = my_unlock_eigval.to(SP)
                    R = preconditioner(R, A, my_unlock_eigval, i_scf)
            del my_unlock_eigval

            # Orthonormalization
            R_ = PH.redistribute(R)
            del R

            with Timer.track("Orthonormalize R", timing, verbosity):

                with Timer.track("Projection (Ortho.)", timing, verbosity):
                    _dtype = SP if use_MP and MP_scheme in [1, 2, 3] else FP
                    tmp = PH.all_reduce(U_.to(_dtype).conj().T @ R_.to(_dtype))

                with Timer.track("Rotation (Ortho.)", timing, verbosity):
                    _dtype = SP if use_MP and MP_scheme in [1, 4, 5] else FP
                    tmp = U_.to(_dtype) @ tmp.to(_dtype)

                R_ -= tmp
                del tmp

                R_ = parallel_orthonormalize(
                    R_,
                    use_MP=use_MP,
                    proj_fp=SP if MP_scheme in [1, 2, 3] else FP,
                    chol_fp=FP,
                    rot_fp=SP if MP_scheme in [1, 4, 5] else FP,
                    out_fp=SP if MP_scheme in [1, 2, 3] else FP,
                    timing=timing,
                    verbosity=verbosity,
                )

                if R_ is None:
                    warnings.warn(
                        f"Failed orthonormalization at i_iter={i_iter}, i_b={i_b}",
                        UserWarning,
                    )
                    break

            R = PH.redistribute(R_, dim0=0, dim1=1)
            sub_dim_list.append(R_.shape[1])
            sub_dim = sum(sub_dim_list)

            # Hamiltonian operation
            with Timer.track(
                "A @ R, B @ R & Redistribution", timing, verbosity
            ):
                _slicing = slice(sum(sub_dim_list[:-1]), sum(sub_dim_list))
                AU_list[:, _slicing] = PH.redistribute(A @ R)
                if B is not None:
                    BU_list[:, _slicing] = PH.redistribute(B @ R)
            del R

            # Subspace projection
            with Timer.track("Projection (R.H @ AU)", timing, verbosity):
                _dtype = SP if use_MP and MP_scheme in [1, 2, 3] else FP

                for i in range(i_b + 1):
                    _slicing = slice(sum(sub_dim_list[:i]), sum(sub_dim_list[: i + 1]))
                    A_sub_list.append(
                        PH.all_reduce(R_.conj().T @ AU_list[:, _slicing].to(_dtype))
                    )
                    if B is not None:
                        B_sub_list.append(
                            PH.all_reduce(R_.conj().T @ BU_list[:, _slicing].to(_dtype))
                        )

            with Timer.track("Stack U", timing, verbosity):
                U_ = torch.hstack([U_, R_]).contiguous()
                del R_

            with Timer.track("Fill Subspace Matrix", timing, verbosity):
                # Construct subspace matrices
                enum = 0
                A_sub = torch.zeros((sub_dim, sub_dim), dtype=FP, device=device)
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

            # Subspace diagonalization
            with Timer.track("Subspace Diagonalization", timing, verbosity):
                if B is None:
                    eigval, C = torch.linalg.eigh(A_sub)
                else:
                    eigval, C = gen_eigh(A_sub, B_sub)
                eigval, C = eigval[:neig], C[:, :neig]

            # Rotation
            with Timer.track("Rotation", timing, verbosity):
                sum_sub_dim_list = sum(sub_dim_list)
                if sum_sub_dim_list == AU_list.shape[1]:
                    _slicing = Ellipsis  # no advanced indexing
                else:
                    _slicing = slice(sum_sub_dim_list)

                _dtype = SP if use_MP and MP_scheme in [1, 4, 5] else FP
                C = C.to(_dtype)

                # If FP is not the same as the original one, it converts its dtype.
                X_ = U_.to(_dtype) @ C
                AX_ = AU_list[:, _slicing].to(_dtype) @ C
                # If A is highly sparse, the following line can be more efficient.
                # AX_ = PH.redistribute(A @ PH.redistribute(X_, dim0=0, dim1=0))
                BX_ = BU_list[:, _slicing].to(_dtype) @ C if B is not None else X_

            i_b += 1

        Timer.stop("Diag. Iter.", timing, verbosity)
        if find:
            if retHistory:
                with Timer.track("Save History", timing, verbosity):
                    if debug_recalc_convg_history:
                        with Timer.track("Recalc. Residual", timing, verbosity):
                            _eigval, _residue = recalc_convg_history(
                                A, B, X_, dtype=to_DP(FP)
                            )
                        eigHistory.append(_eigval.to("cpu"))
                        resHistory.append(_residue.to("cpu"))
                    else:
                        eigHistory.append(eigval.to("cpu"))
                        resHistory.append(residue.to("cpu"))
            break

    if find:
        vprint(
            f"* Block Davidson converged at (iter={i_iter}/{maxiter}, b={sub_dim}/{neig * nblock}) with residual norms: {residue}",
            verbosity=verbosity,
        )
    else:
        vprint(
            f"Block Davidson did not converge with residual norms: {residue}",
            verbosity=verbosity,
        )

    # in-place update for memory efficiency
    X[:, :] = PH.redistribute(X_, dim0=0, dim1=0)
    X = X.to(FP)
    eigval = eigval.to(FP)

    return (eigval, X, eigHistory, resHistory) if retHistory else (eigval, X)


def recalc_convg_history(
    A: LinearOperator,
    B: LinearOperator,
    X_: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recalculate the convergence history for the given eigenvectors."""

    X_ = X_.to(dtype)
    X = PH.redistribute(X_, dim0=0, dim1=1).to(dtype)
    AX_ = PH.redistribute(A @ X)
    BX_ = PH.redistribute(B @ X) if B is not None else X_
    A_sub = PH.all_reduce(X_.T.conj() @ AX_)
    B_sub = PH.all_reduce(X_.T.conj() @ BX_) if B is not None else None

    if B is None:
        eigval, C = torch.linalg.eigh(A_sub)
    else:
        eigval, C = gen_eigh(A_sub, B_sub)

    R_ = AX_ - eigval.unsqueeze(0) * BX_
    residue = torch.sqrt(PH.all_reduce(torch.sum(R_.conj() * R_, dim=0).real))
    return eigval, residue


if __name__ == "__main__":
    A = torch.rand(100, 100).double()
    A = A + A.T
    X = torch.rand(100, 4).double()
    print(A)
    print(davidson(A, X))
