from abc import ABCMeta, abstractmethod
from itertools import product
from typing import Optional
import torch
import numpy as np

from gospel.LinearOperator import LinearOperator
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.util import Timer


class Eigensolver(metaclass=ABCMeta):
    """Eigensolver parent class"""

    def __init__(self):
        self._type = None
        self._preconditioner = None
        self._starting_vector = None
        self._starting_value = None
        return

    def check_residue(self, hamiltonian, eigval, eigvec):
        residue = np.empty_like(eigvec)
        for i_s, i_k in product(range(eigvec.shape[0]), range(eigvec.shape[1])):
            residue[i_s, i_k] = np.linalg.norm(
                eigvec[i_s, i_k].T * eigval[i_s, i_k]
                - hamiltonian[i_s, i_k].dot(eigvec[i_s, i_k].T),
                axis=0,
            )
            print("Eigenvalue / Residue")
            for eigval, r in zip(eigval[i_s, i_k], residue[i_s, i_k]):
                print(f"{eigval:.2E} :  {r:.2E}")
        return residue

    @abstractmethod
    def diagonalize(self, hamiltonian, convg_tol=None, i_scf=None, bands=None):
        """abstract method for hamiltonian diagonalization"""
        pass

    def _initialize_guess(self, hamiltonian, orthogonalize=True):
        # TODO: Make the code simpler using the classmethod initialize_guess.
        # TODO: Add 'guess_type' arguments. (choices=['random', etc.])
        """Initialize guess vectors and values."""
        device = hamiltonian.device

        if self._starting_vector is None:
            nspins = hamiltonian.nspins
            nibzkpts = hamiltonian.nibzkpts
            nbands = hamiltonian.nbands
            ngpts = hamiltonian.ngpts
            # S = hamiltonian.get_S()  # overlap operator

            # self._starting_value = -np.random.rand(nspins, nibzkpts, nbands).astype(  float  )
            self._starting_value = torch.rand(
                nspins, nibzkpts, nbands, dtype=torch.float64, device=device
            )
            # self._starting_value = - torch.rand(nspins, nibzkpts, nbands, dtype= torch.float64, device = device )
            self._starting_vector = np.empty((nspins, nibzkpts), dtype=object)

            for i_s, i_k in product(range(nspins), range(nibzkpts)):
                dtype = hamiltonian.get_kinetic_operator(i_k).dtype

                psi_ = torch.randn(
                    PH.split_size(ngpts), nbands, dtype=dtype, device=device
                )
                psi_ = parallel_orthonormalize(psi_)
                psi = PH.redistribute(psi_, dim0=0, dim1=1)
                del psi_
                self._starting_vector[i_s, i_k] = psi
                # psi = torch.randn(ngpts, nbands, dtype=hamiltonian[i_s, i_k].dtype)
                # psi = torch.linalg.qr(psi)[0]
        return

    @classmethod
    def initialize_guess(cls, hamiltonian, orthogonalize=True):
        """Initialize guess vectors and values."""
        device = hamiltonian.device

        nspins = hamiltonian.nspins
        nibzkpts = hamiltonian.nibzkpts
        nbands = hamiltonian.nbands
        ngpts = hamiltonian.ngpts
        # S = hamiltonian.get_S()  # overlap operator

        _starting_value = torch.rand(
            nspins, nibzkpts, nbands, dtype=torch.float64, device=device
        )
        _starting_vector = np.empty((nspins, nibzkpts), dtype=object)

        for i_s, i_k in product(range(nspins), range(nibzkpts)):
            dtype = hamiltonian.get_kinetic_operator(i_k).dtype
            psi_ = torch.randn(PH.split_size(ngpts), nbands, dtype=dtype, device=device)
            psi_ = parallel_orthonormalize(psi_)
            psi = PH.redistribute(psi_, dim0=0, dim1=1)
            del psi_
            _starting_vector[i_s, i_k] = psi
            # psi = torch.randn(ngpts, nbands, dtype=hamiltonian[i_s, i_k].dtype)
        return _starting_value, _starting_vector

    @property
    def type(self):
        return self._type

    @property
    def preconditioner(self):
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, preconditioner):
        self._preconditioner = preconditioner
        return

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, inp):
        self._maxiter = inp
        return

    def set_initial_eigenpair(self, inp, device=None, orthonormalize=False):
        ## set starting eigenvalues and eigenvectors.

        val, vec = inp
        # val.shape=(nspins, nibzkpts, nbands)
        # vec.shape=(nspins, nibzkpts)---(nbands, ngpts)

        nspins, nibzkpts = vec.shape[0], vec.shape[1]
        self._starting_value = val.to(device)
        self._starting_vector = np.empty((nspins, nibzkpts), dtype=object)

        for i_s, i_k in product(range(nspins), range(nibzkpts)):
            nbands, ngpts = vec[i_s, i_k].shape
            if orthonormalize:
                psi_ = torch.split(vec[i_s, i_k].T, PH.split_size(ngpts, True))
                psi_ = psi_[PH.rank].to(device)
                psi_ = parallel_orthonormalize(psi_)
                psi = PH.redistribute(psi_, dim0=0, dim1=1)
                del psi_
            else:
                psi = torch.split(vec[i_s, i_k], PH.split_size(nbands, True))
                psi = psi[PH.rank].T.to(device)
            self._starting_vector[i_s, i_k] = psi
        return


def gen_eigh(A, B=None):
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
    if B is None:
        val, vec = torch.linalg.eigh(A)
    else:
        L = torch.linalg.cholesky(B)
        Linv = torch.linalg.inv(L)
        _A = (Linv @ A) @ Linv.conj().T
        val, vec = torch.linalg.eigh(_A)
        vec = Linv.conj().T @ vec
    return val, vec


def parallel_orthonormalize(
    X_: torch.Tensor,
    B: Optional[LinearOperator] = None,
    use_MP: bool = False,
    proj_fp: Optional[torch.dtype] = None,
    chol_fp: Optional[torch.dtype] = None,
    rot_fp: Optional[torch.dtype] = None,
    out_fp: Optional[torch.dtype] = None,
    timing: bool = False,
    verbosity: int = 0,
) -> Optional[torch.Tensor]:
    """
    B-orthonormalize the given block vector using Cholesky.
    so that any :math:`v_i,v_j`  :math:`v_iBv_j=\delta_{ij}`

    :type X_: torch.Tensor
    :param X_:
        vectors to orthonormalize which is distributed along 0th axis
    :type B: torch.Tensor or gospel.LinearOperator
    :param B:
        A linear system of size (n, n) that is positive definite hermitian, defaults to None
    :type use_MP: bool
    :param use_MP:
        whether to use mixed-precision, defaults to False
    :type proj_fp: torch.dtype, optional
    :param proj_fp:
        data type for projection matrix, defaults to None
    :type chol_fp: torch.dtype, optional
    :param chol_fp:
        data type for Cholesky decomposition, defaults to None
    :type rot_fp: torch.dtype, optional
    :param rot_fp:
        data type for rotation matrix, defaults to None
    :type out_fp: torch.dtype, optional
    :param out_fp:
        data type for output, defaults to None
    :type timing: bool, optional
    :param timing:
        whether to measure the elapsed time or not, defaults to False

    :rtype: torch.Tensor
    :return:
        orthonormalized vectors which is distributed along 0th axis
    """
    device = X_.device
    if B is not None:
        raise NotImplementedError()

    if use_MP:
        X_ = X_.to(proj_fp)

    # Normalization for numerical stability
    normalization = (
        PH.all_reduce(X_.abs().max(axis=0)[0], op=torch.distributed.ReduceOp.MAX)
        + torch.finfo(X_.dtype).eps
    )
    X_ = X_ / normalization

    # Projection
    with Timer.track("Projection (Ortho.)", timing, verbosity):
        cov = PH.all_reduce(X_.T.conj() @ X_)
    if use_MP:
        cov = cov.to(chol_fp)
    cov = (cov + cov.T.conj()) / 2  # for numerical stability

    # Rotation
    try:
        with Timer.track("inv & Cholesky (Ortho.)", timing, verbosity):
            LTinv = torch.linalg.inv(torch.linalg.cholesky(cov, upper=True))
        with Timer.track("Rotation (Ortho.)", timing, verbosity):
            if use_MP:
                return (X_.to(rot_fp) @ LTinv.to(rot_fp)).to(out_fp)
            else:
                return X_ @ LTinv
    except:
        return None
