import numpy as np
import time
import torch

torch.manual_seed(0)


def randDenseSym(N, matrix_type="diagonal", alpha=0.0):
    """Generate random sparse symmetric matrix.

    :type n: int
    :param n:
        dimension of matrix, thus matrix shape is (N, N)
    :type matrix_type: str
        type of matrix, options=["diagonal", "laplacian"], defaults to "diagonal"
    :type alpha: float:
    :param alpha:
        pre-factor of diagonal term, defaults to 1.0

    :rtype: torch.Tensor
    :return:
        sparse matrix with layout=torch.sparse_coo
    """
    from gospel.util import scipy_to_torch_sparse
    import scipy.sparse

    if matrix_type == "diagonal":
        diag_val = alpha * abs(torch.randn(N, dtype=torch.float64))
        T = torch.diag(diag_val)
    elif matrix_type == "laplacian":
        raise NotImplementedError
    else:
        raise NotImplementedError
    noise = torch.randn(N, N, dtype=torch.float64)
    return T + (noise + noise.conj().T) / 2


def randSparseSym(N, sparsity=0.99, matrix_type="diagonal", alpha=1.0):
    """Generate random sparse symmetric matrix.

    :type n: int
    :param n:
        dimension of one axis of grid, thus matrix shape is (N**3, N**3)
    :type sparsity: float
    :param sparsity:
        sparsity of sparse matrix, defaults to '0.1'. If sparsity=1.0, all elements of matrix is zero.
    :type matrix_type: str
        type of matrix, options=["diagonal", "laplacian"], defaults to "diagonal"

    :rtype: torch.Tensor
    :return:
        sparse matrix with layout=torch.sparse_coo
    """
    from gospel.util import scipy_to_torch_sparse
    import scipy.sparse

    if matrix_type == "diagonal":
        diag_idx = torch.arange(N**3)
        diag_idx = torch.vstack((diag_idx, diag_idx))
        diag_val = alpha * abs(torch.randn(N**3, dtype=torch.float64))
        T = torch.sparse_coo_tensor(diag_idx, diag_val, (N**3, N**3))
    elif matrix_type == "laplacian":
        from gospel.Grid import Grid
        from gospel.FdOperators import make_kinetic_op
        from ase import Atoms

        cell = np.array([0.4, 0.4, 0.4]) * N
        grid = Grid(atoms=Atoms(cell=cell), gpts=(N, N, N))
        # T = make_kinetic_op(grid, combine=True)
        # T = scipy_to_torch_sparse(T).to_sparse_coo()
        T = scipy.sparse.coo_matrix(make_kinetic_op(grid, combine=True))
        T = scipy_to_torch_sparse(T)
    else:
        raise NotADirectoryError

    nnz = int(N**3 * N**3 * (1 - sparsity) * 0.5)
    indices = torch.randint(N**3, (2, nnz))
    values = torch.randn(nnz, dtype=torch.float64)
    noise = torch.sparse_coo_tensor(indices, values, (N**3, N**3))

    # A = T + noise + noise.H
    A = T + (noise + noise.conj().t()) / 2
    A = A.coalesce()
    # print(f"Debug: sparsity = {1 - len(A.values()) / A.shape[0] / A.shape[1]}")
    return A


# def randomSparseSym(N, sparsity=0.99, matrix_type="diagonal"):
#     import scipy.sparse as sparse
#     import scipy.stats as stats
#     from gospel.util import scipy_to_torch_sparse
#
#     if matrix_type == "diagonal":
#         T = sparse.diags(np.random.randn(N**3))
#     elif matrix_type == "laplacian":
#         from gospel.Grid import Grid
#         from gospel.FdOperators import make_kinetic_op
#         from ase import Atoms
#
#         cell = np.array([0.4, 0.4, 0.4]) * N
#         grid = Grid(atoms=Atoms(cell=cell), gpts=(N, N, N))
#         T = make_kinetic_op(grid, combine=True)
#     else:
#         raise NotImplementedError
#
#     ## Make random symmetric matrix
#     rvs = stats.norm().rvs
#     rand_sym = sparse.random(N**3, N**3, density=(1 - sparsity), data_rvs=rvs)
#     upper = sparse.triu(rand_sym)
#     rand_sym = upper + upper.T - sparse.diags(rand_sym.diagonal())
#
#     A = T + rand_sym
#     print(f"Debug: sparsity = {1 - A.nnz / A.shape[0] / A.shape[1]}")
#     return scipy_to_torch_sparse(A)


if __name__ == "__main__":
    A = randSparseSym(50, sparsity=0.999, matrix_type="diagonal")
