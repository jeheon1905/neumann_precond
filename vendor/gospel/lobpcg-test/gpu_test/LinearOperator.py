"""
/home/jhwoo/.conda/envs/gospel/lib/python3.7/site-packages/scipy-1.6.1-py3.7-linux-x86_64.egg/scipy/sparse/linalg/interface.py
"""
import numpy as np
from scipy.sparse import isspmatrix


def aslinearoperator(A):
    """Return A as a LinearOperator.

    :type  A: np.ndarray, scipy.sparse, LinearOperator
    :param A:

    :rtype: LinearOperator
    :return:
        LinearOperator class object
    """
    import torch

    if A is None:
        return None
    elif isinstance(A, LinearOperator):
        return A
    elif isinstance(A, np.ndarray) or isinstance(A, torch.Tensor) or isspmatrix(A):
        if A.ndim != 2:
            raise ValueError("array must have ndim == 2")
        matvec = lambda x: A @ x
        return LinearOperator(A.shape, matvec, A.dtype)
    else:
        raise ValueError(f"{type(A)} is not supported type.")


class LinearOperator:
    """user-specified operations.

    :type  shape: np.ndarray
    :param shape:
        shape of operator
    :type  matvec: function
    :param matvec:
        user-specified 'matrix-vector multiplication' operation
    :type  dtype: type
    :param dtype:
        Data type of the matrix

    **Example**

    >>> from gospel.LinearOpertor import LinearOperator
    >>> f = lambda x: ~~
    >>> H = LinearOperator()
    >>>
    """

    def __init__(self, shape, matvec, dtype=None):
        self.__shape = shape
        self.__matvec = matvec
        self.__dtype = dtype
        return

    def __call__(self, x):
        return self.__matvec(x)

    def __mul__(self, x):
        return self.dot(x)

    def dot(self, x):
        """Matrix-matrix multiplication"""
        return self.__matvec(x)

    def __matmul__(self, x):
        return self.__mul__(x)

    @property
    def dtype(self):
        return self.__dtype

    @property
    def shape(self):
        return self.__shape


if __name__ == "__main__":
    import numpy as np

    np.random.seed(0)

    A = np.random.randn(2, 2)
    B = np.random.randn(2, 2)

    f = lambda x: A @ x + A**2 @ x

    op = LinearOperator(shape=A.shape, matvec=f, dtype=A.dtype)

    print(op)
    print(f"A @ B + A**2 @ B = \n{A @ B + A**2 @ B}")
    print(f"op @ B = \n{op @ B}")
    print(f"op(B) = \n{op(B)}")
    # print(B @ op)  # -> error

    ## Define diagonal() method ##
    def diaognal():
        return A.diagonal() + (A**2).diagonal()

    op.diagonal = diaognal  # you can use diagonal() as you defined.
    print(f"op.diagonal() = {op.diagonal()}")

    ## Define mmm() method ##
    def mmm(x):
        return x.conj().T @ A @ x

    op.mmm = mmm
    print(f"op.mmm(B) = \n{op.mmm(B)}")
    print(f"B.T @ A @ B = \n{B.T @ A @ B}")
