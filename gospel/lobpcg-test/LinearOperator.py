"""
/home/jhwoo/.conda/envs/gospel/lib/python3.7/site-packages/scipy-1.6.1-py3.7-linux-x86_64.egg/scipy/sparse/linalg/interface.py
"""
import numpy as np
import torch
from scipy.sparse import isspmatrix

import torch.multiprocessing as mp


def aslinearoperator(A, A_32=None, multi=False):
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
        if A_32 is not None:
            matvec32 = lambda x: A_32 @ x
        else:
            matvec32 = None
        return LinearOperator(A.shape, 
                matvec, 
                dtype=A.dtype, 
                matvec32=matvec32,
                device=A.device)
    else:
        raise ValueError(f"{type(A)} is not supported type.")

class MultiOperator:
    def __init__(self, 
                 queue_size=5,
                 world_size=4, 
                 dtype=torch.float64):
        
        self.__dtype = dtype
        self.initialized = False
        self.world_size = world_size 
        self.queue_size = queue_size 

    def multiprocess_initilize(self,):
        if self.initialized:
            print("Re-initialize MultiLinearOperator") 
            self.delete_multiprocessing_source()
        
        self.initialized = True
        self.semaphore = mp.Semaphore(self.world_size * self.queue_size)
        self.q_in = mp.Queue()
        self.q_out = mp.Queue()
        self.processes = []
        
        for rank in range(self.world_size):
            args = self.make_args(rank,)
            p = mp.Process(target=self.func, args=args)
            self.processes.append(p)
            p.start()

    def make_args(self, rank):
        raise NotImplementedError

    def delete_multiprocessing_source(self,):
        self.q_in.close()
        self.q_out.close()
        for p in self.processes:
            p.terminate()
        #raise NotImplementedError
    
    def func(self, rank, q_in, q_out, semaphore):
        raise NotImplementedError
    
    def to_tensor(self,):
        raise NotImplementedError

    def to_batch(self,):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if not self.initialized:
            self.multiprocess_initialize()

        batch_size, batch_list = self.to_batch(*args, **kwargs)
        for idx, batch in enumerate(batch_list):
            self.q_in.put((idx, batch))
            self.semaphore.acquire()
        
        out = []
        for i in range(batch_size):
            out_ = self.q_out.get() 
            out.append(out_)

        out.sort(key=self.sort_func)        

        out = self.to_tensor(out, **kwargs)
        return out

class MultiLinearOperator:
    def __init__(self, A, size=4):
        self.A = A
        self.__shape = A.shape
        self.__dtype = A.dtype
        self.size = size
        self.initialized = False
    
    def multiprocess_initialize(self, ):
        if self.initialized:
            print('Re-initialize MulitLinearOperator')
            self.q_in.close()
            self.q_out.close()
            for p in self.processes:
                p.terminate()
        
        self.initialized = True
        self.semaphore = mp.Semaphore(self.world_size)
        self.q_in = mp.Queue()
        self.q_out = mp.Queue()
        self.processes = []
        A = self.A.to(self.dtype)

        for rank in range(self.world_size):
            p = mp.Process(
                    target=self.func, 
                    args=(A, rank, self.q_in, self.q_out, self.semaphore)
                    )
            self.processes.append(p)
            p.start()
    
    @staticmethod
    def func(self, A, rank, q_in, q_out, semaphore):
        A = A.to(f'cuda:{rank}')
        while True:
            idx, x = q_in.get()
            semaphore.release()
            if x.device != A.device:
                x = x.to(A.device)
            y = A@x
            q_out.put((idx, y))

    def __call__(self, X, size=5, dim=1):
        if not self.initialized:
            self.multiprocess_initialize()
        data = self.to_batch(X, size=size, dim=dim)
        N = len(data)
        #data = torch.split(x, size=size, dim=dim)

        for idx, x in enumerate(data):
            self.q_in.put((idx, x))
            self.semaphore.acquire()
        
        out = []
        for i in range(N):
            out_ = self.q_out.get() 
            out.append(out_)
        out.sort(key=self.sort_func)        
        out = torch.cat([x[1].to('cpu') for x in out], dim=dim)
        return out
    
    def to_batch(self, x, size=5, dim=1):
        return torch.split(x, size, dim=dim)

    @property
    def dtype(self,):
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype):
        if self.dtype != dtype:
            self.__dtype = dtype
            self.multiprocess_initialize()
            

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
    
    def __init__(
            self, 
            shape, 
            matvec, 
            dtype=None, 
            matvec32=None,
            device=None
            ):
        self.__shape = shape
        self.__matvec = matvec
        self.__matvec32 = matvec32
        self.__dtype = dtype
        self.device = device
        return
    
    def __call__(self, x):
        if self.dtype in [torch.float32, torch.complex64]:
            if self.__matvec32 is not None:
                print('FP32')
                return self.__matvec32(x)
        elif self.dtype in [torch.float64, torch.complex128]:
            print('FP64')
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
    
    @dtype.setter
    def dtype(self, dtype):
        self.__dtype = dtype

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
