"""Speed comparison between scipy.sparse and torch.sparse

https://pytorch.org/docs/1.10/sparse.html
torch.sparse is faster than scipy.sparse !! (~3 times)
"""
import scipy.sparse as sparse
import torch
import numpy as np
from time import time

np_dtype = np.float64
# np_dtype = np.float32

N = 3000
A = np.random.randn(N, N).astype(np_dtype)
B = np.zeros((N, N))
for i in range(5):
    B += np.diag(np.random.randn(len(B) - i), i)
B = (B + B.T) / 2
B = B.astype(np_dtype)

st = time()
_ = A @ B
print(f"Time: {time() - st} sec")

B_sp = sparse.csr_matrix(B)
st = time()
_ = A @ B_sp
print(f"Time: {time() - st} sec")

######################################################

A = torch.from_numpy(A)
B = torch.from_numpy(B)

st = time()
_ = A @ B
print(f"Time: {time() - st} sec")

# B_sp = B.to_sparse_csr()
# B_sp = torch.sparse_coo_tensor(B)
B_sp = B.to_sparse()
print(B_sp.layout)
st = time()
val1 = torch.sparse.mm(B_sp, A)
print(f"Time: {time() - st} sec")

# B_sp = B.to_sparse_csr()
B_sp = B_sp.to_sparse_csr()
print(B_sp.layout)
st = time()
val2 = torch.sparse.mm(B_sp, A)
print(f"Time: {time() - st} sec")

print(f"|val1 - val2| = {abs(val1 - val2).sum()}")
