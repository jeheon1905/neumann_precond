"""
dtype=float32 & float64 accuracy test.

Reference: float64 on CPU
1. diff of float64 on GPU
2. diff of float32 on CPU
3. diff of float32 on GPU
"""
import numpy as np
import torch
from time import time
# from gospel.LinearOperator import LinearOperator, aslinearoperator
from LinearOperator import LinearOperator, aslinearoperator
np.random.seed(0)
print(torch.__path__)
print(torch.__version__)
print(np.__path__)
print(np.__version__)

np_dtype = np.float64

# m, n = 10000, 10000
m, n = 1000, 1000
M = np.random.randn(m ,n).astype(np_dtype)
X = np.random.randn(n, 10).astype(np_dtype)
X, _ = np.linalg.qr(X)  # orthonormal vector
print(f"* M.shape : {M.shape}")
print(f"* X.shape : {X.shape}")

M64_op = aslinearoperator(M)
val64_np = M64_op @ X
M32_op = aslinearoperator(M.astype(np.float32))
val32_np = M32_op @ X.astype(np.float32)

X64_torch = torch.from_numpy(X)
M64_torch = torch.from_numpy(M)
M64_torch_op = aslinearoperator(M64_torch)
val64_torch = M64_torch_op @ X64_torch

X32_torch = X64_torch.to(dtype=torch.float32)
M32_torch = M64_torch.to(dtype=torch.float32)
M32_torch_op = aslinearoperator(M32_torch)
val32_torch = M32_torch_op @ X32_torch

def calc_diff(A, B):
    if isinstance(A, torch.Tensor) and A.device != torch.device(type="cpu"):
        _A = np.array(A.cpu())
    else:
        _A = np.array(A)
    if isinstance(B, torch.Tensor) and B.device != torch.device(type="cpu"):
        _B = np.array(B.cpu())
    else:
        _B = np.array(B)
    # return np.linalg.norm(_A - _B)
    return abs(_A - _B).mean()

print(f"diff(val64_np, val64_torch) = {calc_diff(val64_np, val64_torch)}")
print(f"diff(val64_np, val32_np)    = {calc_diff(val64_np, val32_np)}")
print(f"diff(val64_np, val32_torch) = {calc_diff(val64_np, val32_torch)}")

if torch.cuda.device_count():
    print(torch.cuda.get_device_name(device=None))

    M64_torch_gpu = M64_torch.cuda()
    X64_torch_gpu = X64_torch.cuda()
    M32_torch_gpu = M32_torch.cuda()
    # M32_torch_gpu = M64_torch_gpu.to(torch.float32)
    X32_torch_gpu = X32_torch.cuda()
    # X32_torch_gpu = X64_torch_gpu.to(torch.float32)

    M64_torch_gpu_op = aslinearoperator(M64_torch_gpu)
    val64_torch_gpu = M64_torch_gpu_op @ X64_torch_gpu
    M32_torch_gpu_op = aslinearoperator(M32_torch_gpu)
    val32_torch_gpu = M32_torch_gpu_op @ X32_torch_gpu

    print(f"diff(val64_np, val64_torch_gpu) = {calc_diff(val64_np, val64_torch_gpu)}")
    print(f"diff(val64_np, val32_torch_gpu) = {calc_diff(val64_np, val32_torch_gpu)}")
    print(f"diff(val64_torch, val64_torch_gpu) = {calc_diff(val64_torch, val64_torch_gpu)}")
    print(f"diff(val32_torch, val32_torch_gpu) = {calc_diff(val32_torch, val32_torch_gpu)}")
