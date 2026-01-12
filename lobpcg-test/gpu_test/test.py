"""
matmul (@) speed test
1. numpy vs torch
2. torch (CPU) vs torch (GPU)
"""
import numpy as np
import torch
from time import time
np.random.seed(0)


## Set dtype and make matrix
st = time()
# dtype = "float32"
dtype = "float64"
if dtype == "float32":
    torch.set_default_dtype(torch.float32)
    np_dtype = np.float32
else:
    torch.set_default_dtype(torch.float64)
    np_dtype = np.float64
print(f"* data type : {dtype}")

num_iter = 100
# m, n = 10000, 10000
m, n = 1000, 1000
M = np.random.rand(m ,n).astype(np_dtype)
print(f"* Matrix : {m} x {n}")
print(f"Initialization Time: {time() - st} sec")


## numpy ##
st = time()
for i in range(num_iter):
    _ = M @ M
print(f" Numpy time : {time() - st} sec")


## pytorch ##
st = time()
M = torch.from_numpy(num_iter)
for i in range(num_iter):
    _ = M @ M
print(f" torch time : {time() - st} sec")

## pytorch (GPU) ##
if torch.cuda.device_count():
    print(torch.cuda.get_device_name(device=None))
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    M = M.cuda()
    # warm up
    _ = M @ M
    _ = M @ M
    _ = M @ M

    start.record()
    st = time()
    for i in range(num_iter):
        _ = M @ M
    print(f"time module Time: {time() - st} sec")
    end.record()
    torch.cuda.synchronize()
    print(f"torch (GPU) Time : {start.elapsed_time(end) / 1000} sec")
