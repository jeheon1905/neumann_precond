import torch
import numpy as np
import time
n = 10000
A = np.random.randn(n,n).astype(np.float32)
x = np.random.randn(n,n).astype(np.float32)

A_ = torch.from_numpy(A).to('cuda')
x_ = torch.from_numpy(x).to('cuda')
for i in range(100):
    b = A_ + x_

end = torch.cuda.Event(enable_timing=True)
start = torch.cuda.Event(enable_timing=True)
total = 0
start.record()
for i in range(1):
    b = torch.matmul(A_,x_)
end.record()
torch.cuda.synchronize()
total += start.elapsed_time(end)
total /= 1000
print(f'{total:0.2f} (1)')


A_ = A_.to('cpu')
x_ = x_.to('cpu')
st = time.time()
for i in range(1):
    b = torch.matmul(A_,x_)
print(f'{time.time()-st:0.2f} ({(time.time()-st)/total:0.2f})')


st = time.time()
for i in range(1):
    np.dot(A,x)
print(f'{time.time()-st:0.2f} ({(time.time()-st)/total:0.2f})')

