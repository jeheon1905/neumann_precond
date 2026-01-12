"""
Examples)

>>> python test.py --N 40 --matrix_type laplacian --dynamicTol 0.005 --neig 30 --sparsity 0.99 --dynamicMaxiter 10 --maxiter 10 --numActiveLowerBound 1.0 --verboseEvery 10000 --use_cuda True
>>> python test.py --N 10000 --is_sparse False --maxiter 10 --dynamicMaxiter 10 --neig 50 --use_cuda True
"""
import numpy as np
import torch
import time
import sys

sys.path.append("../")
from torch_lobpcg import lobpcg
from LinearOperator import LinearOperator

np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)
torch.backends.cuda.matmul.allow_tf32 = False
print(f"torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")


if __name__ == "__main__":
    import argparse

    str2bool = lambda x: True if x == "True" else False

    ## Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=0)
    parser.add_argument("--dynamicTol", type=float, default=0)
    parser.add_argument("--dynamicInit", type=str2bool, default=False)
    parser.add_argument("--staticMask", type=str2bool, default=True)
    parser.add_argument("--numActiveLowerBound", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--dynamicMaxiter", type=int, default=None)
    parser.add_argument("--verbosityLevel", type=int, default=1)
    parser.add_argument("--verboseEvery", type=int, default=100)

    parser.add_argument("--is_sparse", type=str2bool, default=True)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--matrix_type", type=str, default="diagonal")
    parser.add_argument("--neig", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sparsity", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--use_cuda", type=str2bool, default=False)
    parser.add_argument("--save", type=str, default="")
    opt = parser.parse_args()
    print(f"argparser: {opt}")

    ## Generate random symmetric matrix
    torch.manual_seed(opt.seed)
    print(f"============================================")
    from randSym import randSparseSym, randDenseSym

    st = time.time()
    if opt.is_sparse:
        A = randSparseSym(opt.N, opt.sparsity, opt.matrix_type, opt.alpha)
    else:
        A = randDenseSym(opt.N, opt.matrix_type, opt.alpha)
    B = None
    if opt.is_sparse:
        X = torch.randn(opt.N**3, opt.neig, dtype=torch.float64)
    else:
        X = torch.randn(opt.N, opt.neig, dtype=torch.float64)
    X = torch.linalg.qr(X)[0]
    print(f"Matrix initialization Time: {time.time() - st} sec")

    print(f"* A.shape: {A.shape}")
    print(f"* X.shape: {X.shape}")
    if opt.is_sparse:
        print(f"* sparsity: {1 - len(A.values()) / A.shape[0] / A.shape[1]}")
    print(f"* matrix_type: {opt.matrix_type}")

    if opt.use_cuda:
        print("using CUDA")
        if A.layout == torch.strided:
            A = A.cuda()
        elif opt.is_sparse:
            A = A.to_sparse_csr()
            print('AAAAAAAAAAAAAAAAAA')
            crow_indices = A.crow_indices().cuda()
            col_indices = A.col_indices().cuda()
            values = A.values().cuda()
            A = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=A.size())
        if B is not None:
            B = B.cuda()
        X = X.cuda()

        ## Warm-up
        _ = torch.randn(1000, 1000, dtype=torch.float64).cuda()
        for i in range(10):
            _ = _ @ _
        print("Warn-up is finished!!")
    if opt.is_sparse:
        A = A.to_sparse_csr()
    
    opA = lambda x : A @ x
    if A.layout == torch.strided:
        A32 = A.float()
    elif A.layout == torch.sparse_csr:
        crow_indices = A.crow_indices()
        col_indices = A.col_indices()
        values = A.values().float()
        A32 = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=A.size())
    opA32 = lambda x : A32 @ x
    LinOpA = LinearOperator(
                            A.shape, 
                            opA, 
                            dtype=torch.float64, 
                            matvec32=opA32, 
                            device=A.device
                            )

    print(f"============================================")
    if opt.save:
        retLambdaHistory = True
        retResidualNormsHistory = True
    else:
        retLambdaHistory = False
        retResidualNormsHistory = False

    ## Compare speed
    print(f"============================================")
    lobpcg_params = {
        "tol": opt.tol if opt.tol != 0 else None,
        "dynamicTol": opt.dynamicTol if opt.dynamicTol != 0 else None,
        "dynamicInit": opt.dynamicInit,
        "staticMask": opt.staticMask,
        "numActiveLowerBound": opt.numActiveLowerBound,
        "largest": False,
        "maxiter": opt.maxiter,
        "dynamicMaxiter": opt.dynamicMaxiter,
        "verbosityLevel": opt.verbosityLevel,
        "verboseEvery": opt.verboseEvery,
        "retLambdaHistory": retLambdaHistory,
        "retResidualNormsHistory": retResidualNormsHistory,
    }
    print(f"lobpcg_params = {lobpcg_params}")

    print("\n==========================================================")
    LinOpA.dtype = torch.float64
    st = time.time()
    out1 = lobpcg(A=LinOpA, B=B, X=X, dynamic=0, **lobpcg_params)
    print(f"FP64 Time: {time.time() - st} sec")
    print("==========================================================\n")

    print("\n==========================================================")
    LinOpA.dtype = torch.float64
    st = time.time()
    out2 = lobpcg(A=LinOpA, B=B, X=X, dynamic=1, **lobpcg_params)
    print(f"dynamic1 Time: {time.time() - st} sec")
    print("==========================================================\n")
    
    print("\n==========================================================")
    LinOpA.dtype = torch.float64
    st = time.time()
    out3 = lobpcg(A=LinOpA, B=B, X=X, dynamic=2, **lobpcg_params)
    print(f"dynamic2 Time: {time.time() - st} sec")
    print("==========================================================\n")
    
    print("\n==========================================================")
    LinOpA.dtype = torch.float64
    st = time.time()
    out4 = lobpcg(A=LinOpA, B=B, X=X, dynamic=3, **lobpcg_params)
    print(f"dynamic3 Time: {time.time() - st} sec")
    print("==========================================================\n")

    print("\n==========================================================")
    LinOpA.dtype = torch.float32
    st = time.time()
    out4 = lobpcg(A=LinOpA, B=B, X=X.float(), dynamic=0, **lobpcg_params)
    print(f"FP32 Time: {time.time() - st} sec")
    print("==========================================================\n")

    if opt.save:
        outs = [out1, out2, out3, out4]
        lambda_history = [o[2] for o in outs]
        resnorm_history = [o[3] for o in outs]
        data = {'lambda':lambda_history, 'resnorm':resnorm_history}
        torch.save(data, f"{opt.save}_{opt.seed}.pt")
        print(f"saved data {opt.save}")
