"""
Examples)

>>> python laplacian_test.py --N 40 --neig 40 --dynamicTol 0.0001 --maxiter 1000 --verboseEvery 10 --use_cuda True
"""
import numpy as np
import torch
import time
import sys
from ase import Atoms
from gospel.Grid import Grid
from gospel.FdOperators import make_kinetic_op
# from gospel.LinearOperator import LinearOperator
from gospel.util import tensordot

sys.path.append("../")
from torch_lobpcg import lobpcg
from LinearOperator import LinearOperator

np.set_printoptions(precision=9, suppress=True)
torch.set_printoptions(precision=9, sci_mode=False)
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
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--dynamicMaxiter", type=int, default=None)
    parser.add_argument("--verbosityLevel", type=int, default=1)
    parser.add_argument("--verboseEvery", type=int, default=100)

    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--neig", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--use_cuda", type=str2bool, default=False)
    parser.add_argument("--device_number", type=int, default=0)

    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--fp64", type=str2bool, default=True)
    parser.add_argument("--dynamicTest", type=str2bool, default=True)
    opt = parser.parse_args()
    print(f"argparser: {opt}")

    ## Generate random symmetric matrix
    torch.manual_seed(opt.seed)
    print(f"============================================")

    st = time.time()
    cell = np.array([0.2, 0.2, 0.2]) * opt.N
    gpts = (opt.N, opt.N, opt.N)
    grid = Grid(atoms=Atoms(cell=cell), gpts=gpts)
    T_xx, T_yy, T_zz = make_kinetic_op(grid, combine=False, as_sparse=False)
    #T_xx = torch.from_numpy(T_xx)
    #T_yy = torch.from_numpy(T_yy)
    #T_zz = torch.from_numpy(T_zz)
    V_loc = opt.alpha * torch.randn(opt.N**3, dtype=torch.float64).reshape(-1, 1)
    T_xx_32 = T_xx.float()
    T_yy_32 = T_yy.float()
    T_zz_32 = T_zz.float()
    V_loc_32 = V_loc.float()

    if opt.use_cuda:
        T_xx = T_xx.cuda(opt.device_number)
        T_yy = T_yy.cuda(opt.device_number)
        T_zz = T_zz.cuda(opt.device_number)
        V_loc = V_loc.cuda(opt.device_number)
        T_xx_32 = T_xx_32.cuda(opt.device_number)
        T_yy_32 = T_yy_32.cuda(opt.device_number)
        T_zz_32 = T_zz_32.cuda(opt.device_number)
        V_loc_32 = V_loc_32.cuda(opt.device_number)

    def H(x):
        _x = x.reshape(*gpts, -1)
        T = (
            tensordot(T_xx, _x, ([0], [0]))
            + tensordot(T_yy, _x, ([0], [1])).permute([1, 0, 2, 3])
            + tensordot(T_zz, _x, ([0], [2])).permute([1, 2, 0, 3])
        ).reshape(np.prod(gpts), -1)
        retval = T# + V_loc * x
        return retval

    def H32(x):
        st = time.time()
        _x = x.reshape(*gpts, -1)
        T = (
            tensordot(T_xx_32, _x, ([0], [0]))
            + tensordot(T_yy_32, _x, ([0], [1])).permute([1, 0, 2, 3])
            + tensordot(T_zz_32, _x, ([0], [2])).permute([1, 2, 0, 3])
        ).reshape(np.prod(gpts), -1)
        retval = T# + V_loc_32 * x
        return retval
    
    A = LinearOperator(
            (opt.N**3, opt.N**3), 
            H, 
            dtype=torch.float64, 
            matvec32=H32,
            device=T_xx.device)
    B = None
    X = torch.randn(opt.N**3, opt.neig, dtype=torch.float64)
    X = torch.linalg.qr(X)[0]
    if opt.use_cuda:
        X = X.cuda(opt.device_number)
    print(f"Matrix initialization Time: {time.time() - st} sec")

    print(f"* A.shape: {A.shape}")
    print(f"* X.shape: {X.shape}")

    if opt.use_cuda:
        ## Warm-up
        _ = torch.randn(1000, 1000, dtype=torch.float64).cuda(opt.device_number)
        for i in range(10):
            _ = _ @ _
        print("Warn-up is finished!!")
    print(f"============================================")

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
        
    }
    print(f"lobpcg_params = {lobpcg_params}")

    if opt.fp64:
        print("\n==========================================================")
        st = time.time()
        val_, vec_, eig, res = lobpcg(A=A, B=B, X=X, 
                retLambdaHistory=True,
                retResidualNormsHistory=True,
                dynamic=0, 
                **lobpcg_params)
        print(f"FP64 Time: {time.time() - st} sec")
        print("==========================================================\n")
        if opt.save:
            torch.save(eig,opt.save+"_eigval_reference.pt")
            torch.save(res,opt.save+"_resnorm_reference.pt")

    else:
        A.dtype = torch.float64
        print("\n==========================================================")
        st = time.time()
        val_, vec_, eig1, res1, res11 = lobpcg(A=A, B=B, X=X, 
        #_ = lobpcg(A=A, B=B, X=X, 
                retLambdaHistory=True,
                retResidualNormsHistory=True,
                dynamicTest=opt.dynamicTest,
                dynamic=1, 
                **lobpcg_params)
        print(f"dynamic1 Time: {time.time() - st} sec")
        print("==========================================================\n")

        A.dtype = torch.float64
        print("\n==========================================================")
        st = time.time()
        val_, vec_, eig2, res2, res22 = lobpcg(A=A, B=B, X=X, 
        #_ = lobpcg(A=A, B=B, X=X, 
                retLambdaHistory=True,
                retResidualNormsHistory=True,
                dynamicTest=opt.dynamicTest,
                dynamic=4, 
                **lobpcg_params)
        print(f"dynamic4 Time: {time.time() - st} sec")
        print("==========================================================\n")
        if opt.save:
            eig = [eig1, eig2]
            res = [res1, res11, res2, res22]

            torch.save(eig,opt.save+"_eigval.pt")
            torch.save(res,opt.save+"_resnorm.pt")
