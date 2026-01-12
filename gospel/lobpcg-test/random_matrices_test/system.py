"""
Examples)

>>> python system.py --neig --maxiter 10 --dynamicMaxiter 10
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

    parser.add_argument("--neig", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_cuda", type=str2bool, default=False)
    parser.add_argument("--device_number", type=int, default=0)
    opt = parser.parse_args()
    print(f"argparser: {opt}")

    ## Generate random symmetric matrix
    torch.manual_seed(opt.seed)
    print(f"============================================")
    from randSym import randSparseSym, randDenseSym

    st = time.time()

    import pickle
    # with open("system_pickle_files/diamond_111.pkl", "rb") as f:
    with open("system_pickle_files/diamond_222.pkl", "rb") as f:
    # with open("system_pickle_files/diamond_221.pkl", "rb") as f:
        data = pickle.load(f)
    T_xx = data["T_xx"]
    T_yy = data["T_yy"]
    T_zz = data["T_zz"]
    P_I = data["P"]
    D_I = data["D"]
    I_I = data["I"]
    V_loc = data["V_loc"].reshape(-1, 1)
    T_xx = torch.from_numpy(T_xx)
    T_yy = torch.from_numpy(T_yy)
    T_zz = torch.from_numpy(T_zz)
    V_loc = torch.from_numpy(V_loc)
    for i_atom in range(len(P_I)):
        if opt.use_cuda:
            P_I[i_atom] = torch.from_numpy(P_I[i_atom]).cuda(opt.device_number)
            D_I[i_atom] = torch.from_numpy(D_I[i_atom]).cuda(opt.device_number)
            I_I[i_atom] = torch.from_numpy(I_I[i_atom]).cuda(opt.device_number)
        else:
            P_I[i_atom] = torch.from_numpy(P_I[i_atom])
            D_I[i_atom] = torch.from_numpy(D_I[i_atom])
            I_I[i_atom] = torch.from_numpy(I_I[i_atom])
    gpts = (len(T_xx), len(T_yy), len(T_zz))
    N = gpts[0] * gpts[1] * gpts[2]

    T_xx_32 = T_xx.float()
    T_yy_32 = T_yy.float()
    T_zz_32 = T_zz.float()
    V_loc_32 = V_loc.float()
    P_I_32 = [P_I[i_atom].float() for i_atom in range(len(P_I))]
    D_I_32 = [D_I[i_atom].float() for i_atom in range(len(D_I))]

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
        V_NL = torch.zeros_like(x)
        for (D, P, I) in zip(D_I, P_I, I_I):
            V_NL[I] += P.conj().T @ (D @ (P @ x[I]))
        retval = T + V_loc * x + V_NL
        return retval

    def H32(x):
        x = x.to(torch.float32)
        _x = x.reshape(*gpts, -1)
        T = (
            tensordot(T_xx_32, _x, ([0], [0]))
            + tensordot(T_yy_32, _x, ([0], [1])).permute([1, 0, 2, 3])
            + tensordot(T_zz_32, _x, ([0], [2])).permute([1, 2, 0, 3])
        ).reshape(np.prod(gpts), -1)
        V_NL = torch.zeros_like(x)
        for (D, P, I) in zip(D_I_32, P_I_32, I_I):
            V_NL[I] += P.conj().T @ (D @ (P @ x[I]))
        retval = (T + V_loc_32 * x + V_NL).to(torch.float64)
        return retval

    def real_H32(x):
        _x = x.reshape(*gpts, -1)
        T = (
            tensordot(T_xx_32, _x, ([0], [0]))
            + tensordot(T_yy_32, _x, ([0], [1])).permute([1, 0, 2, 3])
            + tensordot(T_zz_32, _x, ([0], [2])).permute([1, 2, 0, 3])
        ).reshape(np.prod(gpts), -1)
        V_NL = torch.zeros_like(x)
        for (D, P, I) in zip(D_I_32, P_I_32, I_I):
            V_NL[I] += P.conj().T @ (D @ (P @ x[I]))
        retval = (T + V_loc_32 * x + V_NL)
        return retval

    # A = LinearOperator((N, N), H, dtype=torch.float64, matvec32=H32)
    A = LinearOperator((N, N), H, dtype=torch.float64, matvec32=real_H32)
    A.device = T_xx.device
    B = None
    X = torch.randn(N, opt.neig, dtype=torch.float64)
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

    print("\n==========================================================")
    st = time.time()
    # val1, vec1 = lobpcg(A=A, B=B, X=X, dynamic=0, **lobpcg_params)
    val1, vec1, his1 = lobpcg(A=A, B=B, X=X, dynamic=0, **lobpcg_params, retLambdaHistory=True)
    print(f"no dynamic Time: {time.time() - st} sec")
    print("==========================================================\n")

    print("\n==========================================================")
    st = time.time()
    # val2, vec2 = lobpcg(A=A, B=B, X=X, dynamic=1, **lobpcg_params)
    val2, vec2, his2 = lobpcg(A=A, B=B, X=X, dynamic=1, **lobpcg_params, retLambdaHistory=True)
    print(f"dynamic Time: {time.time() - st} sec")
    print("==========================================================\n")

    A.dtype = torch.float64
    print("\n==========================================================")
    X = X.float()
    st = time.time()
    # val3, vec3 = lobpcg(A=A, B=B, X=X, dynamic=4, **lobpcg_params)
    val3, vec3, his3 = lobpcg(A=A, B=B, X=X, dynamic=4, **lobpcg_params, retLambdaHistory=True)
    print(f"dynamic G4 Time: {time.time() - st} sec")
    print("==========================================================\n")

    A.dtype = torch.float32
    print("\n==========================================================")
    X = X.float()
    st = time.time()
    # val4, vec4 = lobpcg(A=A, B=B, X=X, dynamic=0, **lobpcg_params)
    val4, vec4, his4= lobpcg(A=A, B=B, X=X, dynamic=0, **lobpcg_params, retLambdaHistory=True)
    print(f"FP32 Time: {time.time() - st} sec")
    print("==========================================================\n")

    # lambda_history = [his1, his2, his3, his4]
    # filename = f"Diamond_221_seed{opt.seed}.pt"
    # filename = f"Diamond_222_seed{opt.seed}.pt"
    # filename = f"Diamond_111_seed{opt.seed}.pt"
    # filename = f"Diamond_111_seed{opt.seed}_tol.pt"
    # torch.save(lambda_history, filename)
    # print(f"{filename} is written!")
