import numpy as np
import torch
from scipy_lobpcg import lobpcg
import time

def set_param(tol=None, maxiter=25, 
              verbosity=1, largest=False):
    param = {"tol"      : tol,
             "maxiter"  : maxiter,
             "largest"  : largest,
             "verbosityLevel"   : verbosity}
    return param

def generate_hamiltonian(n=512, matrix_type='diagnoal', sparsity=1e-1):
    # Diagonal dominant matrix
    if matrix_type == "diagonal":
        A = np.zeros((n, n))
        for i in range(0, n):
            A[i, i] = i - np.random.normal()
        noise = np.random.randn(n,n)
        noise[noise > 0] = 0
        noise = noise * sparsity
        A = A + sparsity * noise
        A = (A.T + A) / 2
    
    # Band dominant matrix
    elif matrix_type == "band":
        from scipy.sparse import diags
        _A = diags(
            [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)], [0, 1, -1]
        ).toarray()
        #noise = np.random.randn(n, n) * sparsity
        noise = np.random.randn(n,n)
        noise[noise > 0] = 0
        noise = noise * sparsity
        A = _A + (noise + noise.T) / 2
    else:
        raise NotImplementedError

    return A

def generate_init_guess(n, neig):
    guess_vector = np.random.randn(n, neig)
    guess_vector, _ = np.linalg.qr(guess_vector)
    return guess_vector

def do_ref(A, neig=10):
    import scipy
    print("================================================================")
    st = time.time()
    val = scipy.linalg.eigh(A, eigvals_only=True)
    print(f'lambda : {val[:neig]}')
    print(f'Done in {time.time()-st:0.2f} secs')
    print("----------------------------------------------------------------\n\n")
    return val[:neig]

def do(func, A, guess, param, B=None, M=None, name=''):
    print("================================================================")
    st = time.time()
    print(name)
    val, vec = func(A, guess, B=B, M=M, **param)
    print(f'lambda : {val}')
    print(f'Done in {time.time()-st:0.2f} secs')
    print("----------------------------------------------------------------\n\n")
    return val, vec

def gpu_warmup():
    A = torch.randn(1000,1000).to('cuda')
    for i in range(10):
        A @ A

if __name__=='__main__':
    import argparse, random, os


    np.set_printoptions(precision=6, suppress=True)
    torch.set_printoptions(precision=6, sci_mode=False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n','-n',type=int,default=5000)
    parser.add_argument('--alpha','-a',type=float,default=2.0)
    parser.add_argument('--seed','-s',type=int,default=1234)
    parser.add_argument('--type','-t',type=str,default='diagonal')
    parser.add_argument('--iter','-i',type=int,default=2000)
    parser.add_argument('--tol','-tol',type=float, default=0)
    parser.add_argument('--neig','-neig',type=int, default=5)
    opt = parser.parse_args()

    n = opt.n
    alpha = opt.alpha
    seed = opt.seed
    matrix_type = opt.type
    maxiter = opt.iter
    sparsity = 1e-4
    neig = opt.neig
    tol = opt.tol if opt.tol != 0 else None
    verbosity = 0
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #random.seed(seed)
    #
    #torch.use_deterministic_algorithms(True)

    lobpcg_params = set_param(tol=tol, maxiter=maxiter, verbosity=verbosity)
    A = generate_hamiltonian(n=n, matrix_type=matrix_type, sparsity=sparsity)
    #u,_ = np.linalg.qr(np.random.randn(n,10))
    #B = (u @ np.diag(np.random.randn(n)**2)) @ u.T
    B = np.eye(n)*0.005 + np.diag(np.random.randn(n)**2)    
    #M = np.diag(np.diag(A) ** -1)
    B = None
    M = None
    guess_vector = generate_init_guess(n, neig)
    #guess_vector = torch.load('X.pt')
    print(f"============================================")
    for k, v in lobpcg_params.items(): print(f"\t*{k}\t\t\t: {v}")
    print(f"\t*matrix shape\t\t\t: {n}")
    print(f"\t*num eigs\t\t\t: {neig}")
    print(f"\t*sparsity\t\t\t: {sparsity}")
    print(f"\t*matrix_type\t\t\t: {matrix_type}")
    print(f"============================================\n\n\n")
    
    
    from scipy_lobpcg import lobpcg as scipy_f
    from torch_lobpcg import lobpcg as torch_f
    
    #val, vec = do(scipy_f, A, guess_vector, lobpcg_params, B=B, M=M, name='scipy reference')
    dev = 'cuda'
    A = torch.from_numpy(A).to(dev)
    guess_vector = torch.from_numpy(guess_vector).to(dev)
    if B is not None:
        B = torch.from_numpy(B).to(dev)

    lobpcg_params['verbosityLevel'] = 1
    lobpcg_params['dynamic'] = False
    lobpcg_params['test'] = False
    lobpcg_params['tol'] = 0.00005811388300841897

    val1, vec1 = do(torch_f, A, guess_vector, lobpcg_params, B=B, M=M, name='torch GPU 64')
    print('Done1')

    lobpcg_params['verbosityLevel'] = 1
    lobpcg_params['dynamic'] = True
    lobpcg_params['test'] = False
    val2, vec2 = do(torch_f, A, guess_vector, lobpcg_params, B=B, M=M, name='torch GPU dyn')
    print('Done2')
    
