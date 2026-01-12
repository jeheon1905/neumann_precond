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
    import sys
    np.random.seed(int(sys.argv[1]))
    np.set_printoptions(precision=6, suppress=True)
    torch.set_printoptions(precision=6, sci_mode=False)
    n = 5000
    tol = None
    maxiter = 2000
    neig = 3
    verbosity = 0
    sparsity = 1e-4
    matrix_type = 'diagonal'

    lobpcg_params = set_param(tol=tol, maxiter=maxiter, verbosity=verbosity)
    A = generate_hamiltonian(n=n, matrix_type=matrix_type, sparsity=sparsity)
    #u,_ = np.linalg.qr(np.random.randn(n,10))
    #B = (u @ np.diag(np.random.randn(n)**2)) @ u.T
    #B = np.diag(np.random.randn(n)**2)    
    #M = np.diag(np.diag(A) ** -1)
    B = None
    M = None
    guess_vector = generate_init_guess(n, neig)
    #guess_vector = torch.load('X.pt')
    print(f"============================================")
    print(f'seed={sys.argv[1]}')
    for k, v in lobpcg_params.items(): print(f"\t*{k}\t\t\t: {v}")
    print(f"\t*matrix shape\t\t\t: {n}")
    print(f"\t*num eigs\t\t\t: {neig}")
    print(f"\t*sparsity\t\t\t: {sparsity}")
    print(f"\t*matrix_type\t\t\t: {matrix_type}")
    print(f"============================================\n\n\n")
    
    
    from scipy_lobpcg import lobpcg as scipy_f
    from torch_lobpcg import lobpcg as torch_f
    A32 = A.astype(np.float32)
    guess_vector32 = guess_vector.astype(np.float32)
    alpha = 0.8
    
    import os
    if not os.path.isfile('valvec.pt'):
        val, vec = do(scipy_f, A, guess_vector, lobpcg_params, B=B, M=M, name='scipy 64')
        val = torch.from_numpy(val)
        vec = torch.from_numpy(vec)
        torch.save([val,vec],'valvec.pt')
    else:
        val, vec = torch.load('valvec.pt')
    #lobpcg_params['tol'] = alpha*(1e-15)**(1/4)*n
    #val1, vec1 = do(scipy_f, A32, guess_vector32, lobpcg_params, B=B, M=M, name='scipy 32')
    #lobpcg_params['tol'] = None
    #val2, vec2 = do(torch_f, A, guess_vector, lobpcg_params, B=B, M=M, name='torch CPU 64')
    #lobpcg_params['dynamic'] = True
    #lobpcg_params['tol'] = None
    #lobpcg_params['tol_'] = alpha * n * (1e-15)**(1/4)
    #val3, vec3 = do(torch_f, A, guess_vector, lobpcg_params, B=B, M=M, name='torch CPU dyn')
    #lobpcg_params['dynamicTol'] = alpha * n * (1e-15)**(1/4)
    #lobpcg_params['test'] = True
    A = torch.from_numpy(A)
    A32 = torch.from_numpy(A32)
    guess_vector = torch.from_numpy(guess_vector)
    guess_vector32 = torch.from_numpy(guess_vector32)


    #val3, vec3 = do(torch_f, A, guess_vector, lobpcg_params, B=B, M=M, name='torch CPU 64')
    #msg1 = ''
    #lobpcg_params['dynamic'] = True
    #lobpcg_params['tol'] = None
    #val4, vec4 = do(torch_f, A, guess_vector, lobpcg_params, B=B, M=M, name='torch CPU dyn32')
    #msg1 = ''
    #except:
    #    val4 = 9999
    #    msg1 = 'dyn32 Failed'
        


    lobpcg_params['dynamic'] = False
    lobpcg_params['tol'] = (1e-15)**(1/4)*n
    #lobpcg_params['dynamicTol'] = None
    #lobpcg_params['test'] = False
    val5, vec5 = do(torch_f, A32, guess_vector32, lobpcg_params, B=B, M=M, name='torch CPU 32')
    msg2 = ''
    #except:
    #    val5 = 9999
    #    msg2 = 'fp32 Failed'
    #lobpcg_params['dynamic'] = False
    #lobpcg_params['tol'] = None
    #val6, vec6 = do(torch_f,A, vec4.to(torch.float64), lobpcg_params, B=B, M=M, name='torch CPU32+CPU64')

    #val1 = torch.from_numpy(val1)
    #vec1 = torch.from_numpy(vec1)

    #print(torch.abs(val1 - val))
    #print(torch.abs(val2 - val))
    #print(torch.abs(val3 - val))
    print(torch.abs(val4 - val))
    print(torch.abs(val5 - val))
    print(msg1)
    print(msg2+'\n')
    #print(torch.abs(val6 - val))
    
    
    for i in range(10):
        gpu_warmup()

    lobpcg_params['dev'] = 'cuda'
    lobpcg_params['dynamic'] = False
    val3, vec3 = do(torch_f, A, guess_vector, lobpcg_params, B=B, M=M, name='torch GPU 64')
    lobpcg_params['dynamic'] = True
    val4, vec4 = do(torch_f, A, guess_vector, lobpcg_params, B=B, M=M, name='torch GPU dyn')

    torch.set_printoptions(precision=10, sci_mode=False)
    print(torch.abs(val1 - val))
    print(torch.abs(val2 - val))
    print(torch.abs(val3.to('cpu') - val))
    print(torch.abs(val4.to('cpu') - val))
