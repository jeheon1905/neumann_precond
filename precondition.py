from collections import OrderedDict
import torch
import numpy as np
from gospel.LinearOperator import LinearOperator
from gospel.Poisson.ISF_solver import ISF_solver
from gospel.util import timer
from gospel.util import to_cuda
from gospel.ParallelHelper import ParallelHelper as PH

# TODO: Remove MergePreconditioner
# TODO: Remove useless code comments

def create_preconditioner(precond_type=None, grid=None, use_cuda=False, options={}):
    """Create Preconditioner object from input parameters.

    :type  precond_type: str or None, optional
    :param precond_type:
        type of preconditioner, available type=[None, "jacobi", "filter", "poisson", "gapp", "shift-and-invert"]
    :type  grid: gospel.Grid or None, optional
    :param grid:
        Grid object

    *Example*
    >>> grid = Grid(atoms, gpts=(40, 40, 40))
    >>> precond = create_preconditioner("poisson", grid)
    """
    # if precond_type == "jacobi":
    #     return PreJacobi()
    if isinstance(precond_type, list):
        assert all(
            [isinstance(item, tuple) for item in precond_type]
        ), "If the type of precond_type is list, its item should be tuple "
        return MergePreconditioner(precond_type, grid, use_cuda)
    elif isinstance(precond_type, OrderedDict):
        precond_type = [ (key,*value) for key, value in precond_type.items() ]
        assert all(
            [isinstance(item, tuple) for item in precond_type]
        ), "If the type of precond_type is OrderedDict, its value should be tuple "
        return MergePreconditioner(precond_type, grid, use_cuda)
    ####################3
    if isinstance(precond_type, tuple):
        return MergePreconditioner(precond_type, grid, use_cuda)
    ####################3
    if precond_type == "poisson":
        return PrePoisson(grid, use_cuda=use_cuda, **options)
    elif precond_type == "gapp":
        ## Reference DOI: 10.1021/acs.jpca.2c09111
        if grid.get_pbc().sum() == 0:
            t_sample = [[1.38], [2.55], 11.96]
        elif grid.get_pbc().sum() == 1:
            t_sample = [[1.31], [2.3], 6.38]
        elif grid.get_pbc().sum() == 2:
            t_sample = [[1.48], [2.55], 8.39]
        else:
            # 3D-PBC
            t_sample = [[1.63], [2.73], 7.89]
        return PrePoisson(grid, t_sample=t_sample, use_cuda=use_cuda, **options)
    elif precond_type == "filter":
        return PreFilter(grid, use_cuda=use_cuda)
    elif precond_type == "shift-and-invert":
        return PreShiftAndInvert(grid, use_cuda=use_cuda, **options)
    elif precond_type == "Neumann":
        ## Reference DOI: 10.1021/acs.jpca.2c09111
        if grid.get_pbc().sum() == 0:
            t_sample = [[1.38], [2.55], 11.96]
        elif grid.get_pbc().sum() == 1:
            t_sample = [[1.31], [2.3], 6.38]
        elif grid.get_pbc().sum() == 2:
            t_sample = [[1.48], [2.55], 8.39]
        else:
            # 3D-PBC
            t_sample = [[1.63], [2.73], 7.89]
        return PreNeumann(grid,use_cuda=use_cuda, t_sample=t_sample, **options) 
    elif precond_type is None:
        return Preconditioner(None, use_cuda)
    else:
        raise NotImplementedError(f"{precond_type} is not available type.")


class Preconditioner:
    """parent class of preconditioners to accelerate convergence of iterative diagonalization.

    :type  precond_type: str or None, optional
    :param precond_type:
        available precondition types, defaults to None
        available precond_type
        - 'jacobi': inverse of diagonal elements of Hamiltonian matrix
        - 'poisson': solution of poisson equation of guess eigenvectors
        - 'filter': filtering using neighboring points
        - 'shift-and-invert': shift-and-invert preconditioning
        - 'Neumann': using neumann expansion for approximation (H-eI)^-1
    :type  use_cuda: bool, optional
    :param use_cuda:
        whether to use cuda, defaults to False
    """

    def __init__(self, precond_type, use_cuda=False):
        assert precond_type in [
            "jacobi",
            "poisson",
            "filter",
            "GKP",
            "shift-and-invert",
            "merge",
            "Neumann",
            None,
        ], f"{precond_type} is not supported precondition type."
        self._precond_type = precond_type
        self._device = PH.get_device(use_cuda)
        self._use_cuda = use_cuda
        self.num_called = 0  # this count number of calls in one scf step
        return

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type  : {self._precond_type}"
        s += "\n=====================================================================\n"
        return str(s)

    def call(self, residue):
        # no preconditioning
        return residue

    @timer
    def __call__(self, residue, H=None, eigval=None, i_iter=None, #eigvec=None
                 ):  # eigvec added
        self.num_called += 1
        if self._precond_type == "shift-and-invert" or self._precond_type == "Neumann":
            residue = self.call(residue, H, eigval) #eigvec added
        elif self._precond_type == "merge":
            residue = self.call(residue, H, eigval, i_iter)
        else:
            residue = self.call(residue)
        return residue

    def reset_num_called(self):
        self.num_called = 0

#    def set_device(self, device=None):
#        self._device = device
#        return

#    @property
#    def device(self):
#        return self._device


class MergePreconditioner(Preconditioner):
    def __init__(self, list_precond_types, grid, use_cuda=False):
        super().__init__("merge", use_cuda)
        self.preconditioners = []
        self.iterations = [[], []]
        #for precond_type, num_iter in list_precond_types:
        #    self.preconditioners.append( create_preconditioner(precond_type, grid, use_cuda) )
        #    self.iterations.append(num_iter)
    
        for precond_type, options, num_iter in list_precond_types:
            self.preconditioners.append(
                create_preconditioner(precond_type, grid, use_cuda, options)
            )

            assert type(num_iter)==int or type(num_iter)==list, "maxiter type should be int or list of int"
            if type(num_iter)==list: 
                assert len(num_iter)==2, "if list is given for maxiter, its length should be 2"
                self.iterations[0].append(num_iter[0])
                self.iterations[1].append(num_iter[1])
            else:
                self.iterations[0].append(num_iter)
                self.iterations[1].append(num_iter)


        self.iterations = np.cumsum(np.array(self.iterations), axis=-1)

    def call(self, residue, H, eigval, i_scf):
        iterations = self.iterations[1] if i_scf!=0 else self.iterations[0]
        try: 
            idx = iterations.tolist().index( min( iterations[(iterations-self.num_called) >0]  ) )
        except ValueError:
            idx=-1
            #return residue

        return self.preconditioners[idx](residue, H, eigval, i_scf)
class GKP(Preconditioner):
    """Gaussian Kernel Preconditioner"""

    def __init__(self, grid, t_sample1, t_sample2, nextra, use_cuda=False):
        super().__init__("GKP", use_cuda)
        self.__grid = grid
        self.__solver1 = ISF_solver(grid, t_sample1, fp="DP", device=self._device)
        self.__solver2 = ISF_solver(grid, t_sample2, fp="DP", device=self._device)
        self.__nextra = nextra  # the number of extra states
        self.linear_op = None
        return

    def call(self, residue):
        if self.linear_op is None:
            ps1 = self.__solver1
            ps2 = self.__solver2

            @timer
            def f(x):
                x = x.T  # shape=(nbands, ngpts)
                x1 = x[: -self.__nextra]
                x2 = x[-self.__nextra :]
                retval = torch.zeros_like(x)
                if x.dtype in [torch.complex64, torch.complex128]:
                    retval[: -self.__nextra] = (
                        ps1.compute_potential(x1.real)
                        + ps1.compute_potential(x1.imag) * 1j
                    )
                    retval[-self.__nextra :] = (
                        ps2.compute_potential(x2.real)
                        + ps2.compute_potential(x2.imag) * 1j
                    )
                elif x.dtype in [torch.float32, torch.float64]:
                    # retval = ps.compute_potential(x)
                    retval[: -self.__nextra] = ps1.compute_potential(x1)
                    retval[-self.__nextra :] = ps2.compute_potential(x2)
                else:
                    raise TypeError(f"x.dtype={x.dtype} is inappropriate.")
                return 2.0 * retval.T

            shape = (self.__grid.ngpts, self.__grid.ngpts)
            self.linear_op = LinearOperator(shape, f)
        return linear_op @ residue

#    def set_device(self, device=None):
#        self.__solver.set_device(device)
#        self._device = device
#        return


class PrePoisson(Preconditioner):
    """Poisson Preconditioner

    :type  grid: gospel.Grid or None, optional
    :param grid:
        Grid class object, defaults to None
    :type  t_sample: list or None
    :param t_sample:
        list of t values and weights and t_delta
    :type  use_cuda: bool
    :param use_cuda:
        using cuda device
    :type  fp: str
    :param fp:
        floating-point precision, choices=['DP', 'SP', 'MP'])
    """

    def __init__(self, grid, t_sample=None, use_cuda=False, fp="DP"):
        super().__init__("poisson", use_cuda)
        self.__grid = grid
        self.__fp = fp
        self.__solver = ISF_solver(grid, t_sample, fp=fp, device=self._device)
        self.linear_op = None
        return

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type  : poisson"
        s += f"\n* fp    : {self.__fp}"
        s += self.__solver.__str__()
        s += "\n=====================================================================\n"
        return str(s)

    def call(self, residue):
        """Poisson filter preconditioning.

        :rtype: LinearOperator
        :return:
            precondition operator
        """
        if self.linear_op is None:
            # NOTE: batch_compute_potential2 is fastest on GPU.
            # kernel = self.__solver.compute_potential
            # kernel = self.__solver.batch_compute_potential
            kernel = self.__solver.batch_compute_potential2

            @timer
            def f(x):
                # x = x.T  # x.T.shape=(nbands, ngpts)
                # x = x.T.contiguous()  # x.T.shape=(nbands, ngpts)
                if x.dtype in [torch.complex64, torch.complex128]:
                    retval = (
                        kernel(x.real) + kernel(x.imag) * 1j
                    )
                elif x.dtype in [torch.float32, torch.float64]:
                    retval = self.__solver.batch_compute_potential2(x)
                else:
                    raise TypeError(f"x.dtype={x.dtype} is inappropriate.")
                # return 2.0 * retval.T
                # return (2.0 * retval.T).contiguous()
                return (2.0 * retval).contiguous()

            shape = (self.__grid.ngpts, self.__grid.ngpts)
            self.linear_op = LinearOperator(shape, f)
        return self.linear_op @ residue


#    def set_device(self, device=None):
#        self.__solver.set_device(device)
#        self._device = device
#        return

# class PreJacobi(Preconditioner):
#    """Jacobi Preconditioner"""
#
#    def __init__(self):
#        super().__init__("jacobi")
#        self.linear_op = None
#        return
#
#    def call(self, residue, H):
#        """Jacobi preconditioning. Inverse of diagonal elements
#
#        :type  H: torch.Tensor or LinearOperator
#        :param H:
#            Hamiltonian operator
#
#        :rtype: LinearOperator or None
#        :return:
#            precondition operator
#        """
#        if hasattr(H, "diagonal"):
#            f = lambda x: (1.0 / H.diagonal()) * x
#            return LinearOperator(H.shape, f, dtype=H.dtype)
#        else:
#            print(
#                "Warning: Preconditioner is not used. ('diagonal' is not defined in H.)"
#            )
#            return None


class PreFilter(Preconditioner):
    """Low-pass filter preconditioner"""

    def __init__(self, grid, alpha=0.5, use_cuda=False):
        super().__init__("filter", use_cuda)
        self.__grid = grid
        self.__alpha = alpha
        self.__filter = self.make_filter(grid)
        self.__kernel = "sparse"  # type of kernel, options=["sparse", "conv"]
        # convolution version (conv) will also be implemented.
        self.__filter = to_cuda(self.__filter, self._device)
        self.linear_op = None
        return

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type  : filter"
        s += f"\n* alpha : {self.__alpha}"
        s += f"\n* kernel: {self.__kernel}"
        s += "\n=====================================================================\n"
        return str(s)

    def call(self, residue):
        """Low-pass filter preconditioning.

        :rtype: LinearOperator
        :return:
            precondition operator
        """
        if self.linear_op is None:

            @timer
            def f(x):
                if x.dtype in [torch.complex64, torch.complex128]:
                    retval = self.__filter @ x.real + self.__filter @ x.imag * 1j
                elif x.dtype in [torch.float32, torch.float64]:
                    retval = self.__filter @ x
                else:
                    raise TypeError(f"x.dtype={x.dtype} is inappropriate.")
                return retval

            shape = (self.__grid.ngpts, self.__grid.ngpts)
            self.linear_op = LinearOperator(shape, f)
        return self.linear_op @ residue

    def make_filter(self, grid):
        from scipy.sparse import identity, kron
        from gospel.util import scipy_to_torch_sparse

        fx = self.make_filter_axis(grid, 0)
        fy = self.make_filter_axis(grid, 1)
        fz = self.make_filter_axis(grid, 2)
        Ix = identity(grid.gpts[0])
        Iy = identity(grid.gpts[1])
        Iz = identity(grid.gpts[2])
        flter = kron(fx, kron(Iy, Iz)) + kron(Ix, kron(fy, Iz)) + kron(kron(Ix, Iy), fz)
        return scipy_to_torch_sparse(flter)

    def make_filter_axis(self, grid, axis):
        from scipy.sparse import diags
        import numpy as np

        pbc = grid.get_pbc()[axis]
        gpt = grid.gpts[axis]
        mat = diags(np.ones(gpt - 1), 1)
        if pbc:
            mat += diags(np.ones(1), gpt - 1)
        mat += mat.T
        mat *= (1 - self.__alpha) / 6
        mat += diags(np.ones(gpt) * self.__alpha / 3, 0)
        return mat


#    def set_device(self, device=None):
#        from gospel.util import to_cuda
#
#        self.__filter = to_cuda(self.__filter, device)
#        self._device = device
#        return


class PreShiftAndInvert(Preconditioner):
    def __init__(
        self,
        grid,
        use_cuda=False,
        #rtol=1e-5, # 
        rtol=0.25, # defalts setting
        max_iter=5,
        #max_iter=1000, # AX = B diff 
        correction_scale=0.1,
        no_shift_thr=10.0,
        #inner_precond="gapp",
        inner_precond=None,
        #inner_precond="Neumann",############added Neumann inner precond
        fp="DP",
        order="None" ##added
    ):
        super().__init__("shift-and-invert", use_cuda)
        # self.prev_solution = None
        self.rtol = rtol
        self.max_iter = max_iter
        self.correction_scale = correction_scale
        self.no_shift_thr = no_shift_thr
        self.inner_precond = inner_precond
        self.fp = fp  # TODO: Implementation
        self.order = order #added
        
        print("inner_precond = ", inner_precond,
              "pcg = ", max_iter ) ## added
        #
        # self.__precond_for_pcg = create_preconditioner(inner_precond, grid, use_cuda)
        # NOTE: Here, add 'fp' to create preconditioner (jeheon)

        #self.__precond_for_pcg = create_preconditioner(
        #        inner_precond, grid, use_cuda, options={"fp": fp},  original
        #)
 
        if inner_precond == "Neumann":
            self.__precond_for_pcg = create_preconditioner(
                inner_precond, grid, use_cuda, options={"fp": fp,
                                                        "order" : order, # added
                                                        },
            )
        else:
            self.__precond_for_pcg = create_preconditioner(
                inner_precond, grid, use_cuda, options={"fp": fp,
                                                        },
            )
        return

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type             : {self._precond_type}"
        s += f"\n* rtol             : {self.rtol}"
        s += f"\n* max_iter         : {self.max_iter}"
        s += f"\n* correction_scale : {self.correction_scale}"
        s += f"\n* no_shift_thr     : {self.no_shift_thr}"
        s += f"\n* inner_precond    : {self.inner_precond}"
        s += f"\n* fp               : {self.fp}"
        s += "\n=====================================================================\n"
        return str(s)

    def call(self, residue, H, eigval):
        orig_residue = residue.clone()
        residue = _pcg_sparse_solve(
            H,
            eigval,
            residue,
            # x0= (None if self.prev_solution is None else self.prev_solution),
            x0=None,
            preconditioner=self.__precond_for_pcg,
            rtol=self.rtol,
            max_iter=self.max_iter,
            correction_scale=self.correction_scale,
            no_shift_thr=self.no_shift_thr,
        )
        
        return residue


def blockPrint(func):
    import sys, os

    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, "w")
        result = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return result

    return wrapper


def _pcg_sparse_solve(
    H,
    eigval,
    b,
    x0=None,
    preconditioner=None,
    rtol=0.25,
    max_iter=300,
    correction_scale=0.1,
    no_shift_thr=10.0,
    # verbosityLevel=0,
    verbosityLevel=1,
):
    """
    Solve the linear equation, (H - eigval) x = b.

    Parameters
    ----------
e   :type H: torch.Tensor or LinearOperator
    :param H:
        The Hamiltonian opertor, usually a sparse matrix.
    :type eigval: torch.Tensor
    :param eigval:
        Eigenvalues. shape=(nbands,)
    :type b: torch.Tensor
    :param b:
        residual vectors. shape=(ngpts, nbands)
    :type x0: torch.Tensor, optional
    :param x0:
        initial for solution vectors with shape=(ngpts, nbands).
    :type preconditioner: torch.Tensor or LinearOperator, optional
    :param preconditioner:
        Preconditioner
    :type rtol: float, optional
    :param rtol:
        relative tolerance for convergence (tol=rtol * norm of b). The default is "rtol=1e-1".
    :type max_iter: int, optional
    :param max_iter:
        the number of maximum iteration. The default is "max_iter=300".
    :type correction_scale: float, optional
    :param correction_scale:
        the scale of perturbation on eigenvalues (for numerical stability). The default is "correction_scale=1.0".
    :type verbosityLevel: int, optional
    :param verbosityLevel:
        print intermediate solver output.  The default is "verbosityLevel=0".

    :rtype: torch.Tensor
    :return:
        solution vectors. shape=(ngpts, nbands)
    """

    ## Initialization
    if x0 is None:
        x0 = b.clone()
        print("_pcg_sparse_solve: x0 is initialized to b.")
    else:
        pass
    cumBlockSize = 0  # count the number of H operations
    b_norm = b.norm(dim=0, keepdim=True)

    ## Modify shift values
    perturb = -(b_norm.conj() * b_norm)
    eigval = eigval + correction_scale * perturb
    eigval[abs(perturb) > no_shift_thr] = 0.0  # no shift to states with large residues

    if preconditioner is not None:
        preconditioner = blockPrint(preconditioner)

    ## preconditioning of x0 (when x0=b)
    #x0[:,:] = preconditioner(x0) #memory issue # original code  
    x0[:,:] = preconditioner(x0, H , eigval) # memory issue 

    ## Compute residual and initialize search direction
    r = b - (H @ x0 - eigval * x0)
    cumBlockSize += b.size(1)
    if verbosityLevel > 0:
        print(f"(cg iter=1): res norm={r.norm(dim=0)}")
    if preconditioner is None:
        p = r.clone()
        z = r  # shallow copy
    else:
        #z = preconditioner(r)
        z = preconditioner(r,H,eigval)
        p = z.clone()

    rzold = torch.sum(r.conj() * z, dim=0, keepdim=True)

    is_convg = torch.zeros(b.size(1), dtype=torch.bool, device=b.device)
    for i in range(2, max_iter + 1):
        ## Compute step size
        Ap = H @ p[:, ~is_convg] - eigval[:, ~is_convg] * p[:, ~is_convg]
        alpha = rzold[:, ~is_convg] / torch.sum(
            p[:, ~is_convg].conj() * Ap, dim=0, keepdim=True
        )
        cumBlockSize += (~is_convg).sum()

        ## Update solution and residual
        x0[:, ~is_convg] = x0[:, ~is_convg] + alpha * p[:, ~is_convg]
        r[:, ~is_convg] = r[:, ~is_convg] - alpha * Ap
        del Ap  # for memory efficiency
        if preconditioner is not None:
            #z[:, ~is_convg] = preconditioner(r[:, ~is_convg])
            z[:,~is_convg] = preconditioner(r[:,~is_convg], H , eigval[:,~is_convg])  
        rznew = torch.sum(r[:, ~is_convg].conj() * z[:, ~is_convg], dim=0, keepdim=True)

        ## Check convergence
        r_norm = (
            rznew.real.sqrt() if preconditioner is None else r[:, ~is_convg].norm(dim=0)
        )
        is_convg2 = r_norm < rtol * b_norm[0][~is_convg]
        if torch.all(is_convg2):
            is_convg[~is_convg] = is_convg2
            break
        if verbosityLevel > 0:
            print(
                f"(cg iter={i}): {len(is_convg) - is_convg.sum()} remaining res norm={r_norm}"
            )

        ## Update search direction
        beta = rznew / rzold[:, ~is_convg]
        p[:, ~is_convg] = z[:, ~is_convg] + beta * p[:, ~is_convg]
        rzold[:, ~is_convg] = rznew
        is_convg[~is_convg] = is_convg2

    if verbosityLevel > 0:
        r_norm = rzold.real.sqrt() if preconditioner is None else r.norm(dim=0)
        if torch.all(is_convg):
            print(
                f"* PCG converged!\n"
                f"  - final iteration: {i}\n"
                f"  - res norm: {r_norm}\n"
                f"  - Cumulative # of blocks: {cumBlockSize}\n"
            )
        else:
            print(
                f"* PCG not converged.\n"
                f"  - final iteration: {i}\n"
                f"  - is_convg: {is_convg}\n"
                f"  - # of remaining: {len(is_convg) - is_convg.sum()}\n"
                f"  - res norm: {r_norm}\n"
                f"  - Cumulative # of blocks: {cumBlockSize}\n"
            )
    return -x0

class PreNeumann(Preconditioner):
    def __init__(
                 self, 
                 grid, 
                 use_cuda=False, 
                 order=3, 
                 t_sample=None, 
                 correction_scale=0.1,
                 no_shift_thr=10.0,
                 fp = "DP",
                 MAX_ORDER = 20,
                 error_cutoff = - 0.4

                 ):
        super().__init__("Neumann", use_cuda)
        self._precond_type = "Neumann"
        self.order = order
        print("order = ", order) #### added
        #assert type(order)==int
        #assert order>=0
        self.grid = grid
        self.correction_scale = correction_scale
        self.no_shift_thr = no_shift_thr
        self.MAX_ORDER = int(MAX_ORDER)
        self.error_cutoff = error_cutoff
        print("using cutoff =", error_cutoff)
        #ISF_solver 초기화
        self.ISF_solver = ISF_solver(
                grid = self.grid,
                )
        self.list_ISF_solvers = [ ISF_solver(grid=self.grid, t_sample=t_sample) for i in range(self.MAX_ORDER) ]
        self.list_ISF_solvers = [ ISF_solver(grid=self.grid,t_sample=t_sample) ] + self.list_ISF_solvers
        
        
#        self.list_ISF_solvers = [ ISF_solver(grid=self.grid) for i in range(self.MAX_ORDER) ]
#        self.list_ISF_solvers = [ ISF_solver(grid=self.grid) ] + self.list_ISF_solvers
        
        #self.list_ISF_solvers = [ISF_solver(grid=self.grid), ISF_solver(grid=self.grid, t_sample=t_sample) ]
    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type             : {self._precond_type}"
        s += f"\n* order            : {self.order}"
        s += "\n=====================================================================\n"
         
        return str(s)

################################################kinetic = H-eI approx
    def call(self, residue, H, eigval, 
             #eigvec  ## eigvec added  
             #,correction_scale, no_shift_thr):
            ):             
        correction_scale = self.correction_scale
        no_shift_thr=self.no_shift_thr
        residue_norm = residue.norm(dim=0, keepdim=True)
        print("eigval : ", eigval) ##added
        # Modify shift values
        perturb = -(residue_norm * residue_norm)
        eigval = eigval + correction_scale * perturb
        eigval[abs(perturb) > no_shift_thr] = 0.0  # no shift to states with large residues
        error_cutoff = self.error_cutoff
        self.H = H
        print("hamiltonian shape = ", H.shape)
        # 차원 맞추기
        if residue.ndim == 1:
            residue = residue.unsqueeze(0)

        print("residue :",residue)
        #residue = torch.real(residue)
        #print(f"residue dtype ={residue.dtype}, pi dtype = {torch_pi.dtype}")
        preconditioned_result = self.list_ISF_solvers[0].batch_compute_potential2(residue)/(2*np.pi)
        #preconditioned_result = self.ISF_solver.batch_compute_potential2(residue)/(2*np.pi)

        #cutoff setting and dynamic order 
        if self.order != 0:
            neumann_term = preconditioned_result.clone()
            if self.order == "dynamic":
                # check accuracy ## 이전 계산이 아니라 나머지 일때 키기 cutoff 만 보려면 꺼야함
                diff = (H @ preconditioned_result - preconditioned_result * eigval.reshape(1,-1)) - residue
                pre_error = torch.norm(diff, dim=0)/torch.norm(residue, dim=0)
                pre_result   = preconditioned_result.clone() ######## cutoff 만 계산할 때 켜야함
                error_log = torch.log10(pre_error)
#               print("test_error (using order = 0) = ", error_log )
                

                for order in range(1, self.MAX_ORDER + 1): ##########정확도 평가가 들어가는 것만 이렇게 표현
#               for order in range(self.order):
                    #Compute the next Neumann series 
                    #neumann_term = neumann_term - self.ISF_solver.batch_compute_potential2(H @ neumann_term - eigval.reshape(1,-1) * neumann_term)/(2*np.pi)
     
#                   neumann_term = neumann_term - self.list_ISF_solvers[1 + order].batch_compute_potential2(H @ neumann_term - eigval.reshape(1,-1) * neumann_term)/(2*np.pi)
                    neumann_term = neumann_term - self.list_ISF_solvers[order].batch_compute_potential2(H @ neumann_term - eigval.reshape(1,-1) * neumann_term)/(2*np.pi)
            
                    # Accumulate the result(always)
                    preconditioned_result += neumann_term
                    diff = (H @ preconditioned_result - preconditioned_result * eigval.reshape(1,-1)) - residue
                    error = torch.norm(diff, dim=0)/torch.norm(residue, dim=0)
                    error_log = torch.log10(error)
                    if pre_error.sum() > error.sum():
                        if error_cutoff >= torch.max(error_log):
                            pre_result = preconditioned_result.clone()
                            print("Preconditioned diagonalization error(log10) =", error_log, "(","using order(low_order_cutoff) = ", order,")")
                            break

                        elif error_cutoff < torch.max(error_log) and order == self.MAX_ORDER:
                            pre_result = preconditioned_result.clone()
                            print("Preconditioned diagonalization error(log10) =", error_log, "(","using order(high_error_cutoff) = ", order,")")
                            break

                        else:
                            pre_error = error
                            pre_result = preconditioned_result.clone()
                            continue
                    else:
                        preconditioned_result -= neumann_term
                        pre_result = preconditioned_result.clone()
                        print("Preconditioned diagonalization error(log10)=", torch.log10(pre_error), "(","using order(pre<now_break) = ", order - 1,")")
                        break
            else: 
                order = int(self.order)
                for order in range(1, order + 1): 
                    #Compute the next Neumann series 
                    #neumann_term = neumann_term - self.ISF_solver.batch_compute_potential2(H @ neumann_term - eigval.reshape(1,-1) * neumann_term)/(2*np.pi)
     
#                   neumann_term = neumann_term - self.list_ISF_solvers[1 + order].batch_compute_potential2(H @ neumann_term - eigval.reshape(1,-1) * neumann_term)/(2*np.pi)
                    neumann_term = neumann_term - self.list_ISF_solvers[order].batch_compute_potential2(H @ neumann_term - eigval.reshape(1,-1) * neumann_term)/(2*np.pi)
            
                    # Accumulate the result
                    preconditioned_result += neumann_term

                diff = (H @ preconditioned_result - preconditioned_result * eigval.reshape(1,-1)) - residue   #### 이거부터 아래 3줄은 테스트를 위한 것 
                error = torch.norm(diff, dim=0)/torch.norm(residue, dim=0)
                error_log = torch.log10(error)
                print(f"error (using order = {order}) = ", error_log)

            #inversion accuracy
#            print("Neumann result:", torch.linalg.norm(preconditioned_result, axis=0))
#            print("(H-eI)@precond result:", torch.linalg.norm( H@preconditioned_result - eigval.reshape(1,-1)*preconditioned_result , axis=0))
#            print("(H-eI)@precond - residue result:", torch.linalg.norm( H@preconditioned_result - eigval.reshape(1,-1)*preconditioned_result - residue , axis=0))
            print ("res norm (Neumann): ", torch.linalg.norm( H@preconditioned_result - eigval.reshape(1,-1)*preconditioned_result , axis=0))
            
            #return pre_result 
        #print ("res norm (Neumann): ", torch.linalg.norm( (H@preconditioned_result - eigval.reshape(1,-1)*preconditioned_result)@eigvec - eigvec , axis=0) )### eigvec added
            #return pre_result
            return preconditioned_result 
        else:
            return preconditioned_result

if __name__ == "__main__":
    from ase import Atoms
    from gospel.Grid import Grid

    grid = Grid(Atoms(cell=[3, 4, 5]), gpts=(100, 100, 100))
    precond = PreFilter(grid)
    print(precond)
    flter = precond.make_filter(grid)
    print(flter)

    nbands = 10
    R = torch.randn(grid.ngpts, nbands, dtype=torch.float64)
    R = precond(R)
